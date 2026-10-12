# -*- coding: utf-8 -*-
"""
data_load.py
============

数据加载与基础清洗模块。

职责：读取四个 CSV（dynamic / label / static / drug），完成基础清洗
（全空行删除、主键缺失删除、日期解析、主键去重、极端值标记、药物聚合与
合并、按年份拆分上下游），并统一保存上游、下游预处理前的数据集
（.pt 与 .csv 双份，01_upstream_raw / 02_downstream_raw）。

设计约定：
- 所有清洗函数不在内部保存文件，统一由 run_data_load 保存；
- 每个函数写清 docstring，逐步打印状态，格式 [STEP LOAD.X] 描述... 结果；
- 不做缺失值插补：除主键缺失和全空行外，所有缺失保留；
- 文件不存在、列缺失时抛出明确错误。

依赖: os, logging, datetime, pandas, numpy, torch
"""

import os
import logging
import pandas as pd
import numpy as np
import torch

logger = logging.getLogger(__name__)

# 定性（枚举）特征列：去重时取 first
CATEGORIC_FEATURE_COLS = {
    "visit_type", "urine_prot_qual", "urine_bld", "urine_glu", "urine_leuk",
}

# 临床合理范围（来自 data_dictionary.csv），越界值标记为 NaN
EXTREME_RANGES = {
    "u24pro": (0.1, 20.0),
    "glu_f": (3.0, 15.0),
    "PLA2R_ab": (0, 2000),
    "WBC": (2.0, 25.0),
    "HGB": (60, 180),
    "PLT": (30, 450),
    "NEUT_pct": (20, 90),
    "LYMPH_pct": (10, 70),
    "ALT": (5, 250),
    "AST": (5, 250),
    "TBIL": (2, 80),
    "salb": (15.0, 55.0),
    "stp": (40.0, 85.0),
    "SCR": (40, 800),
    "BUN": (2.5, 35.0),
    "UA": (120, 600),
    "TC": (2.0, 12.0),
    "TG": (0.3, 6.0),
    "LDL_C": (0.5, 8.0),
    "HDL_C": (0.6, 2.5),
    "urine_sg": (1.005, 1.035),
}

REQUIRED_COLUMNS = {
    "dynamic": ["patient_id", "visit_date"],
    "label": ["patient_id", "visit_date", "label"],
    "static": ["patient_id", "birth_date"],
    "drug": ["patient_id", "visit_date", "drug_name"],
}


def _is_categorical(series: pd.Series) -> bool:
    """判断某列是否为类别列（枚举），用于去重聚合策略。"""
    if series.name in CATEGORIC_FEATURE_COLS:
        return True
    try:
        pd.to_numeric(series, errors="raise")
        return False
    except (ValueError, TypeError):
        return True


def load_raw_tables(cfg: dict) -> dict:
    """读取四个 CSV 文件。

    输入:
        cfg: 配置字典，需含 data_dir 与各文件名。
    返回:
        dict: {"dynamic","label","static","drug"} 四个 DataFrame。
    """
    data_dir = cfg["data_dir"]
    tables = {}
    for name, fname in [("dynamic", cfg["dynamic_file"]),
                        ("label", cfg["label_file"]),
                        ("static", cfg["static_file"]),
                        ("drug", cfg["drug_file"])]:
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"输入文件不存在: {path}")
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        missing = [c for c in REQUIRED_COLUMNS[name] if c not in df.columns]
        if missing:
            raise ValueError(f"{fname} 缺少必需列: {missing}")
        tables[name] = df
        logger.info("[STEP LOAD.1] 读取 %s: 行数 %d, 列数 %d, 列名 %s",
                    fname, len(df), len(df.columns), list(df.columns))
    return tables


def clean_empty_rows(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """删除完全空白行。

    输入:
        df: 待清洗 DataFrame；name: 表名。
    返回:
        删除全空行后的 DataFrame。
    """
    before = len(df)
    df = df.dropna(how="all")
    logger.info("[STEP LOAD.2] 删除完全空白行 %s: %d -> %d", name, before, len(df))
    return df


def clean_key_columns(df: pd.DataFrame, name: str, key_cols) -> pd.DataFrame:
    """删除主键（patient_id / visit_date 等）缺失的行。

    输入:
        df: 待清洗 DataFrame；name: 表名；key_cols: 主键列名列表。
    返回:
        主键非空行的 DataFrame。
    """
    before = len(df)
    mask = df[key_cols].apply(lambda s: s.str.strip() if s.dtype == "object" else s)
    mask = mask.apply(lambda s: s != "").all(axis=1) & df[key_cols].notna().all(axis=1)
    df = df[mask].copy()
    dropped = before - len(df)
    logger.info("[STEP LOAD.3] 删除主键缺失行 %s(%s): 删除 %d 行", name, key_cols, dropped)
    return df


def parse_dates(df: pd.DataFrame, date_cols) -> pd.DataFrame:
    """解析日期列。

    输入:
        df: DataFrame；date_cols: 需解析的日期列名列表。
    返回:
        日期列已转为 datetime 的 DataFrame。
    """
    success, failed = 0, 0
    for c in date_cols:
        if c not in df.columns:
            logger.warning("[STEP LOAD.4] 日期列 %s 不存在，跳过", c)
            continue
        parsed = pd.to_datetime(df[c], errors="coerce", format="mixed", dayfirst=False)
        ok = parsed.notna()
        success += int(ok.sum())
        failed += int((~ok).sum())
        df[c] = parsed
    logger.info("[STEP LOAD.4] 解析日期列 %s: 成功 %d, 失败 %d", list(date_cols), success, failed)
    return df


def deduplicate_by_key(df: pd.DataFrame, key_cols, name: str) -> pd.DataFrame:
    """按主键去重：数值列取均值，类别列取首次。

    输入:
        df: DataFrame；key_cols: 主键列；name: 表名。
    返回:
        去重后的 DataFrame。
    """
    before = len(df)
    dup_rows = int(df.duplicated(subset=key_cols, keep=False).sum())
    if dup_rows == 0:
        df = df.drop_duplicates(subset=key_cols, keep="first")
        logger.info("[STEP LOAD.5] 主键去重 %s: 无重复, %d 行", name, len(df))
        return df
    agg = {}
    for c in df.columns:
        if c in list(key_cols):
            continue
        agg[c] = "first" if _is_categorical(df[c]) else "mean"
    df = df.groupby(list(key_cols), as_index=False).agg(agg)
    logger.info("[STEP LOAD.5] 主键去重 %s: %d -> %d (重复 %d)", name, before, len(df), dup_rows)
    return df


def mark_extreme_values(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """按临床合理范围将极端异常值标记为 NaN。

    输入:
        df: DataFrame；cfg: 配置字典。
    返回:
        极端值被替换为 NaN 的 DataFrame。
    """
    per_col = {}
    total = 0
    for col, (lo, hi) in EXTREME_RANGES.items():
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        out = (s < lo) | (s > hi)
        df[col] = s.mask(out)  # 无论是否有越界，统一转为数值类型
        n = int(out.sum())
        if n > 0:
            per_col[col] = n
            total += n
    logger.info("[STEP LOAD.6] 极端值标记: 共 %d, 各指标 %s", total, per_col)
    return df


def aggregate_drug_table(drug_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """按 patient_id + visit_date + drug_name 聚合药物表。

    剂量相加，记录 n_dose；药物名中空格替换为下划线。
    输入:
        drug_df: 原始药物表；cfg: 配置字典。
    返回:
        聚合后的药物表（patient_id, visit_date, drug_name, dose_mg, n_dose）。
    """
    before = len(drug_df)
    df = drug_df.copy()
    df["drug_name"] = df["drug_name"].astype(str).str.strip().str.replace(r"\s+", "_", regex=True)
    df["dose_mg"] = pd.to_numeric(df["dose_mg"], errors="coerce")
    df = df.groupby(["patient_id", "visit_date", "drug_name"], as_index=False).agg(
        dose_mg=("dose_mg", "sum"), n_dose=("dose_mg", "size"))
    logger.info("[STEP LOAD.7] 药物聚合: %d -> %d 行", before, len(df))
    return df


def merge_dynamic_drug(dynamic_df: pd.DataFrame, drug_df: pd.DataFrame) -> pd.DataFrame:
    """把聚合药物表按 (patient_id, visit_date) 左连接进动态宽表。

    每个 visit 聚合为一个 drug_list（元素为 (drug_name, dose_mg)）。
    输入:
        dynamic_df: 动态宽表；drug_df: 聚合后的药物表。
    返回:
        新增 drug_list 列的动态宽表。
    """
    rows = []
    for (pid, vd), g in drug_df.groupby(["patient_id", "visit_date"]):
        rows.append((pid, vd, list(zip(g["drug_name"], g["dose_mg"]))))
    agg = pd.DataFrame(rows, columns=["patient_id", "visit_date", "drug_list"])
    merged = dynamic_df.merge(agg, on=["patient_id", "visit_date"], how="left")
    merged["drug_list"] = merged["drug_list"].apply(lambda x: x if isinstance(x, list) else [])
    matched = int(merged["drug_list"].apply(len).gt(0).sum())
    logger.info("[STEP LOAD.8] 合并药物: 匹配到药物的 visit %d", matched)
    return merged


def split_upstream_downstream(dynamic_df: pd.DataFrame, cfg: dict):
    """按年份把动态宽表拆分为上游（2012-2022）与下游（2023-2024）。

    输入:
        dynamic_df: 动态宽表；cfg: 配置字典。
    返回:
        上游宽表、下游宽表。
    """
    start_up = pd.Timestamp(cfg["upstream_start"])
    end_up = pd.Timestamp(cfg["upstream_end"])
    start_dn = pd.Timestamp(cfg["downstream_start"])
    end_dn = pd.Timestamp(cfg["downstream_end"])
    vd = dynamic_df["visit_date"]
    up = dynamic_df[(vd >= start_up) & (vd <= end_up)]
    dn = dynamic_df[(vd >= start_dn) & (vd <= end_dn)]
    logger.info("[STEP LOAD.9] 上下游拆分: 上游患者 %d / visit %d; 下游患者 %d / visit %d",
                up["patient_id"].nunique(), len(up),
                dn["patient_id"].nunique(), len(dn))
    return up.copy(), dn.copy()


def df_to_pt_and_csv(df: pd.DataFrame, pt_path: str, csv_path: str, cfg: dict) -> None:
    """把一个 DataFrame 同时保存为 .pt 与 .csv。

    .pt: torch.save 保存 DataFrame 对象，训练时 torch.load 可直接还原。
    .csv: 直接可读的表格。
    输入:
        df: 待保存 DataFrame；pt_path/csv_path: 保存路径；cfg: 配置字典。
    """
    os.makedirs(os.path.dirname(pt_path), exist_ok=True)
    torch.save(df, pt_path)
    df.to_csv(csv_path, index=False)
    logger.info("[STEP LOAD.10] 保存 %s(行 %d) 与 %s", pt_path, len(df), csv_path)


def run_data_load(cfg: dict):
    """数据加载与清洗主入口。

    依次调用上述函数，最终保存 01_upstream_raw / 02_downstream_raw 双份文件。
    输入:
        cfg: 配置字典。
    返回:
        上游宽表、下游宽表、static、药物聚合表（四元组）。
    """
    tables = load_raw_tables(cfg)
    dynamic_df = clean_empty_rows(tables["dynamic"], "dynamic")
    label_df = clean_empty_rows(tables["label"], "label")
    static_df = clean_empty_rows(tables["static"], "static")
    drug_df = clean_empty_rows(tables["drug"], "drug")

    dynamic_df = clean_key_columns(dynamic_df, "dynamic", ["patient_id", "visit_date"])
    static_df = clean_key_columns(static_df, "static", ["patient_id"])
    drug_df = clean_key_columns(drug_df, "drug", ["patient_id", "visit_date"])
    label_df = clean_key_columns(label_df, "label", ["patient_id", "visit_date"])

    dynamic_df = parse_dates(dynamic_df, ["visit_date"])
    static_df = parse_dates(static_df, ["birth_date"])
    drug_df = parse_dates(drug_df, ["visit_date"])
    label_df = parse_dates(label_df, ["visit_date"])

    dynamic_df = deduplicate_by_key(dynamic_df, ["patient_id", "visit_date"], "dynamic")
    drug_df = deduplicate_by_key(drug_df, ["patient_id", "visit_date", "drug_name"], "drug")

    dynamic_df = mark_extreme_values(dynamic_df, cfg)

    drug_agg = aggregate_drug_table(drug_df, cfg)
    dynamic_df = merge_dynamic_drug(dynamic_df, drug_agg)

    upstream_df, downstream_df = split_upstream_downstream(dynamic_df, cfg)

    save_dir = cfg["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    df_to_pt_and_csv(upstream_df,
                     os.path.join(save_dir, "01_upstream_raw.pt"),
                     os.path.join(save_dir, "01_upstream_raw.csv"), cfg)
    df_to_pt_and_csv(downstream_df,
                     os.path.join(save_dir, "02_downstream_raw.pt"),
                     os.path.join(save_dir, "02_downstream_raw.csv"), cfg)

    logger.info("[STEP LOAD.END] 数据加载完成，保存路径: %s", save_dir)
    return upstream_df, downstream_df, static_df, drug_agg


if __name__ == "__main__":
    import param_setting
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    cfg = param_setting.get_config()
    run_data_load(cfg)
