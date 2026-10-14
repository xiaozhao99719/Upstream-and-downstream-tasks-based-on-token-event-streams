# -*- coding: utf-8 -*-
"""
data_process.py
===============

Token 化、预处理参数学习与数据集划分模块。

职责：把清洗后的上游/下游宽表转为事件 Token 长表，从上游训练集学习
预处理参数（词表、rare 归并、min/max、年龄中心化），把同一套参数应用到
所有集合，做三路时间编码、静态特征构建、序列截断与填充，统一保存
Token 长表、划分后数据集与预处理参数（.pt 与 .csv 双份）。

严格防泄漏（核心规则）：
- u24pro 和 label 不生成 Token；
- 下游地标输入排除地标当天 u24pro/label；
- 测试集（下游后 10%）不参与任何参数学习；预处理参数只从上游训练集学习。

设计约定：
- 每个函数写清 docstring，逐步打印状态 [STEP PROCESS.X] 描述... 结果；
- 不在函数内部保存文件，统一由 run_data_process 保存；
- 所有路径用 os.path.join；.pt 用 torch.save，dtype 由 cfg["pt_dtype"] 控制。

缺失值处理原则（第 21 项约定，逻辑不变，仅文档澄清）：
- mark_extreme_values 将临床合理范围外的极端异常值标记为 NaN；
- 预处理阶段不做任何插补，NaN 不落盘为特殊值：归一化后缺失/极端值的
  value_mask=0、value=0.0；
- 下游在 model.move_batch_to_device / handle_nan_values 中统一执行
  value = torch.nan_to_num(value, nan=0.0)，value_masks *= (~torch.isnan(values_orig))，
  让模型通过 mask 显式感知缺失，而不是用填充值污染分布。
- 该约定仅在下游消费时生效，预处理侧保持上述 mask 语义。

依赖: os, logging, json, pandas, numpy, torch
"""

import os
import logging
import json
import pandas as pd
import numpy as np
import torch

logger = logging.getLogger(__name__)

# 数值型检验列（生成 LAB Token；u24pro 是标签目标，不生成 Token）
NUMERIC_LAB_COLS = [
    "glu_f", "PLA2R_ab", "WBC", "HGB", "PLT", "NEUT_pct", "LYMPH_pct",
    "ALT", "AST", "TBIL", "salb", "stp", "SCR", "BUN", "UA", "TC", "TG",
    "LDL_C", "HDL_C", "urine_sg",
]
# 定性检验列（生成 CAT Token）
QUALITATIVE_COLS = ["urine_prot_qual", "urine_bld", "urine_glu", "urine_leuk"]

PAD_TOKEN_ID = 0
UNK_TOKEN_ID = 1
RARE_TOKEN_ID = 2
MASK_TOKEN_ID = 3     # 供上游 masked event modeling 挖空输入使用；真实 token 从 4 起

TORCH_DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
}


# ---------------------------------------------------------------------------
# 通用小工具
# ---------------------------------------------------------------------------
def _pt_dtype(cfg: dict):
    """根据 cfg["pt_dtype"] 返回 torch dtype（默认 float32）。"""
    return TORCH_DTYPES.get(cfg["pt_dtype"], torch.float32)


def _load_label_df(cfg: dict) -> pd.DataFrame:
    """从 cfg 指定的 label 文件读取并解析标签表（供地标识别使用）。"""
    path = os.path.join(cfg["data_dir"], cfg["label_file"])
    if not os.path.exists(path):
        raise FileNotFoundError(f"标签文件不存在: {path}")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df["visit_date"] = pd.to_datetime(df["visit_date"], errors="coerce", format="mixed")
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["patient_id", "visit_date"])
    return df


# ---------------------------------------------------------------------------
# 1. 绝对年龄
# ---------------------------------------------------------------------------
def compute_abs_age_days(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """计算 abs_age_days = visit_date - birth_date（单位：天）。

    缺失 birth_date 时，以该患者首次就诊为 Day 0。
    输入:
        df: 含 visit_date（含 birth_date 时使用）的宽表；cfg: 配置字典。
    返回:
        新增 abs_age_days 列的 DataFrame。
    """
    df = df.copy()
    has_birth = "birth_date" in df.columns
    missing_birth = int(df["birth_date"].isna().sum()) if has_birth else len(df)
    if has_birth:
        df["abs_age_days"] = (df["visit_date"] - df["birth_date"]).dt.days.astype(float)
    else:
        df["abs_age_days"] = np.nan
    # 缺失 birth_date 的患者：以首次就诊为 Day 0
    missing_mask = df["abs_age_days"].isna()
    if missing_mask.any():
        first = df[missing_mask].groupby("patient_id")["visit_date"].transform("min")
        df.loc[missing_mask, "abs_age_days"] = (
            df.loc[missing_mask, "visit_date"] - first[missing_mask]).dt.days.astype(float)
    df["abs_age_days"] = df["abs_age_days"].astype(float)
    logger.info("[STEP PROCESS.1] 绝对年龄计算: 缺失 birth_date 患者 %d, 缺失时以首次就诊为 Day 0",
                missing_birth)
    return df


# ---------------------------------------------------------------------------
# 2. 事件 Token 化
# ---------------------------------------------------------------------------
def tokenize_events(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """把每次随访展开为事件 Token 长表。

    规则：
      - 数值型 LAB：token_name = LAB_指标名，value=数值，value_mask=1；
      - 定性 CAT：token_name = CAT_指标名_取值，value=0，value_mask=0；
      - visit_type：CAT_visit_type_OPD/IPD/ER；
      - 药物：DRUG_药物名，有剂量 value_mask=1，无剂量 value_mask=0；
      - u24pro 和 label 不生成 Token；空值不生成 Token。
    attn_mask 对真实 Token 恒为 1。
    输入:
        df: 含 abs_age_days、visit_type、检验列、drug_list 的宽表；cfg: 配置字典。
    返回:
        Token 长表：patient_id, visit_date, abs_age_days, token_name,
        value, value_mask, attn_mask, event_type。
    """
    df = df.copy()
    lab_cols = [c for c in NUMERIC_LAB_COLS if c in df.columns]
    qual_cols = [c for c in QUALITATIVE_COLS if c in df.columns]
    has_type = "visit_type" in df.columns
    has_drug = "drug_list" in df.columns

    records = []
    for idx, row in df.iterrows():
        pid = row["patient_id"]
        vd = row["visit_date"]
        age = row["abs_age_days"]
        if has_type and pd.notna(row["visit_type"]):
            records.append(dict(patient_id=pid, visit_date=vd, abs_age_days=age,
                                token_name=f"CAT_visit_type_{row['visit_type']}",
                                value=0.0, value_mask=0, attn_mask=1, event_type="CAT"))
        for c in lab_cols:
            v = row[c]
            if pd.notna(v):
                records.append(dict(patient_id=pid, visit_date=vd, abs_age_days=age,
                                    token_name=f"LAB_{c}", value=float(v),
                                    value_mask=1, attn_mask=1, event_type="LAB"))
        for c in qual_cols:
            v = row[c]
            if pd.notna(v):
                records.append(dict(patient_id=pid, visit_date=vd, abs_age_days=age,
                                    token_name=f"CAT_{c}_{v}", value=0.0,
                                    value_mask=0, attn_mask=1, event_type="CAT"))
        if has_drug and isinstance(row.get("drug_list"), list):
            for dname, dose in row["drug_list"]:
                dname = str(dname)
                has_dose = pd.notna(dose)
                records.append(dict(patient_id=pid, visit_date=vd, abs_age_days=age,
                                    token_name=f"DRUG_{dname}",
                                    value=float(dose) if has_dose else 0.0,
                                    value_mask=1 if has_dose else 0,
                                    attn_mask=1, event_type="DRUG"))
    tokens = pd.DataFrame(records)
    if tokens.empty:
        tokens = pd.DataFrame(columns=["patient_id", "visit_date", "abs_age_days",
                                       "token_name", "value", "value_mask",
                                       "attn_mask", "event_type"])
    logger.info("[STEP PROCESS.2] Token 化: 总 %d, event_type 分布 %s",
                len(tokens), tokens["event_type"].value_counts().to_dict())
    return tokens


# ---------------------------------------------------------------------------
# 3. 下游地标识别
# ---------------------------------------------------------------------------
def identify_downstream_landmarks(downstream_df: pd.DataFrame, label_df: pd.DataFrame,
                                  cfg: dict) -> pd.DataFrame:
    """对下游每个 visit，判定 (landmark, landmark+window] 内是否有非空 label。

    y = 窗口内所有非空 label 按 label_agg 聚合（默认 mean）；
    记录 n_measure；窗口内无 label（n_measure=0）或窗口超出 data_cutoff 则剔除；
    生成 landmark_id = patient_id + "_" + landmark_visit_date；计算 overlap_flag。
    输入:
        downstream_df: 下游宽表；label_df: 标签表；cfg: 配置字典。
    返回:
        合格地标 DataFrame（patient_id, landmark_visit_date, landmark_id,
        y, n_measure, overlap_flag）。
    """
    window = int(cfg["prediction_window_days"])
    cutoff = pd.Timestamp(cfg["data_cutoff"])
    agg = cfg["label_agg"]
    labels = label_df.dropna(subset=["label"]).copy()

    qualified = []
    rejected = 0
    groups = labels.groupby("patient_id")
    for pid, g in downstream_df.groupby("patient_id"):
        visits = g["visit_date"].sort_values().values
        if pid not in groups.groups:
            rejected += len(visits)
            continue
        labs = groups.get_group(pid).sort_values("visit_date")
        lab_dates = labs["visit_date"].values
        lab_vals = labs["label"].values
        for vd in visits:
            end = vd + pd.Timedelta(days=window)
            if end > cutoff:
                rejected += 1
                continue
            inwin = np.where((lab_dates > vd) & (lab_dates <= end))[0]
            n = len(inwin)
            if n == 0:
                rejected += 1
                continue
            vals = lab_vals[inwin]
            if agg == "median":
                y = float(np.nanmedian(vals))
            elif agg == "first":
                y = float(vals[0])
            else:
                y = float(np.nanmean(vals))
            qualified.append(dict(patient_id=pid, landmark_visit_date=vd,
                                  y=y, n_measure=n))
    qualified_df = pd.DataFrame(qualified)
    if qualified_df.empty:
        logger.warning("[STEP PROCESS.3] 下游无合格地标！")
        qualified_df = pd.DataFrame(columns=["patient_id", "landmark_visit_date",
                                             "landmark_id", "y", "n_measure",
                                             "overlap_flag"])
        return qualified_df
    qualified_df["landmark_id"] = (qualified_df["patient_id"].astype(str) + "_" +
                                   qualified_df["landmark_visit_date"].dt.strftime("%Y-%m-%d"))

    # overlap_flag：同一患者当前地标窗口 (vd, vd+window] 内是否还有另一个地标
    qualified_df["overlap_flag"] = 0
    overlap = 0
    win = pd.Timedelta(days=window)
    for _, g in qualified_df.groupby("patient_id"):
        g = g.sort_values("landmark_visit_date")
        if len(g) < 2:
            continue
        next_dates = g["landmark_visit_date"].shift(-1)
        ov = next_dates.notna() & (next_dates <= g["landmark_visit_date"] + win)
        qualified_df.loc[g.index, "overlap_flag"] = ov.astype(int).values
        overlap += int(ov.sum())

    logger.info("[STEP PROCESS.3] 地标识别: 合格 %d, 剔除 %d, 重叠地标 %d",
                len(qualified_df), rejected, overlap)
    return qualified_df


# ---------------------------------------------------------------------------
# 4. 为下游 Token 关联 landmark_id
# ---------------------------------------------------------------------------
def attach_landmark_id_to_tokens(tokens_df: pd.DataFrame, landmarks_df: pd.DataFrame,
                                 cfg: dict) -> pd.DataFrame:
    """为下游 Token 长表标记每个地标对应的 landmark_id。

    地标输入取 visit_date <= landmark_visit_date 的 Token，并排除地标当天
    u24pro/label（u24pro/label 本身不生成 Token，此步为显式保护）。
    一个 Token 可同时属于多个地标，此时在长表中展开为多行。
    输入:
        tokens_df: 下游 Token 长表；landmarks_df: 合格地标表；cfg: 配置字典。
    返回:
        带 landmark_id 的下游 Token 长表。
    """
    tok = tokens_df.copy()
    out_parts = []
    counts = []
    exclude_prefix = ("LAB_u24pro", "LAB_label")
    by_patient = {pid: g for pid, g in tok.groupby("patient_id")}
    for pid, g in landmarks_df.groupby("patient_id"):
        lds = g.sort_values("landmark_visit_date")
        toks = by_patient.get(pid)
        if toks is None or len(toks) == 0:
            counts.extend([0] * len(lds))
            continue
        toks = toks.sort_values("visit_date").reset_index(drop=True)
        tok_time = toks["visit_date"].to_numpy()          # 已按时间升序
        for _, lrow in lds.iterrows():
            lvd = np.datetime64(pd.Timestamp(lrow["landmark_visit_date"]))
            # 二分定位 visit_date <= lvd 的前缀区间 [0, pos)，避免逐地标全表扫描，
            # 使整体复杂度从 O(地标数 × Token 数) 降为 O(地标数 × log Token 数)。
            pos = int(np.searchsorted(tok_time, lvd, side="right"))
            sub = toks.iloc[:pos]
            if pos > 0:
                same_day = sub["visit_date"].to_numpy() == lvd
                if same_day.any():
                    sub = sub[~(same_day &
                                sub["token_name"].str.startswith(exclude_prefix))]
            if len(sub) == 0:
                counts.append(0)
                continue
            s = sub.copy()
            s["landmark_id"] = lrow["landmark_id"]
            out_parts.append(s)
            counts.append(len(s))
    if out_parts:
        out = pd.concat(out_parts, ignore_index=True)
    else:
        out = tok.copy()
        out["landmark_id"] = np.nan
    counts = np.array(counts) if counts else np.zeros(len(landmarks_df), dtype=int)
    logger.info("[STEP PROCESS.4] 地标关联: 地标 %d 个, 每地标 Token 数 "
                "(min/median/P90/max)=%.0f/%.0f/%.0f/%.0f",
                len(landmarks_df), counts.min(), np.median(counts),
                np.percentile(counts, 90), counts.max())
    return out


# ---------------------------------------------------------------------------
# 5. 上游患者划分
# ---------------------------------------------------------------------------
def split_upstream_patients(upstream_df: pd.DataFrame, cfg: dict):
    """按患者 ID 划分上游：90% 训练 / 10% 验证。

    输入:
        upstream_df: 上游宽表；cfg: 配置字典。
    返回:
        训练患者列表、验证患者列表。
    """
    rng = np.random.RandomState(cfg["random_seed"])
    patients = np.array(sorted(upstream_df["patient_id"].unique()))
    rng.shuffle(patients)
    n_val = max(1, int(round(len(patients) * cfg["upstream_val_ratio"])))
    val = sorted(patients[:n_val].tolist())
    train = sorted(patients[n_val:].tolist())
    logger.info("[STEP PROCESS.5] 上游划分: 训练患者 %d, 验证患者 %d",
                len(train), len(val))
    return train, val


# ---------------------------------------------------------------------------
# 6. 下游地标划分
# ---------------------------------------------------------------------------
def split_downstream_landmarks(landmarks_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """按 landmark_visit_date 升序，后 10% 为测试集，前 90% 为训练+验证。

    新增 split 列（trainval / test）。下游链式 CV 的 train/val 折留到训练阶段。
    输入:
        landmarks_df: 合格地标表；cfg: 配置字典。
    返回:
        带 split 列的地标 DataFrame。
    """
    if landmarks_df.empty:
        landmarks_df["split"] = []
        logger.info("[STEP PROCESS.6] 下游划分: 地标为空")
        return landmarks_df
    df = landmarks_df.sort_values("landmark_visit_date").reset_index(drop=True)
    n = len(df)
    n_test = int(round(n * cfg["test_ratio"]))
    n_test = min(n_test, n)
    split = ["trainval"] * n
    if n_test > 0:
        for i in range(n - n_test, n):
            split[i] = "test"
    df["split"] = split
    logger.info("[STEP PROCESS.6] 下游划分: trainval %d, test %d, 日期范围 %s ~ %s",
                n - n_test, n_test,
                df["landmark_visit_date"].min().date(),
                df["landmark_visit_date"].max().date())
    return df


# ---------------------------------------------------------------------------
# 7. Token 词表
# ---------------------------------------------------------------------------
def build_token_vocab(upstream_train_tokens: pd.DataFrame, cfg: dict):
    """统计 Token 频次，低频归并 RARE，构建词表，并输出 token 频率。

    预留 id：0=PAD, 1=UNK, 2=RARE, 3=MASK（上游 MEM 挖空用）；真实 Token 从 id=4 起。
    token_freq（第 15 项）：
    - 仅基于 05_upstream_train.pt（上游训练集）统计，严格防泄漏；
    - 键为 token_id（int），值为该 id 出现次数；
    - 被归并为 RARE 的低频 token，其频率全部计入 RARE 的 token_id（=2）；
    - 下游 masked_event_modeling_loss 用 w_i = 1/sqrt(freq_i+1) 加权交叉熵。

    输入:
        upstream_train_tokens: 上游训练集 Token 长表；cfg: 配置字典。
    返回:
        vocab 字典（token_name -> id）、rare 映射（token_name -> RARE_TOKEN_ID）、
        token_freq 字典（token_id -> count）。
    """
    counts = upstream_train_tokens["token_name"].value_counts()
    rare_names = set(counts[counts < cfg["rare_threshold"]].index.tolist())
    vocab = {}
    vid = 4          # 0=PAD, 1=UNK, 2=RARE, 3=MASK 已被占用
    for name in counts.index:
        if name in rare_names:
            continue
        vocab[name] = vid
        vid += 1
    rare_map = {name: RARE_TOKEN_ID for name in rare_names}

    # token_freq：真实 token 计入自身 id；低频 token 频率合并入 RARE id
    token_freq = {}
    rare_total = 0
    for name, c in counts.items():
        c = int(c)
        if name in vocab:
            token_freq[vocab[name]] = c
        else:
            rare_total += c
    token_freq[RARE_TOKEN_ID] = rare_total

    logger.info("[STEP PROCESS.7] 词表大小=%d, RARE 数量=%d, 低频 token 数=%d",
                len(vocab), len(rare_map), len(rare_map))
    return vocab, rare_map, token_freq


# ---------------------------------------------------------------------------
# 8. 数值 Min-Max
# ---------------------------------------------------------------------------
def compute_numeric_minmax(upstream_train_tokens: pd.DataFrame) -> dict:
    """对 value_mask=1 的 Token 统计 min/max（按 token_name）。

    单类型在训练集内无变化（max==min）时无法做 Min-Max，改用同事件类型
    （token_name 前缀，如 DRUG_*/LAB_*）在训练集内**合并统计**的区间兜底，
    键名形如 "DRUG_*"（不会与真实 token_name 冲突）。
    这样所有 value_mask=1 的数值 Token 都能被归一化到 [0,1]：
    否则剂量/量级（可达数百）会以原始值进入 value 通道，使 mask_value 损失被放大
    数个量级，并在混合精度下引发梯度溢出。

    输入:
        upstream_train_tokens: 上游训练集 Token 长表。
    返回:
        {token_name 或 "前缀_*": (min, max)} 字典。
    """
    sub = upstream_train_tokens[upstream_train_tokens["value_mask"] == 1]
    g = sub.groupby("token_name")["value"].agg(["min", "max"])
    minmax = {}
    skipped = []
    for name, row in g.iterrows():
        if pd.notna(row["min"]) and pd.notna(row["max"]) and (row["max"] - row["min"]) > 0:
            minmax[name] = (float(row["min"]), float(row["max"]))
        else:
            skipped.append(str(name))
    # 前缀级兜底：把同前缀（LAB_/DRUG_ 等）的数值合并统计
    prefixes = {}
    if len(sub) > 0:
        pref = sub["token_name"].astype(str).str.split("_").str[0]
        pg = sub.groupby(pref)["value"].agg(["min", "max"])
        for p, row in pg.iterrows():
            if pd.notna(row["min"]) and pd.notna(row["max"]) and (row["max"] - row["min"]) > 0:
                prefixes[f"{p}_*"] = (float(row["min"]), float(row["max"]))
    minmax.update(prefixes)
    logger.info("[STEP PROCESS.8] 数值 Token 范围: 单类型 %d 个, 前缀兜底 %s, "
                "训练集内无变化类型 %d 个 %s",
                len(minmax) - len(prefixes), sorted(prefixes), len(skipped),
                sorted(skipped)[:20])
    return minmax


# ---------------------------------------------------------------------------
# 9. 年龄中心化
# ---------------------------------------------------------------------------
def compute_age_center(upstream_train_tokens: pd.DataFrame) -> float:
    """计算 abs_age_days / 365.25 的均值（年龄中心化参数）。

    输入:
        upstream_train_tokens: 上游训练集 Token 长表。
    返回:
        中心化均值（年）。
    """
    vals = upstream_train_tokens["abs_age_days"] / 365.25
    center = float(vals.mean())
    logger.info("[STEP PROCESS.9] 年龄中心化: 均值 %.6f 年", center)
    return center


# ---------------------------------------------------------------------------
# 10. Token ID 映射
# ---------------------------------------------------------------------------
def apply_token_id_mapping(tokens_df: pd.DataFrame, vocab: dict, rare_map: dict) -> pd.DataFrame:
    """把 token_name 映射为 token_id；新 Token -> UNK，低频 -> RARE。

    输入:
        tokens_df: Token 长表；vocab: 词表；rare_map: rare 映射。
    返回:
        新增 token_id 列的 DataFrame。
    """
    df = tokens_df.copy()
    ids = df["token_name"].map(vocab)
    unknown_mask = ids.isna()
    rare_ids = df.loc[unknown_mask, "token_name"].map(rare_map)
    n_rare = int(rare_ids.notna().sum())
    ids[unknown_mask] = rare_ids.fillna(UNK_TOKEN_ID)
    df["token_id"] = ids.astype(np.int64)
    n_unk = int(unknown_mask.sum()) - n_rare
    logger.info("[STEP PROCESS.10] Token ID 映射: UNK %d, RARE %d", n_unk, n_rare)
    return df


# ---------------------------------------------------------------------------
# 11. 数值归一化
# ---------------------------------------------------------------------------
def normalize_numeric_values(tokens_df: pd.DataFrame, minmax_dict: dict) -> pd.DataFrame:
    """对 value_mask=1 的 Token 做 Min-Max 归一化，超出范围截断。

    区间查找顺序：token_name 专属区间 -> 同前缀兜底区间（如 "DRUG_*"）。
    两者都缺失（训练集内没有任何可用标定）时，该 Token 按"无数值"处理
    （value=0, value_mask=0），仅保留事件本身，避免原始量级污染 value 通道。

    输入:
        tokens_df: Token 长表；minmax_dict: {token_name: (min,max)}。
    返回:
        归一化后的 DataFrame。
    """
    df = tokens_df.copy()
    df["value"] = df["value"].astype(float).fillna(0.0)
    lo_map = {k: v[0] for k, v in minmax_dict.items()}
    hi_map = {k: v[1] for k, v in minmax_dict.items()}
    name = df["token_name"].astype(str)
    lo = name.map(lo_map)
    hi = name.map(hi_map)
    miss = lo.isna() | hi.isna()
    if bool(miss.any()):
        pref = name.str.split("_").str[0] + "_*"
        lo = lo.fillna(pref.map(lo_map))
        hi = hi.fillna(pref.map(hi_map))
    span = hi - lo
    has_value = df["value_mask"] == 1
    ok = has_value & span.notna() & (span > 0)
    scaled = ((df["value"] - lo) / span.replace(0.0, np.nan)).clip(0.0, 1.0)
    df["value"] = scaled.where(ok, 0.0)
    dropped = has_value & ~ok
    if bool(dropped.any()):
        names = sorted(set(df.loc[dropped, "token_name"].astype(str).tolist()))
        logger.warning("[STEP PROCESS.11] 以下数值 Token 在训练集中无可用标定区间，"
                       "已按无数值处理(value=0, value_mask=0): %s", names[:20])
        df.loc[dropped, "value_mask"] = 0
    vm = df["value_mask"] == 1
    logger.info("[STEP PROCESS.11] 数值归一化: 覆盖 %d 个区间, 数值 Token 值域 "
                "[%.3f, %.3f], 无标定降级 %d 行",
                len(minmax_dict),
                df.loc[vm, "value"].min() if bool(vm.any()) else 0.0,
                df.loc[vm, "value"].max() if bool(vm.any()) else 0.0,
                int(dropped.sum()))
    return df


# ---------------------------------------------------------------------------
# 12. 时间特征编码
# ---------------------------------------------------------------------------
def encode_time_features(tokens_df: pd.DataFrame, age_center: float) -> pd.DataFrame:
    """三路时间编码。

      - abs_age_years = abs_age_days / 365.25 - age_center；
      - interval_univ  = log(1 + 距上次任意就诊天数)；
      - interval_same  = log(1 + 距上次同类 token 天数)；
    首次事件置 0。按患者在其时间轴上计算，保证序列内正确。
    输入:
        tokens_df: Token 长表；age_center: 年龄中心化参数。
    返回:
        新增 abs_age_years / interval_univ / interval_same 列的 DataFrame。
    """
    df = tokens_df.copy().reset_index(drop=True)
    df["abs_age_years"] = df["abs_age_days"] / 365.25 - age_center

    # 通用间隔：按患者、时间排序，取上一访问
    df["_order"] = np.arange(len(df))
    df = df.sort_values(["patient_id", "visit_date"]).reset_index(drop=True)
    prev_any = df.groupby("patient_id")["visit_date"].shift(1)
    df["interval_univ"] = np.log1p(
        (df["visit_date"] - prev_any).dt.days.fillna(0).astype(float).values)

    # 同类间隔：按患者、token、时间排序，取上一同类
    df = df.sort_values(["patient_id", "token_name", "visit_date"]).reset_index(drop=True)
    prev_same = df.groupby(["patient_id", "token_name"])["visit_date"].shift(1)
    df["interval_same"] = np.log1p(
        (df["visit_date"] - prev_same).dt.days.fillna(0).astype(float).values)

    df = df.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    logger.info("[STEP PROCESS.12] 时间特征编码: abs_age_years [%.3f, %.3f], "
                "interval_univ [%.3f, %.3f], interval_same [%.3f, %.3f]",
                df["abs_age_years"].min(), df["abs_age_years"].max(),
                df["interval_univ"].min(), df["interval_univ"].max(),
                df["interval_same"].min(), df["interval_same"].max())
    return df


# ---------------------------------------------------------------------------
# 13. 静态特征
# ---------------------------------------------------------------------------
def build_static_features(static_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """从 static.csv 提取静态特征，类别 one-hot 编码，缺失映射为 Unknown。

    严格防泄漏：one-hot 的类别层级（列集合）只从上游训练集患者统计学习，
    其他患者（含下游测试）按同一列集合编码，未见类别 → 全 0，
    避免测试集参与静态特征的定义。

    输入:
        static_df: 静态表；cfg: 配置字典（需含 upstream_train_patients）。
    返回:
        S 向量 DataFrame（patient_id + one-hot 列）。
    """
    df = static_df.copy()
    cat_cols = [c for c in df.columns if c not in ("patient_id", "birth_date")]
    missing_cnt = int(df[cat_cols].isna().sum().sum())
    for c in cat_cols:
        df[c] = df[c].fillna("Unknown").astype(str)
    train_pat = set(cfg.get("upstream_train_patients") or [])
    fit_df = df[df["patient_id"].isin(train_pat)] if train_pat else df
    cols = list(pd.get_dummies(fit_df[cat_cols], prefix_sep="=").columns)
    encoded = pd.get_dummies(df[cat_cols], prefix_sep="=").reindex(
        columns=cols, fill_value=0).astype(np.float32)
    sdf = pd.concat([df[["patient_id"]].reset_index(drop=True),
                     encoded.reset_index(drop=True)], axis=1)
    d_s = sdf.shape[1] - 1
    logger.info("[STEP PROCESS.13] 静态特征: 维度 d_s=%d, 缺失值 %d 个(映射为 Unknown), "
                "类别层级拟合自上游训练集 %d 名患者",
                d_s, missing_cnt, len(fit_df))
    return sdf


# ---------------------------------------------------------------------------
# 14. 序列截断与填充
# ---------------------------------------------------------------------------
def _build_sequence(group: pd.DataFrame, seq_length: int):
    """把一组按时间升序的 Token 构建为定长序列（左填充）。

    返回: (token_ids, values, value_masks, attn_masks, abs_ages,
           interval_univ, interval_same, n_tokens) 各为 numpy 数组/标量。
    """
    keep = group.tail(seq_length)
    n_tokens = len(keep)
    pad = seq_length - n_tokens
    ids = np.full(seq_length, PAD_TOKEN_ID, dtype=np.int64)
    vals = np.zeros(seq_length, dtype=np.float32)
    vmasks = np.zeros(seq_length, dtype=np.int8)
    amasks = np.zeros(seq_length, dtype=np.int8)
    ages = np.zeros(seq_length, dtype=np.float32)
    iuniv = np.zeros(seq_length, dtype=np.float32)
    isame = np.zeros(seq_length, dtype=np.float32)
    if n_tokens > 0:
        ids[pad:] = keep["token_id"].values.astype(np.int64)
        vals[pad:] = keep["value"].values.astype(np.float32)
        vmasks[pad:] = keep["value_mask"].values.astype(np.int8)
        amasks[pad:] = keep["attn_mask"].values.astype(np.int8)
        ages[pad:] = keep["abs_age_years"].values.astype(np.float32)
        iuniv[pad:] = keep["interval_univ"].values.astype(np.float32)
        isame[pad:] = keep["interval_same"].values.astype(np.float32)
        # 截断边界锚定：原序列起点被 tail() 丢弃，第一个保留 token 的间隔
        # 应重新锚定为 0（位置 pad 即左填充后第一个非 PAD token）。
        iuniv[pad] = 0.0
        isame[pad] = 0.0
    return ids, vals, vmasks, amasks, ages, iuniv, isame, n_tokens


def truncate_and_pad(tokens_df: pd.DataFrame, landmarks_df: pd.DataFrame,
                     seq_length: int, cfg: dict) -> dict:
    """截断与填充，生成四个张量字典。

    输入:
        tokens_df: 上游与下游拼接后的 Token 长表（上游 landmark_id 为 NaN）；
        landmarks_df: 下游地标表（含 split）；seq_length: 序列长度；cfg: 配置字典。
    返回:
        {"upstream_train": {...}, "upstream_val": {...},
         "downstream_trainval": {...}, "downstream_test": {...}}。
    """
    S_lookup = {}
    d_s = 0
    if cfg.get("_S_df") is not None:
        sdf = cfg["_S_df"]
        d_s = sdf.shape[1]
        for pid, row in sdf.iterrows():
            S_lookup[pid] = row.values.astype(np.float32)
    zero_s = np.zeros(d_s, dtype=np.float32)

    tok = tokens_df.copy()
    up_mask = tok["landmark_id"].isna()
    up_tokens = tok[up_mask]
    dn_tokens = tok[~up_mask]

    # 上游：按患者
    train_pat = set(cfg["upstream_train_patients"])
    val_pat = set(cfg["upstream_val_patients"])
    up_train, up_val = {}, {}
    for name, pat_set, bucket in [("upstream_train", train_pat, up_train),
                                  ("upstream_val", val_pat, up_val)]:
        arrays = {"token_ids": [], "values": [], "value_masks": [], "attn_masks": [],
                  "abs_ages": [], "interval_univ": [], "interval_same": [],
                  "S": [], "patient_id": [], "n_tokens": []}
        for pid, g in up_tokens.groupby("patient_id"):
            if pid not in pat_set:
                continue
            g = g.sort_values("visit_date")
            ids, vals, vmasks, amasks, ages, iu, isa, nt = _build_sequence(g, seq_length)
            arrays["token_ids"].append(ids)
            arrays["values"].append(vals)
            arrays["value_masks"].append(vmasks)
            arrays["attn_masks"].append(amasks)
            arrays["abs_ages"].append(ages)
            arrays["interval_univ"].append(iu)
            arrays["interval_same"].append(isa)
            arrays["S"].append(S_lookup.get(pid, zero_s))
            arrays["patient_id"].append(str(pid))
            arrays["n_tokens"].append(nt)
        bucket.update(_stack_arrays(arrays))
        bucket["seq_len"] = seq_length
        logger.info("[STEP PROCESS.14] 序列构建 %s: %d 样本, 截断前 Token 数分布 "
                    "(min/median/P90/max)=%.0f/%.0f/%.0f/%.0f",
                    name, len(arrays["patient_id"]),
                    np.min(arrays["n_tokens"]) if arrays["n_tokens"] else 0,
                    np.median(arrays["n_tokens"]) if arrays["n_tokens"] else 0,
                    np.percentile(arrays["n_tokens"], 90) if arrays["n_tokens"] else 0,
                    np.max(arrays["n_tokens"]) if arrays["n_tokens"] else 0)

    # 下游：按地标
    lmeta = landmarks_df.set_index("landmark_id")
    dn_trainval, dn_test = {}, {}
    for split_name, bucket in [("trainval", dn_trainval), ("test", dn_test)]:
        arrays = {"token_ids": [], "values": [], "value_masks": [], "attn_masks": [],
                  "abs_ages": [], "interval_univ": [], "interval_same": [],
                  "S": [], "patient_id": [], "landmark_id": [], "y": [],
                  "n_measure": [], "overlap_flag": [], "n_tokens": []}
        for lid, g in dn_tokens.groupby("landmark_id"):
            if lid not in lmeta.index:
                continue
            meta = lmeta.loc[lid]
            if meta["split"] != split_name:
                continue
            g = g.sort_values("visit_date")
            ids, vals, vmasks, amasks, ages, iu, isa, nt = _build_sequence(g, seq_length)
            arrays["token_ids"].append(ids)
            arrays["values"].append(vals)
            arrays["value_masks"].append(vmasks)
            arrays["attn_masks"].append(amasks)
            arrays["abs_ages"].append(ages)
            arrays["interval_univ"].append(iu)
            arrays["interval_same"].append(isa)
            arrays["S"].append(S_lookup.get(meta["patient_id"], zero_s))
            arrays["patient_id"].append(str(meta["patient_id"]))
            arrays["landmark_id"].append(str(lid))
            arrays["y"].append(float(meta["y"]))
            arrays["n_measure"].append(int(meta["n_measure"]))
            arrays["overlap_flag"].append(int(meta["overlap_flag"]))
            arrays["n_tokens"].append(nt)
        bucket.update(_stack_arrays(arrays))
        bucket["seq_len"] = seq_length
        logger.info("[STEP PROCESS.14] 序列构建 %s(%s): %d 样本, 截断前 Token 数 "
                    "(min/median/P90/max)=%.0f/%.0f/%.0f/%.0f",
                    "downstream", split_name, len(arrays["landmark_id"]),
                    np.min(arrays["n_tokens"]) if arrays["n_tokens"] else 0,
                    np.median(arrays["n_tokens"]) if arrays["n_tokens"] else 0,
                    np.percentile(arrays["n_tokens"], 90) if arrays["n_tokens"] else 0,
                    np.max(arrays["n_tokens"]) if arrays["n_tokens"] else 0)

    return {"upstream_train": up_train, "upstream_val": up_val,
            "downstream_trainval": dn_trainval, "downstream_test": dn_test}


def _stack_arrays(arrays: dict) -> dict:
    """把列表型张量堆叠为 numpy 数组（含 str 元数据）。"""
    out = {}
    for k, v in arrays.items():
        if k in ("patient_id", "landmark_id"):
            out[k] = list(v)
        else:
            out[k] = np.array(v, dtype=object) if k in ("y", "n_measure", "overlap_flag", "n_tokens") \
                else np.stack(v) if v else np.empty((0,))
    # 把标量型转成 float/int 数组
    for k in ("y", "n_measure", "overlap_flag", "n_tokens"):
        if k in out:
            out[k] = np.asarray(out[k], dtype=np.float32 if k == "y" else np.int64)
    return out


# ---------------------------------------------------------------------------
# 15. 张量保存
# ---------------------------------------------------------------------------
def tensors_to_pt_and_csv(tensors: dict, pt_path: str, csv_path: str, cfg: dict,
                          is_downstream: bool) -> None:
    """把张量字典同时保存为 .pt 与 .csv。

    .pt: 数组转 torch.tensor（dtype 由 cfg["pt_dtype"] 控制），torch.save。
    .csv: 展平保存可视化摘要字段。
    输入:
        tensors: 张量字典；pt_path/csv_path: 保存路径；cfg: 配置字典；
        is_downstream: 是否为下游张量。
    """
    os.makedirs(os.path.dirname(pt_path), exist_ok=True)
    torch_tensors = {}
    for k, v in tensors.items():
        if k == "seq_len":
            continue
        if k in ("patient_id", "landmark_id", "split"):
            torch_tensors[k] = v
        elif k == "token_ids":
            torch_tensors[k] = torch.tensor(v, dtype=torch.int64)
        elif k in ("value_masks", "attn_masks"):
            torch_tensors[k] = torch.tensor(v, dtype=torch.int8)
        else:
            torch_tensors[k] = torch.tensor(v, dtype=_pt_dtype(cfg))
    torch.save(torch_tensors, pt_path)

    n = len(tensors.get("patient_id", []))
    seq_len = tensors.get("seq_len", 0)
    rows = []
    if is_downstream:
        for i in range(n):
            rows.append(dict(
                landmark_id=tensors["landmark_id"][i],
                patient_id=tensors["patient_id"][i],
                y=float(tensors["y"][i]),
                n_measure=int(tensors["n_measure"][i]),
                overlap_flag=int(tensors["overlap_flag"][i]),
                split=tensors.get("split", [""] * n)[i] if isinstance(
                    tensors.get("split"), list) else "",
                n_tokens=int(tensors["n_tokens"][i]),
                seq_len=seq_len))
    else:
        for i in range(n):
            rows.append(dict(patient_id=tensors["patient_id"][i],
                             n_tokens=int(tensors["n_tokens"][i]),
                             seq_len=seq_len))
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    logger.info("[STEP PROCESS.15] 保存 %s(样本 %d, 各张量 %s) 与 %s",
                pt_path, n,
                {k: (list(v.shape) if hasattr(v, "shape") else len(v))
                 for k, v in torch_tensors.items()}, csv_path)


# ---------------------------------------------------------------------------
# 16. 主入口
# ---------------------------------------------------------------------------
def run_data_process(upstream_df: pd.DataFrame, downstream_df: pd.DataFrame,
                     static_df: pd.DataFrame, cfg: dict) -> dict:
    """数据预处理主入口。

    依次调用上述函数，最终保存 03~08 与 preprocess_params 双份文件。
    输入:
        upstream_df: 上游清洗宽表；downstream_df: 下游清洗宽表；
        static_df: 静态表；cfg: 配置字典。
    返回:
        张量与参数字典。
    """
    save_dir = cfg["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    # 合并出生日期（年龄计算用）
    birth = static_df[["patient_id", "birth_date"]].drop_duplicates("patient_id")
    upstream_df = upstream_df.merge(birth, on="patient_id", how="left")
    downstream_df = downstream_df.merge(birth, on="patient_id", how="left")

    # 1. 绝对年龄
    upstream_df = compute_abs_age_days(upstream_df, cfg)
    downstream_df = compute_abs_age_days(downstream_df, cfg)

    # 2. Token 化
    upstream_tokens = tokenize_events(upstream_df, cfg)
    downstream_tokens = tokenize_events(downstream_df, cfg)

    # 3. 下游地标识别 + 划分
    label_df = _load_label_df(cfg)
    landmarks = identify_downstream_landmarks(downstream_df, label_df, cfg)
    landmarks = split_downstream_landmarks(landmarks, cfg)

    # 5. 上游患者划分（写入 cfg，供截断使用）
    train_pat, val_pat = split_upstream_patients(upstream_df, cfg)
    cfg["upstream_train_patients"] = train_pat
    cfg["upstream_val_patients"] = val_pat

    # 7-9. 从上游训练集学习参数（测试集绝不参与）
    train_ids = set(train_pat)
    up_train_tokens = upstream_tokens[upstream_tokens["patient_id"].isin(train_ids)]
    vocab, rare_map, token_freq = build_token_vocab(up_train_tokens, cfg)
    minmax = compute_numeric_minmax(up_train_tokens)
    age_center = compute_age_center(up_train_tokens)

    # 10-11. 应用 ID 映射与归一化（上下游同一套参数）
    upstream_tokens = apply_token_id_mapping(upstream_tokens, vocab, rare_map)
    downstream_tokens = apply_token_id_mapping(downstream_tokens, vocab, rare_map)
    upstream_tokens = normalize_numeric_values(upstream_tokens, minmax)
    downstream_tokens = normalize_numeric_values(downstream_tokens, minmax)

    # 12. 三路时间编码（按患者时间轴，先于地标展开以保证间隔正确）
    upstream_tokens = encode_time_features(upstream_tokens, age_center)
    downstream_tokens = encode_time_features(downstream_tokens, age_center)

    # 4. 为下游 Token 关联 landmark_id（展开，保留已编码特征）
    downstream_tokens = attach_landmark_id_to_tokens(downstream_tokens, landmarks, cfg)
    downstream_tokens["landmark_id"] = downstream_tokens["landmark_id"].fillna("")
    upstream_tokens["landmark_id"] = np.nan

    # 13. 静态特征
    sdf = build_static_features(static_df, cfg)
    cfg["_S_df"] = sdf.set_index("patient_id")

    # 14. 截断填充
    combined = pd.concat([upstream_tokens, downstream_tokens], ignore_index=True)
    tensors = truncate_and_pad(combined, landmarks, cfg["seq_length"], cfg)

    # 补充下游 split 元数据
    for bucket in (tensors["downstream_trainval"], tensors["downstream_test"]):
        bucket["split"] = ["trainval"] * len(bucket["landmark_id"]) if bucket is \
            tensors["downstream_trainval"] else ["test"] * len(bucket["landmark_id"])

    # 保存 03/04 Token 长表
    df_to_pt_and_csv(upstream_tokens,
                     os.path.join(save_dir, "03_upstream_tokens.pt"),
                     os.path.join(save_dir, "03_upstream_tokens.csv"), cfg)
    df_to_pt_and_csv(downstream_tokens,
                     os.path.join(save_dir, "04_downstream_tokens.pt"),
                     os.path.join(save_dir, "04_downstream_tokens.csv"), cfg)

    # 保存 05~08
    names = [("05_upstream_train", tensors["upstream_train"], False),
             ("06_upstream_val", tensors["upstream_val"], False),
             ("07_downstream_trainval", tensors["downstream_trainval"], True),
             ("08_downstream_test", tensors["downstream_test"], True)]
    for tag, tdict, is_dn in names:
        tensors_to_pt_and_csv(tdict, os.path.join(save_dir, tag + ".pt"),
                              os.path.join(save_dir, tag + ".csv"), cfg, is_dn)

    # preprocess_params（新增 token_freq 字段，不破坏旧格式）
    params = {"token_vocab": vocab, "rare_mapping": rare_map,
              "numeric_minmax": minmax, "age_center": age_center,
              "token_freq": token_freq}
    torch.save(params, os.path.join(save_dir, "preprocess_params.pt"))
    pr_rows = []
    for k, v in params.items():
        if k == "token_freq":
            # 展平为 token_freq_{id},count 行
            for tid, cnt in sorted(token_freq.items(), key=lambda x: int(x[0])):
                pr_rows.append((f"token_freq_{int(tid)}", int(cnt)))
        elif isinstance(v, (int, float, str)):
            pr_rows.append((k, v))
        else:
            pr_rows.append((k, json.dumps(v, ensure_ascii=False, default=str)))
    pd.DataFrame(pr_rows, columns=["key", "value"]).to_csv(
        os.path.join(save_dir, "preprocess_params.csv"), index=False)
    logger.info("[STEP PROCESS.16] 预处理完成，输出到 %s", save_dir)
    return {"tensors": tensors, "params": params}


def df_to_pt_and_csv(df: pd.DataFrame, pt_path: str, csv_path: str, cfg: dict) -> None:
    """把一个 DataFrame 同时保存为 .pt 与 .csv。"""
    os.makedirs(os.path.dirname(pt_path), exist_ok=True)
    torch.save(df, pt_path)
    df.to_csv(csv_path, index=False)
    logger.info("[STEP SAVE] 保存 %s(行 %d) 与 %s", pt_path, len(df), csv_path)


if __name__ == "__main__":
    import param_setting
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    cfg = param_setting.get_config()
    up = pd.read_pickle("01_upstream_raw.pt") if False else None
    print("请在 main.py 中调用本模块。")
