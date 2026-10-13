# -*- coding: utf-8 -*-
"""
main.py
=======

CLP 医学时序对比学习管线主控。

阶段：
  load -> process -> upstream_train:{method}:{seed} -> upstream_test:{method}
  -> downstream_train:{model}:{seed} -> downstream_test:{model}
  -> report_table -> final summary

特性：
- StateManager 自动识别阶段是否已完成（指纹匹配则跳过）；
- 断点续跑：running/failed 阶段自动重跑，--force 强制重跑；
- 严格防泄漏：validate_no_leakage 校验时间边界；
- 日志：save/preprocess.log（管线级）+ save/logs/{stage}.log（各阶段，由阶段模块写入）；
- 输出精简：checkpoint / state JSON / evaluation_results.csv /
  hpo_results_{model}.csv / report_table.csv / test_audit.json / run_config。

依赖: os, sys, json, logging, datetime, hashlib, numpy, torch
"""

import os
import sys
import json
import hashlib
import logging
import datetime

import pandas as pd
import torch

import param_setting
import data_load
import data_process

logger = logging.getLogger(__name__)

CODE_FILES = ["param_setting.py", "data_load.py", "data_process.py",
              "model.py", "train_upstream.py", "train_downstream.py",
              "test_upstream.py", "test_downstream.py", "main.py"]


# ---------------------------------------------------------------------------
# 日志与配置转储
# ---------------------------------------------------------------------------
def setup_logging(cfg: dict) -> str:
    """创建目录并配置控制台 + save/preprocess.log 日志。"""
    for d in (cfg["save_dir"], cfg["model_dir"], cfg["state_dir"], cfg["logs_dir"]):
        os.makedirs(d, exist_ok=True)
    log_path = param_setting.get_pipeline_log_path(cfg)
    level = getattr(logging, str(cfg["log_level"]).upper(), logging.INFO)
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    handlers = [logging.StreamHandler(sys.stdout),
                logging.FileHandler(log_path, mode="a", encoding="utf-8")]
    logging.basicConfig(level=level, handlers=handlers, format=fmt,
                        datefmt="%Y-%m-%d %H:%M:%S")
    return log_path


def dump_run_config(cfg: dict) -> str:
    """写入 run_config_{timestamp}.json。"""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(cfg["save_dir"], f"run_config_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2, default=str)
    print(f"[STEP MAIN.3] 转储 {path}")
    return path


def validate_no_leakage(cfg: dict) -> None:
    """时间边界校验（时间外推场景只校验时间，不校验患者重叠）。"""
    up_start = pd.Timestamp(cfg["upstream_start"])
    up_end = pd.Timestamp(cfg["upstream_end"])
    dn_start = pd.Timestamp(cfg["downstream_start"])
    dn_end = pd.Timestamp(cfg["downstream_end"])
    cutoff = pd.Timestamp(cfg["data_cutoff"])
    if not (up_start < up_end):
        raise ValueError(f"上游时间范围非法: {up_start} >= {up_end}")
    if not (dn_start <= dn_end):
        raise ValueError(f"下游时间范围非法: {dn_start} > {dn_end}")
    if up_end >= dn_start:
        raise ValueError(f"时间泄漏: upstream_end={up_end} >= downstream_start={dn_start}")
    if dn_end > cutoff:
        raise ValueError(f"下游结束 {dn_end} 超过数据截止日 {cutoff}")
    if int(cfg["prediction_window_days"]) <= 0:
        raise ValueError("prediction_window_days 必须为正整数")
    print("[STEP MAIN.4] 防泄漏校验... 通过")


# ---------------------------------------------------------------------------
# 指纹
# ---------------------------------------------------------------------------
def _sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


def file_fingerprint(path: str) -> str:
    """小文件(<100MB) SHA1；大文件 (size, sha1(first 1MB))。"""
    if not os.path.exists(path):
        return "missing:" + path
    size = os.path.getsize(path)
    if size < 100 * 1024 * 1024:
        with open(path, "rb") as f:
            return _sha1_bytes(f.read())
    with open(path, "rb") as f:
        head = f.read(1024 * 1024)
    return f"{size}:{_sha1_bytes(head)}"


def _code_hash(cfg: dict) -> dict:
    return {cf: file_fingerprint(os.path.join(cfg["work_dir"], cf)) for cf in CODE_FILES}


def compute_fingerprint(cfg: dict, stage: str) -> str:
    """计算阶段指纹：阶段参数分组 + 依赖文件指纹 + 代码哈希。"""
    groups = param_setting.get_fingerprint_groups(cfg)
    save_dir = cfg["save_dir"]
    parts = {"stage": stage}
    if stage == "load":
        parts["params"] = groups["load"]
        parts["inputs"] = {
            f: file_fingerprint(os.path.join(cfg["data_dir"], cfg[f]))
            for f in ("dynamic_file", "label_file", "static_file", "drug_file")}
    elif stage == "process":
        parts["params"] = groups["data"]
    elif stage.startswith("upstream_train"):
        parts["params"] = groups["upstream_train"]
        parts["data_dep"] = file_fingerprint(
            os.path.join(save_dir, "05_upstream_train.pt"))
    elif stage == "upstream_test":
        parts["params"] = groups["upstream_train"]
        parts["ckpt_deps"] = {
            seed: file_fingerprint(param_setting.get_upstream_ckpt_path(cfg, seed, True))
            for seed in cfg["seed_list"]}
    elif stage.startswith("downstream_train"):
        parts["params"] = groups["downstream_train"]
        parts["data_dep"] = file_fingerprint(
            os.path.join(save_dir, "07_downstream_trainval.pt"))
        parts["upstream_ckpt_dep"] = file_fingerprint(
            param_setting.get_upstream_ckpt_path(cfg, best=True))
    elif stage == "downstream_test":
        parts["params"] = groups["downstream_train"]
        parts["test_dep"] = file_fingerprint(
            os.path.join(save_dir, "08_downstream_test.pt"))
        parts["ckpt_deps"] = {
            seed: file_fingerprint(param_setting.get_downstream_ckpt_path(cfg, seed, True))
            for seed in cfg["seed_list"]}
    parts["code"] = _code_hash(cfg)
    raw = json.dumps(parts, sort_keys=True, default=str)
    return _sha1_bytes(raw.encode("utf-8"))


# ---------------------------------------------------------------------------
# StateManager
# ---------------------------------------------------------------------------
class StateManager:
    """阶段状态与指纹管理，持久化到 save/project_state.json。"""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.path = os.path.join(cfg["save_dir"], "project_state.json")
        self.state = {"stages": {}}
        self.load_state()

    def load_state(self) -> None:
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                self.state = json.load(f)
        self.state.setdefault("stages", {})

    def save_state(self) -> None:
        param_setting.write_json(self.path, self.state)

    def is_stage_done(self, stage: str, fp: str) -> bool:
        st = self.state["stages"].get(stage, {})
        return st.get("status") == "done" and st.get("fingerprint") == fp

    def mark_running(self, stage: str) -> None:
        self.state["stages"][stage] = {
            "status": "running",
            "started_at": datetime.datetime.now().isoformat()}
        self.save_state()

    def mark_done(self, stage: str, fp: str, outputs=None) -> None:
        self.state["stages"][stage] = {
            "status": "done", "fingerprint": fp,
            "finished_at": datetime.datetime.now().isoformat(),
            "outputs": outputs or []}
        self.save_state()

    def mark_failed(self, stage: str, error: str) -> None:
        self.state["stages"][stage] = {
            "status": "failed", "error": str(error)[:500],
            "failed_at": datetime.datetime.now().isoformat()}
        self.save_state()

    def should_rerun(self, stage: str, fp: str, force: bool = False) -> bool:
        if force:
            return True
        st = self.state["stages"].get(stage, {})
        if st.get("status") in ("running", "failed"):
            return True
        return not self.is_stage_done(stage, fp)


# ---------------------------------------------------------------------------
# 阶段执行
# ---------------------------------------------------------------------------
def _run_load_process(cfg, sm, force):
    """执行 load 与 process 阶段。"""
    fp_load = compute_fingerprint(cfg, "load")
    if sm.should_rerun("load", fp_load, force):
        print("[STEP MAIN.5] 阶段 load... 状态=running")
        sm.mark_running("load")
        data_load.run_data_load(cfg)
        sm.mark_done("load", fp_load)
    else:
        print("[STEP MAIN.5] 阶段 load... 状态=skipped")

    fp_proc = compute_fingerprint(cfg, "process")
    if sm.should_rerun("process", fp_proc, force):
        print("[STEP MAIN.6] 阶段 process... 状态=running")
        sm.mark_running("process")
        # 01/02 是 load 阶段用 torch.save 直接保存的 pandas.DataFrame（非权重）。
        # PyTorch>=2.6 的 torch.load 默认 weights_only=True 无法反序列化 DataFrame，
        # 故显式 weights_only=False。文件由本管线本地生成、来源可信。
        up_df = torch.load(os.path.join(cfg["save_dir"], "01_upstream_raw.pt"),
                           map_location="cpu", weights_only=False)
        dn_df = torch.load(os.path.join(cfg["save_dir"], "02_downstream_raw.pt"),
                           map_location="cpu", weights_only=False)
        static_raw = data_load.load_raw_tables(cfg)["static"]
        static_raw = data_load.clean_empty_rows(static_raw, "static")
        static_raw = data_load.clean_key_columns(static_raw, "static", ["patient_id"])
        static_raw = data_load.parse_dates(static_raw, ["birth_date"])
        data_process.run_data_process(up_df, dn_df, static_raw, cfg)
        sm.mark_done("process", fp_proc)
    else:
        print("[STEP MAIN.6] 阶段 process... 状态=skipped")


def _run_stage_module(module_name, func_name, cfg, stage, sm, force):
    """执行一个由模块承载的阶段；失败时记录状态并向上抛出，避免级联误跑。"""
    fp = compute_fingerprint(cfg, stage)
    if not sm.should_rerun(stage, fp, force):
        print(f"[STEP MAIN] 阶段 {stage}... 状态=skipped")
        return
    print(f"[STEP MAIN] 阶段 {stage}... 状态=running")
    sm.mark_running(stage)
    try:
        mod = __import__(module_name)
        getattr(mod, func_name)(cfg)
        sm.mark_done(stage, fp)
    except Exception as exc:  # noqa: BLE001
        sm.mark_failed(stage, str(exc))
        logger.exception("阶段 %s 失败: %s", stage, exc)
        raise


def _run_seed_stage(module_name, func_name, cfg, stage, sm, force, seed):
    """按种子执行的阶段：临时写入 _current_seed，结束后清除，避免影响后续阶段。"""
    cfg["_current_seed"] = seed
    try:
        _run_stage_module(module_name, func_name, cfg, stage, sm, force)
    finally:
        cfg.pop("_current_seed", None)


def run_pipeline(cfg: dict, sm: StateManager) -> None:
    """按阶段编排整个管线。"""
    force = cfg.get("force", False)
    only_stage = cfg.get("stage", "")

    def _go(stage):
        return (not only_stage) or only_stage == stage or stage.startswith(only_stage + ":")

    # load / process
    if _go("load") or _go("process"):
        _run_load_process(cfg, sm, force)

    # 上游训练（多种子，逐种子分阶段）
    for seed in cfg["seed_list"]:
        st = f"upstream_train:{cfg['upstream_method']}:{seed}"
        if _go(st):
            _run_seed_stage("train_upstream", "main", cfg, st, sm, force, seed)
    # 上游测试
    st = f"upstream_test:{cfg['upstream_method']}"
    if _go(st):
        _run_stage_module("test_upstream", "main", cfg, st, sm, force)

    # 下游训练（多种子，逐种子分阶段；HPO/CV 由模块内部缓存复用）
    for seed in cfg["seed_list"]:
        st = f"downstream_train:{cfg['downstream_model']}:{seed}"
        if _go(st):
            _run_seed_stage("train_downstream", "main", cfg, st, sm, force, seed)
    # 下游测试
    st = f"downstream_test:{cfg['downstream_model']}"
    if _go(st):
        _run_stage_module("test_downstream", "main", cfg, st, sm, force)


# ---------------------------------------------------------------------------
# 报告与摘要
# ---------------------------------------------------------------------------
def generate_report_table(cfg: dict) -> None:
    """从 evaluation_results.csv 聚合生成 report_table.csv（行=上游×下游）。"""
    src = param_setting.get_evaluation_results_path(cfg)
    if not os.path.exists(src):
        print("[STEP MAIN.N] 无 evaluation_results.csv，跳过报告生成")
        return
    df = pd.read_csv(src)
    if "stage" not in df.columns:
        print("[STEP MAIN.N] evaluation_results.csv 缺少 stage 列，跳过报告生成")
        return
    test_df = df[df["stage"] == "downstream_test"]
    if test_df.empty:
        print("[STEP MAIN.N] 无 downstream_test 行，跳过报告生成")
        return
    numeric = ["mae", "mae_ci_low", "mae_ci_high", "rmse", "rmse_ci_low",
               "rmse_ci_high", "r2", "r2_ci_low", "r2_ci_high", "pearson",
               "pearson_ci_low", "pearson_ci_high", "spearman",
               "spearman_ci_low", "spearman_ci_high"]
    for c in numeric:
        if c in test_df.columns:
            test_df = test_df.assign(**{c: pd.to_numeric(test_df[c], errors="coerce")})
    agg = test_df.groupby(["upstream_method", "downstream_model"]).agg(
        n_seeds=("seed", "nunique"),
        mae_mean=("mae", "mean"), mae_std=("mae", "std"),
        mae_ci_low=("mae_ci_low", "mean"), mae_ci_high=("mae_ci_high", "mean"),
        rmse_mean=("rmse", "mean"), r2_mean=("r2", "mean"),
        pearson_mean=("pearson", "mean"), spearman_mean=("spearman", "mean"),
    ).reset_index()
    out = param_setting.get_report_table_path(cfg)
    agg.to_csv(out, index=False)
    print(f"[STEP MAIN.N] 生成 {out}")


def _safe_csv_count(path: str):
    """返回 CSV 行数（文件不存在时返回 None）。"""
    if not os.path.exists(path):
        return None
    try:
        return len(pd.read_csv(path))
    except Exception:  # noqa: BLE001
        return None


def print_final_summary(cfg: dict, sm: StateManager) -> None:
    """打印最终摘要（规模、词表、指标、阶段状态、输出文件）。"""
    print("[STEP MAIN.N+1] 最终摘要...")
    print(f"  upstream_method={cfg['upstream_method']} "
          f"downstream_model={cfg['downstream_model']} seed_list={cfg['seed_list']} "
          f"seq_length={cfg['seq_length']}")
    try:
        pre = torch.load(os.path.join(cfg["save_dir"], "preprocess_params.pt"),
                         map_location="cpu", weights_only=False)
        vocab = pre.get("token_vocab", {})
        print(f"  词表大小={len(vocab)} RARE 数量={len(pre.get('rare_mapping', {}))} "
              f"数值 token 类数={len(pre.get('numeric_minmax', {}))}")
    except Exception as exc:  # noqa: BLE001
        print(f"  （预处理参数读取失败: {exc}）")
    print(f"  上游训练样本={_safe_csv_count(os.path.join(cfg['save_dir'], '05_upstream_train.csv'))} "
          f"上游验证样本={_safe_csv_count(os.path.join(cfg['save_dir'], '06_upstream_val.csv'))} "
          f"下游 trainval 地标={_safe_csv_count(os.path.join(cfg['save_dir'], '07_downstream_trainval.csv'))} "
          f"下游 test 地标={_safe_csv_count(os.path.join(cfg['save_dir'], '08_downstream_test.csv'))}")
    # 指标
    src = param_setting.get_evaluation_results_path(cfg)
    if os.path.exists(src):
        try:
            df = pd.read_csv(src)
            sub = df[(df["stage"] == "downstream_test")
                     & (df["upstream_method"] == cfg["upstream_method"])
                     & (df["downstream_model"] == cfg["downstream_model"])]
            if not sub.empty:
                print("  最新下游测试指标：")
                for _, r in sub.iterrows():
                    print(f"    seed={r['seed']} MAE={r['mae']} "
                          f"[{r['mae_ci_low']}, {r['mae_ci_high']}] "
                          f"RMSE={r['rmse']} R2={r['r2']}")
        except Exception as exc:  # noqa: BLE001
            print(f"  （评估结果读取失败: {exc}）")
    # 阶段状态
    print("  阶段状态：")
    for stage, st in sm.state.get("stages", {}).items():
        print(f"    {stage}: {st.get('status')}")
    # 输出文件
    print("  输出文件：")
    for d in (cfg["save_dir"], cfg["model_dir"], cfg["state_dir"], cfg["logs_dir"]):
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                p = os.path.join(d, f)
                if os.path.isfile(p):
                    print(f"    {p}")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def main() -> None:
    args = param_setting.parse_args()
    cfg = param_setting.get_config(args)
    print(f"[STEP MAIN.1] 解析参数... upstream_method={cfg['upstream_method']} "
          f"downstream_model={cfg['downstream_model']} seeds={cfg['seed_list']}")
    log_path = setup_logging(cfg)
    print(f"[STEP MAIN.2] 创建目录、配置日志... 日志={log_path}")
    dump_run_config(cfg)
    validate_no_leakage(cfg)

    sm = StateManager(cfg)
    try:
        run_pipeline(cfg, sm)
    except Exception as exc:  # noqa: BLE001
        logger.exception("管线失败: %s", exc)
        print_final_summary(cfg, sm)
        sys.exit(1)

    if cfg.get("generate_report", True):
        generate_report_table(cfg)
    print_final_summary(cfg, sm)


if __name__ == "__main__":
    main()
