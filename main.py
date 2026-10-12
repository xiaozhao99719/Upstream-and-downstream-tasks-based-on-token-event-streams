# -*- coding: utf-8 -*-
"""
main.py
=======

项目主控脚本：控制整个数据预处理执行过程。

流程:
  1. 解析参数 param_setting.parse_args()；
  2. 创建保存目录、配置日志（控制台 + save/preprocess.log）；
  3. 数据加载清洗 data_load.run_data_load(cfg)；
  4. 数据预处理 data_process.run_data_process(...)；
  5. 打印最终摘要与输出文件清单。

支持 --skip_load：若已加载过，从 01_upstream_raw.pt / 02_downstream_raw.pt 读取。
每个步骤打印时间戳，异常统一捕获并给出明确错误。

依赖: os, sys, logging, datetime, torch, param_setting, data_load, data_process
"""

import os
import sys
import logging
from datetime import datetime

import torch

import param_setting
import data_load
import data_process


def setup_logging(cfg: dict) -> str:
    """创建保存目录并配置日志（控制台 + 文件 preprocess.log）。

    输入:
        cfg: 配置字典。
    返回:
        日志文件路径。
    """
    os.makedirs(cfg["save_dir"], exist_ok=True)
    log_path = os.path.join(cfg["save_dir"], "preprocess.log")
    level = getattr(logging, str(cfg["log_level"]).upper(), logging.INFO)
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    handlers = [logging.StreamHandler(sys.stdout)]
    handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    logging.basicConfig(level=level, handlers=handlers, format=fmt,
                        datefmt="%Y-%m-%d %H:%M:%S")
    return log_path


def _load_from_pt(tag: str, cfg: dict):
    """从 save_dir 读取已保存的 raw .pt（DataFrame 对象）。"""
    path = os.path.join(cfg["save_dir"], tag)
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到 {path}，无法执行 --skip_load")
    obj = torch.load(path, map_location="cpu")
    if not hasattr(obj, "columns"):
        raise TypeError(f"{path} 不是 DataFrame，无法作为 raw 表使用")
    return obj


def run(cfg: dict) -> None:
    """执行完整预处理流程。"""
    logger = logging.getLogger(__name__)
    log_path = setup_logging(cfg)
    logger.info("===== CLP 预处理开始 | 时间 %s =====",
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("工作路径: %s | 数据路径: %s | 保存路径: %s",
                cfg["work_dir"], cfg["data_dir"], cfg["save_dir"])

    # 数据加载
    if cfg["skip_load"]:
        logger.info("--skip_load 生效：从 01/02 raw .pt 读取已加载数据")
        upstream_df = _load_from_pt("01_upstream_raw.pt", cfg)
        downstream_df = _load_from_pt("02_downstream_raw.pt", cfg)
        static_df = data_load.load_raw_tables(cfg)["static"]
        static_df = data_load.clean_empty_rows(static_df, "static")
        static_df = data_load.clean_key_columns(static_df, "static", ["patient_id"])
        static_df = data_load.parse_dates(static_df, ["birth_date"])
        logger.info("读取上游行 %d, 下游行 %d, 静态行 %d",
                    len(upstream_df), len(downstream_df), len(static_df))
    else:
        logger.info(">>> 阶段一: 数据加载与清洗")
        upstream_df, downstream_df, static_df, _ = data_load.run_data_load(cfg)

    # 数据预处理
    logger.info(">>> 阶段二: Token 化 / 参数学习 / 划分")
    result = data_process.run_data_process(upstream_df, downstream_df, static_df, cfg)

    # 最终摘要
    tensors = result["tensors"]
    params = result["params"]
    logger.info("===== 最终摘要 =====")
    logger.info("上游训练患者数: %d", len(tensors["upstream_train"]["patient_id"]))
    logger.info("上游验证患者数: %d", len(tensors["upstream_val"]["patient_id"]))
    logger.info("下游训练+验证地标数: %d",
                len(tensors["downstream_trainval"]["landmark_id"]))
    logger.info("下游测试地标数: %d", len(tensors["downstream_test"]["landmark_id"]))
    logger.info("词表大小: %d | RARE 数量: %d",
                len(params["token_vocab"]), len(params["rare_mapping"]))
    logger.info("序列长度 N: %d", cfg["seq_length"])

    logger.info("----- 输出文件清单 -----")
    for f in sorted(os.listdir(cfg["save_dir"])):
        fpath = os.path.join(cfg["save_dir"], f)
        logger.info("  %-30s %10d bytes", f, os.path.getsize(fpath))
    logger.info("===== 预处理结束 | 时间 %s | 日志 %s =====",
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"), log_path)


def main() -> None:
    """入口：解析参数、执行流程、捕获异常。"""
    args = param_setting.parse_args()
    cfg = param_setting.get_config(args)
    try:
        run(cfg)
    except Exception as exc:  # noqa: BLE001
        logging.getLogger(__name__).exception("预处理失败: %s", exc)
        print(f"[ERROR] 预处理失败: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
