# -*- coding: utf-8 -*-
"""
param_setting.py
================

集中管理医学时序数据预处理管线的所有可调参数。

本模块使用 argparse 解析命令行参数，提供两个入口：
- parse_args(): 返回 argparse.Namespace，供 main.py 直接使用
- get_config():  返回配置字典（dict），供各业务模块以 cfg["key"] 方式访问

所有可调参数都在这里集中定义，data_load / data_process / main 只读取配置，
避免参数散落在各业务模块中。默认值即为本项目的正式配置。

依赖: argparse, os
"""

import argparse
import os


def _build_parser() -> argparse.ArgumentParser:
    """构建并返回 argparse 解析器。"""
    parser = argparse.ArgumentParser(
        description="CLP 医学时序数据预处理管线参数",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 路径类参数
    parser.add_argument("--work_dir", default=r"D:\PythonProgram\CLP", help="工作路径")
    parser.add_argument("--data_dir", default=r"D:\PythonProgram\CLP\data", help="数据文件路径")
    parser.add_argument("--save_dir", default=r"D:\PythonProgram\CLP\save", help="预处理输出保存路径")

    # 输入文件名
    parser.add_argument("--dynamic_file", default="dynamic.csv", help="动态随访文件名")
    parser.add_argument("--label_file", default="label.csv", help="标签文件名")
    parser.add_argument("--static_file", default="static.csv", help="静态特征文件名")
    parser.add_argument("--drug_file", default="drug.csv", help="药物文件名")

    # 时间范围
    parser.add_argument("--upstream_start", default="2012-01-01", help="上游起始日期")
    parser.add_argument("--upstream_end", default="2022-12-31", help="上游结束日期")
    parser.add_argument("--downstream_start", default="2023-01-01", help="下游起始日期")
    parser.add_argument("--downstream_end", default="2024-12-31", help="下游结束日期")
    parser.add_argument("--data_cutoff", default="2024-12-31", help="数据截止日")

    # 建模参数
    parser.add_argument("--prediction_window_days", type=int, default=90, help="预测窗口天数")
    parser.add_argument("--seq_length", type=int, default=512, help="序列最大长度 N")
    parser.add_argument("--label_agg", choices=["mean", "median", "first"],
                        default="mean", help="标签聚合方式")

    # 词表与划分参数
    parser.add_argument("--rare_threshold", type=int, default=10, help="低频 Token 归并阈值")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="下游测试集比例")
    parser.add_argument("--upstream_val_ratio", type=float, default=0.1, help="上游验证集比例")
    parser.add_argument("--random_seed", type=int, default=42, help="随机种子")

    # 运行参数
    parser.add_argument("--log_level", default="INFO", help="日志级别")
    parser.add_argument("--pt_dtype", default="float32", help="张量默认 dtype")
    parser.add_argument("--skip_load", action="store_true",
                        help="跳过数据加载，从已保存 01/02 raw .pt 读取")
    return parser


def parse_args(argv=None) -> argparse.Namespace:
    """解析命令行参数，返回 argparse.Namespace。

    参数:
        argv: 可选，覆盖 sys.argv 的参数列表（便于测试）。
    返回:
        argparse.Namespace: 所有可调参数。
    """
    return _build_parser().parse_args(argv)


def get_config(args: argparse.Namespace = None) -> dict:
    """把参数转为配置字典。

    参数:
        args: 可选，argparse.Namespace；为 None 时自动调用 parse_args()。
    返回:
        dict: 以参数名为键的配置字典。
    """
    if args is None:
        args = parse_args()
    cfg = dict(vars(args))
    # 便于业务模块统一拼接路径
    cfg["upstream_start"] = str(cfg["upstream_start"])
    cfg["upstream_end"] = str(cfg["upstream_end"])
    cfg["downstream_start"] = str(cfg["downstream_start"])
    cfg["downstream_end"] = str(cfg["downstream_end"])
    cfg["data_cutoff"] = str(cfg["data_cutoff"])
    return cfg


if __name__ == "__main__":
    # 直接运行本文件时打印当前配置，便于人工核对
    import json
    print(json.dumps(get_config(), indent=2, ensure_ascii=False, default=str))
