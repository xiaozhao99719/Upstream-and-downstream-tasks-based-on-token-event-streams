# -*- coding: utf-8 -*-
"""
param_setting.py
================

集中管理医学时序对比学习管线（预处理 + 上游自监督预训练 + 下游监督回归）的
所有可调参数。

本模块使用 argparse 解析命令行参数，提供两个入口：

- parse_args(argv=None)：返回 argparse.Namespace，供 main.py 直接使用；
- get_config(args=None)：返回配置字典（dict），供各业务模块以 cfg["key"] 访问。

参数分层（上游 / 下游训练参数完全分开，便于分别调参）：

    路径/数据/预处理   work_dir、data_dir、save_dir、时间窗、seq_length、词表、
                     划分比例、pt_dtype 等；
    共享编码器结构      d_model / n_heads / n_layers / dropout
                     （下游 reuse 必须与上游同构，故三处不拆分）；
    上游训练           up_*：lr、weight_decay、batch_size、epochs、patience、
                     grad_clip、scheduler、warmup_epochs、grad_accum_steps、
                     amp、mask_ratio、tau、损失权重策略、void_ratio、
                     amcpc_steps、lambda_*；
    下游训练           down_*：lr、weight_decay、batch_size、epochs、patience、
                     grad_clip、scheduler、warmup_epochs、grad_accum_steps、
                     amp、dropout、loss、hpo_max_epochs；
    HPO/最终重训        hpo_*、final_*、cv_folds；
    Bootstrap/报告      bootstrap_*、generate_report、eval_new_patients_subset。

阶段键：

- 预处理：load / process；
- 上游：upstream_train:{method}:{seed}、upstream_test:{method}；
- 下游：downstream_train:{model}:{seed}、downstream_test:{model}。

依赖: os, json, hashlib, datetime, argparse, random, numpy, pandas, torch
"""

import os
import json
import hashlib
import math
import warnings
import argparse
import random
import logging
import contextlib

import numpy as np
import pandas as pd
import torch

# torch 2.0.x 的 MHA 快速路径在推理（no_grad）阶段会对传入的 bool padding mask 重复
# 发出 "Converting mask without torch.bool dtype to bool" 性能告警（由 torch 内部把
# bool mask 转成内核所需格式所致，与本项目输入无关）。精确过滤该条消息，避免噪声
# 掩盖真实问题；其余告警不受影响。
warnings.filterwarnings(
    "ignore",
    message="Converting mask without torch.bool dtype to bool.*",
    category=UserWarning,
)


# ---------------------------------------------------------------------------
# 枚举合法取值
# ---------------------------------------------------------------------------
UPSTREAM_METHODS = ["void", "amcpc", "ts2vec", "tstcc", "all"]
DOWNSTREAM_MODELS = ["strats", "duett", "gru_d", "lstm", "transformer"]
DOWNSTREAM_INITS = ["upstream", "scratch"]
DOWNSTREAM_ARCHS = ["reuse", "full"]
SCHEDULERS = ["cosine", "plateau", "none"]
LOSS_WEIGHTINGS = ["fixed", "uncertainty", "gradnorm"]
TOKEN_LOSS_WEIGHTINGS = ["none", "freq_inv_sqrt", "freq_inv"]
HPO_STRATEGIES = ["grid", "random", "none"]
FINAL_EPOCHS_MODES = ["cv_median", "fixed"]
UP_SELECT_METRICS = ["mask_loss", "val_loss"]
DOWN_LOSSES = ["l1", "smooth_l1", "mse"]
PT_DTYPES = ["float16", "float32", "float64"]

# evaluation_results.csv 的统一列顺序（所有阶段共用，避免追加行错位）
EVAL_COLUMNS = [
    "timestamp", "stage", "upstream_method", "downstream_model", "downstream_arch",
    "seed", "dataset", "loss", "token_acc", "value_mae",
    "mae", "mae_ci_low", "mae_ci_high",
    "rmse", "rmse_ci_low", "rmse_ci_high",
    "r2", "r2_ci_low", "r2_ci_high",
    "pearson", "pearson_ci_low", "pearson_ci_high",
    "spearman", "spearman_ci_low", "spearman_ci_high",
    "bootstrap_n", "bootstrap_unit", "best_hparams", "notes",
]


def _build_parser() -> argparse.ArgumentParser:
    """构建并返回 argparse 解析器。"""
    parser = argparse.ArgumentParser(
        description="CLP 医学时序对比学习管线参数",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ===== 路径类参数 =====
    parser.add_argument("--work_dir", default=r"D:\PythonProgram\CLP", help="工作路径")
    parser.add_argument("--data_dir", default=r"D:\PythonProgram\CLP\data", help="数据文件路径")
    parser.add_argument("--save_dir", default=r"D:\PythonProgram\CLP\save", help="输出保存路径")

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

    # 建模参数（预处理）
    parser.add_argument("--prediction_window_days", type=int, default=90, help="预测窗口天数")
    parser.add_argument("--seq_length", type=int, default=512, help="序列最大长度 N")
    parser.add_argument("--label_agg", choices=["mean", "median", "first"],
                        default="mean", help="标签聚合方式")

    # 词表与划分参数
    parser.add_argument("--rare_threshold", type=int, default=10, help="低频 Token 归并阈值")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="下游测试集比例")
    parser.add_argument("--upstream_val_ratio", type=float, default=0.1, help="上游验证集比例")
    parser.add_argument("--random_seed", type=int, default=42, help="全局随机种子")
    parser.add_argument("--seeds", default="42", help="种子列表，逗号分隔，如 42,43,44")

    # 运行环境
    parser.add_argument("--log_level", default="INFO", help="日志级别")
    parser.add_argument("--pt_dtype", default="float32", choices=PT_DTYPES,
                        help="张量默认 dtype（token_ids/掩码不受影响）")
    parser.add_argument("--device", default="cuda", help="训练设备 cuda/cpu（不可用时自动回退 cpu）")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader 线程数")
    parser.add_argument("--skip_load", action="store_true",
                        help="跳过数据加载，从已保存 01/02 raw .pt 读取")

    # ===== 核心选择参数 =====
    parser.add_argument("--upstream_method", default="void", choices=UPSTREAM_METHODS,
                        help="上游对比学习方法")
    parser.add_argument("--downstream_model", default="duett", choices=DOWNSTREAM_MODELS,
                        help="下游模型")
    parser.add_argument("--downstream_init", default="upstream", choices=DOWNSTREAM_INITS,
                        help="下游编码器是否用上游权重初始化")
    parser.add_argument("--downstream_arch", default="reuse", choices=DOWNSTREAM_ARCHS,
                        help="reuse：复用上游编码器 + 下游头；full：下游自建完整模型（忽略上游权重）")
    parser.add_argument("--freeze_upstream", action="store_true",
                        help="下游训练时冻结上游编码器（仅 reuse 生效）")

    # ===== 共享编码器结构参数（上下游必须一致） =====
    parser.add_argument("--d_model", type=int, default=128, help="隐藏维度")
    parser.add_argument("--n_heads", type=int, default=4, help="注意力头数")
    parser.add_argument("--n_layers", type=int, default=2, help="Transformer 层数")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="编码器内部 Dropout（上下游共享结构）")

    # ===== 上游训练参数 =====
    parser.add_argument("--up_lr", type=float, default=1e-4, help="上游学习率")
    parser.add_argument("--up_weight_decay", type=float, default=1e-5, help="上游权重衰减")
    parser.add_argument("--up_batch_size", type=int, default=32, help="上游 batch size")
    parser.add_argument("--up_epochs", type=int, default=100, help="上游最大训练轮数")
    parser.add_argument("--up_patience", type=int, default=15, help="上游早停耐心轮数")
    parser.add_argument("--up_grad_clip", type=float, default=1.0, help="上游梯度裁剪范数")
    parser.add_argument("--up_scheduler", default="cosine", choices=SCHEDULERS,
                        help="上游学习率调度")
    parser.add_argument("--up_warmup_epochs", type=int, default=5,
                        help="上游余弦调度 warmup 轮数")
    parser.add_argument("--up_grad_accum_steps", type=int, default=1, help="上游梯度累积步数")
    parser.add_argument("--up_amp", action="store_true", default=True, help="上游启用混合精度")
    parser.add_argument("--up_no_amp", action="store_false", dest="up_amp",
                        help="上游关闭混合精度")
    parser.add_argument("--up_amp_init_scale", type=float, default=2048.0,
                        help="上游 AMP 初始缩放；torch 默认 65536 偏大易首步溢出，"
                             "默认改用 2048（若仍溢出可继续调小，或 --up_no_amp）")
    parser.add_argument("--up_mask_ratio", type=float, default=0.15,
                        help="上游 masked event modeling 的 mask 比例")
    parser.add_argument("--up_tau", type=float, default=0.1, help="上游对比损失温度系数")
    parser.add_argument("--up_log_sigma_clip", type=float, default=2.0,
                        help="不确定性加权的 logσ 截断范围（防止 σ 过小导致权重爆炸，"
                             "σ∈[e^-c, e^c]）；仅 up_loss_weighting=uncertainty 生效")
    parser.add_argument("--up_select_metric", default="mask_loss", choices=UP_SELECT_METRICS,
                        help="上游最优模型/早停监控指标")

    # 上游损失与权重
    parser.add_argument("--up_loss_weighting", default="uncertainty", choices=LOSS_WEIGHTINGS,
                        help="上游多损失加权策略")
    parser.add_argument("--up_token_loss_weighting", default="freq_inv_sqrt",
                        choices=TOKEN_LOSS_WEIGHTINGS, help="上游 token 交叉熵频率加权")
    parser.add_argument("--up_void_ratio", type=float, default=0.15, help="Void epoch 插入比例")
    parser.add_argument("--up_amcpc_steps", type=int, default=3, help="AMCPC 预测步长候选上限")
    parser.add_argument("--lambda_mask", type=float, default=1.0,
                        help="固定权重：mask token/value（loss_weighting=fixed）")
    parser.add_argument("--lambda_void", type=float, default=0.5,
                        help="固定权重：void epoch 对比")
    parser.add_argument("--lambda_amcpc", type=float, default=0.5, help="固定权重：AMCPC")
    parser.add_argument("--lambda_ts2vec", type=float, default=0.5, help="固定权重：TS2Vec")
    parser.add_argument("--lambda_tstcc", type=float, default=0.5, help="固定权重：TSTCC")

    # ===== 下游训练参数 =====
    parser.add_argument("--down_lr", type=float, default=1e-4,
                        help="下游学习率（--no_hpo 时的默认值）")
    parser.add_argument("--down_weight_decay", type=float, default=1e-5,
                        help="下游权重衰减（--no_hpo 时的默认值）")
    parser.add_argument("--down_batch_size", type=int, default=32,
                        help="下游 batch size（--no_hpo 时的默认值）")
    parser.add_argument("--down_epochs", type=int, default=100,
                        help="下游折内训练最大轮数（早停上限）")
    parser.add_argument("--down_patience", type=int, default=15, help="下游折内早停耐心轮数")
    parser.add_argument("--down_grad_clip", type=float, default=1.0, help="下游梯度裁剪范数")
    parser.add_argument("--down_scheduler", default="cosine", choices=SCHEDULERS,
                        help="下游学习率调度")
    parser.add_argument("--down_warmup_epochs", type=int, default=5,
                        help="下游余弦调度 warmup 轮数")
    parser.add_argument("--down_grad_accum_steps", type=int, default=1, help="下游梯度累积步数")
    parser.add_argument("--down_amp", action="store_true", default=True, help="下游启用混合精度")
    parser.add_argument("--down_no_amp", action="store_false", dest="down_amp",
                        help="下游关闭混合精度")
    parser.add_argument("--down_amp_init_scale", type=float, default=2048.0,
                        help="下游 AMP 初始缩放；torch 默认 65536 偏大易首步溢出，"
                             "默认改用 2048（若仍溢出可继续调小，或 --down_no_amp）")
    parser.add_argument("--down_dropout", type=float, default=0.2,
                        help="下游回归头 Dropout（HPO 调参目标）")
    parser.add_argument("--down_loss", default="l1", choices=DOWN_LOSSES,
                        help="下游回归损失")
    parser.add_argument("--down_hpo_max_epochs", type=int, default=30,
                        help="HPO 单折训练最大轮数上限")

    # ===== HPO 参数 =====
    parser.add_argument("--hpo_strategy", default="grid", choices=HPO_STRATEGIES,
                        help="HPO 搜索策略")
    parser.add_argument("--hpo_trials", type=int, default=27, help="随机搜索次数")
    parser.add_argument("--no_hpo", action="store_true", help="跳过 HPO，使用 down_* 默认值")
    parser.add_argument("--hpo_fold_ratio", type=float, default=0.6,
                        help="前 hpo_fold_ratio 折做 HPO，其余折做无偏评估")
    parser.add_argument("--cv_folds", type=int, default=5, help="expanding window CV 折数")
    parser.add_argument("--hpo_lr_grid", default="1e-4,5e-5,1e-5", help="lr 搜索空间")
    parser.add_argument("--hpo_dropout_grid", default="0.1,0.2,0.3",
                        help="down_dropout 搜索空间")
    parser.add_argument("--hpo_wd_grid", default="0,1e-5,1e-4", help="weight_decay 搜索空间")
    parser.add_argument("--hpo_bs_grid", default="16,32,64", help="batch_size 搜索空间")
    parser.add_argument("--hpo_primary_metric", default="mae", choices=["mae"],
                        help="HPO 主指标（当前支持 mae）")

    # ===== 最终重训与 Bootstrap 参数 =====
    parser.add_argument("--final_epochs_mode", default="cv_median",
                        choices=FINAL_EPOCHS_MODES, help="最终重训轮数确定方式")
    parser.add_argument("--final_fixed_epochs", type=int, default=60,
                        help="final_epochs_mode=fixed 时的固定轮数")
    parser.add_argument("--final_watch_ratio", type=float, default=0.05,
                        help="最终重训留出监控比例（仅监控，不参与梯度）")
    parser.add_argument("--bootstrap_n", type=int, default=1000, help="Bootstrap 次数")
    parser.add_argument("--bootstrap_unit", default="patient",
                        choices=["patient"], help="Bootstrap 单位（固定 patient）")

    # ===== 其他 =====
    parser.add_argument("--generate_report", action="store_true", default=True,
                        help="生成 report_table.csv")
    parser.add_argument("--no_generate_report", action="store_false", dest="generate_report",
                        help="不生成 report_table.csv")
    parser.add_argument("--eval_new_patients_subset", action="store_true",
                        help="是否额外评估新患者子集")
    parser.add_argument("--force", action="store_true", help="忽略状态指纹/断点，强制重跑")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="自动识别已完成/中断状态并断点续跑（默认开启）")
    parser.add_argument("--no_resume", action="store_false", dest="resume",
                        help="关闭断点续跑（每个阶段都从头开始）")
    parser.add_argument("--stage", default="",
                        help="只运行指定阶段（如 upstream_train:void:42），默认全流程")
    return parser


def parse_args(argv=None) -> argparse.Namespace:
    """解析命令行参数，返回 argparse.Namespace。

    输入:
        argv: 可选，覆盖 sys.argv 的参数列表（便于测试）。
    返回:
        argparse.Namespace: 所有可调参数（枚举合法性已由 choices 校验）。
    """
    return _build_parser().parse_args(argv)


def parse_grid(grid_str: str):
    """解析 "1e-4,5e-5" 形式的搜索空间为 float 列表。

    输入:
        grid_str: 逗号分隔字符串。
    返回:
        list[float]: 解析后的数值列表。
    """
    return [float(x.strip()) for x in str(grid_str).split(",") if x.strip() != ""]


def get_config(args: argparse.Namespace = None) -> dict:
    """把参数转为配置字典，并补充派生字段。

    - seeds 解析为 int 列表 seed_list；
    - HPO grid 解析为 float/int 列表；
    - 派生目录路径（model_dir/state_dir/logs_dir）。
    输入:
        args: 可选，argparse.Namespace；为 None 时自动调用 parse_args()。
    返回:
        dict: 配置字典。
    """
    if args is None:
        args = parse_args()
    cfg = dict(vars(args))
    # 日期字符串化
    for k in ("upstream_start", "upstream_end", "downstream_start",
              "downstream_end", "data_cutoff"):
        cfg[k] = str(cfg[k])
    # seeds -> list[int]
    cfg["seed_list"] = [int(s.strip()) for s in str(cfg["seeds"]).split(",") if s.strip() != ""]
    if not cfg["seed_list"]:
        raise ValueError("--seeds 不能为空")
    # HPO grids -> list
    cfg["hpo_lr_grid_parsed"] = parse_grid(cfg["hpo_lr_grid"])
    cfg["hpo_dropout_grid_parsed"] = parse_grid(cfg["hpo_dropout_grid"])
    cfg["hpo_wd_grid_parsed"] = parse_grid(cfg["hpo_wd_grid"])
    cfg["hpo_bs_grid_parsed"] = [int(x) for x in parse_grid(cfg["hpo_bs_grid"])]
    # 派生目录
    cfg["model_dir"] = os.path.join(cfg["save_dir"], "model")
    cfg["state_dir"] = os.path.join(cfg["save_dir"], "state")
    cfg["logs_dir"] = os.path.join(cfg["save_dir"], "logs")
    return cfg


def resolve_seeds(cfg: dict) -> list:
    """返回本次调用需要训练的种子列表。

    main.py 会把"当前种子"写入 cfg["_current_seed"]（用于按种子分阶段的断点续跑），
    此时只训练该种子；直接运行 train_*.py 时则训练 cfg["seed_list"] 中的全部种子。
    """
    cur = cfg.get("_current_seed")
    return [int(cur)] if cur is not None else [int(s) for s in cfg["seed_list"]]


def set_random_seed(seed: int) -> None:
    """固定 Python / NumPy / PyTorch 随机种子，保证可复现。

    输入:
        seed: 随机种子。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # 卷积/注意力算子在不同实现间存在非确定性，固定后处理以保证可复现。
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# AMP 兼容层（torch<2.3 没有 torch.amp.GradScaler）
# ---------------------------------------------------------------------------
class _NullScaler:
    """GradScaler 的兼容空实现（不缩放、不做 unscale）。"""

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        return None

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        return None

    def state_dict(self):
        return None

    def load_state_dict(self, state_dict):
        return None


def autocast_ctx(device, enabled: bool):
    """返回自动混合精度上下文；enabled=False 时为 nullcontext。

    自动兼容 torch.amp.autocast / torch.autocast 两种 API 位置。
    """
    if not enabled:
        return contextlib.nullcontext()
    amp_mod = getattr(torch, "amp", None)
    if amp_mod is not None and hasattr(amp_mod, "autocast"):
        return amp_mod.autocast(device_type=str(device.type), enabled=True)
    return torch.autocast(device_type=str(device.type), enabled=True)


def make_grad_scaler(device, enabled: bool, init_scale=None):
    """构建 GradScaler，兼容 torch<2.3（无 torch.amp.GradScaler）。

    仅当 device 为 CUDA 且 enabled=True 时才真正启用缩放；否则返回等价的空实现，
    保证训练循环代码在有无 AMP 时完全一致。
    init_scale: 可选的初始缩放（None/0 表示使用 torch 默认值 65536）。
    """
    use = bool(enabled) and str(device.type) == "cuda"
    kwargs = {"enabled": use}
    if init_scale is not None and float(init_scale) > 0:
        kwargs["init_scale"] = float(init_scale)
    amp_mod = getattr(torch, "amp", None)
    if amp_mod is not None and hasattr(amp_mod, "GradScaler"):
        try:
            return amp_mod.GradScaler(device.type, **kwargs)
        except (TypeError, AttributeError):
            pass
    cuda_amp = getattr(getattr(torch, "cuda", None), "amp", None)
    if cuda_amp is not None and hasattr(cuda_amp, "GradScaler"):
        try:
            return cuda_amp.GradScaler(**kwargs)
        except (TypeError, AttributeError):
            pass
    return _NullScaler()


def cosine_with_warmup_lambda(total_steps: int, warmup_steps: int, min_factor: float = 0.01):
    """构造 LambdaLR 的 lr_lambda：先线性 warmup，再余弦衰减到 min_factor。

    用单个 LambdaLR 实现 warmup+cosine（避免 SequentialLR 链式调度在不同 torch
    版本下的兼容性问题），返回值乘在 base_lr 上。
    """
    total = max(1, int(total_steps))
    warm = max(0, int(warmup_steps))
    floor = float(min_factor)

    def lr_lambda(step):
        step = int(step)
        if warm > 0 and step < warm:
            return max(floor, float(step + 1) / float(warm))
        span = max(1, total - warm)
        progress = min(1.0, max(0.0, float(step - warm) / float(span)))
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return lr_lambda


class _StepTracker:
    """统计 optimizer 实际执行 step 的次数。"""

    def __init__(self, optimizer):
        self.count = 0
        inner_step = optimizer.step
        # torch 的 LRScheduler 在构造时会包装 optimizer.step 并给包装器打上
        # `_wrapped_by_lr_sched` 标记；这里必须把这个标记透传下去，否则 torch 的
        # 顺序自检会误判成"调度器初始化后又有人覆写了 optimizer.step"。
        wrapped_by_lr_sched = getattr(inner_step, "_wrapped_by_lr_sched", False)

        def counted_step(*args, **kwargs):
            out = inner_step(*args, **kwargs)
            self.count += 1
            return out

        if wrapped_by_lr_sched:
            counted_step._wrapped_by_lr_sched = True
        optimizer.step = counted_step

    def steps(self) -> int:
        return self.count


def attach_step_tracker(optimizer) -> None:
    """给 optimizer 装上 step 计数器（幂等），用于判断每次更新是否真实发生。

    必须在学习率调度器**构造之后**调用：torch 的 LRScheduler 构造时会包装
    optimizer.step 以记录 _opt_called，这里再包一层即可精确计数。
    原因：torch 2.9 不再提供 optimizer._step_count，无法用 torch 内部属性计数，
    而 AMP 下 GradScaler 因梯度溢出跳过更新时不会进入 optimizer.step，
    据此可避免"更新被跳过却推进了学习率调度"。
    """
    if getattr(optimizer, "_clp_step_tracker", None) is not None:
        return
    optimizer._clp_step_tracker = _StepTracker(optimizer)


def optimizer_step_count(optimizer):
    """optimizer 实际执行 step 的次数（自带计数器优先，其次 torch 内部计数）。"""
    tracker = getattr(optimizer, "_clp_step_tracker", None)
    if tracker is not None:
        return tracker.steps()
    return getattr(optimizer, "_step_count", None)


def did_optimizer_step(optimizer, before) -> bool:
    """判断本轮是否发生了真实的参数更新。

    返回 False 时不应推进学习率调度：既避免 torch 的
    "lr_scheduler.step() before optimizer.step()" 告警，也保证调度步数与
    真实参数更新次数一致（AMP 下 GradScaler 可能因梯度溢出跳过更新）。
    无法精确计数时，退化为"至少成功更新过一次"（optimizer._opt_called），
    这已足以保证首次调度推进一定发生在一次真实更新之后。
    """
    after = optimizer_step_count(optimizer)
    if before is None or after is None:
        return bool(getattr(optimizer, "_opt_called", False))
    return after > before


# ---------------------------------------------------------------------------
# 路径函数
# ---------------------------------------------------------------------------
def get_pipeline_log_path(cfg: dict) -> str:
    """管线级日志（save_dir/preprocess.log）。"""
    return os.path.join(cfg["save_dir"], "preprocess.log")


def get_upstream_ckpt_path(cfg: dict, seed=None, best: bool = True) -> str:
    """上游 checkpoint 路径。"""
    if seed is None:
        seed = cfg["random_seed"]
    tag = "best" if best else "last"
    return os.path.join(cfg["model_dir"],
                        f"upstream_{cfg['upstream_method']}_seed{seed}_{tag}.pt")


def get_downstream_ckpt_path(cfg: dict, seed=None, best: bool = True) -> str:
    """下游 checkpoint 路径。"""
    if seed is None:
        seed = cfg["random_seed"]
    tag = "best" if best else "last"
    return os.path.join(cfg["model_dir"],
                        f"downstream_{cfg['downstream_model']}_seed{seed}_{tag}.pt")


def get_upstream_state_path(cfg: dict) -> str:
    """上游状态 JSON 路径。"""
    return os.path.join(cfg["state_dir"],
                        f"upstream_state_{cfg['upstream_method']}.json")


def get_downstream_state_path(cfg: dict) -> str:
    """下游状态 JSON 路径（同时缓存 HPO/CV 结果）。"""
    return os.path.join(cfg["state_dir"],
                        f"downstream_state_{cfg['downstream_model']}.json")


def get_evaluation_results_path(cfg: dict) -> str:
    """统一评估结果表 evaluation_results.csv 路径。"""
    return os.path.join(cfg["save_dir"], "evaluation_results.csv")


def get_hpo_results_path(cfg: dict) -> str:
    """HPO 结果 hpo_results_{model}.csv 路径。"""
    return os.path.join(cfg["save_dir"], f"hpo_results_{cfg['downstream_model']}.csv")


def get_report_table_path(cfg: dict) -> str:
    """论文报告表 report_table.csv 路径。"""
    return os.path.join(cfg["save_dir"], "report_table.csv")


def get_hpo_progress_path(cfg: dict) -> str:
    """HPO 断点续跑进度文件路径（逐候选落盘，便于中断后续跑）。"""
    return os.path.join(cfg["state_dir"], f"hpo_progress_{cfg['downstream_model']}.json")


def resume_enabled(cfg: dict) -> bool:
    """是否允许断点续跑：默认开启；--no_resume 或 --force 时关闭。"""
    return bool(cfg.get("resume", True)) and not bool(cfg.get("force", False))


def get_rng_state() -> dict:
    """收集 python / numpy / torch / cuda 的随机状态（断点续训用）。

    随 checkpoint 一起保存，续训时恢复，可让"中断后继续"的训练轨迹与
    "不中断"完全一致（数据打乱顺序、增强采样等随机性都不受影响）。
    """
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": (torch.cuda.get_rng_state_all()
                 if torch.cuda.is_available() else None),
    }


def set_rng_state(state: dict) -> None:
    """恢复 get_rng_state() 保存的随机状态（字段缺失时跳过）。"""
    if not state:
        return
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])
    if state.get("torch") is not None:
        torch.set_rng_state(state["torch"])
    if state.get("cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


# ---------------------------------------------------------------------------
# 日志与评估结果 IO（供各阶段共用，避免逐模块重复实现导致列错位）
# ---------------------------------------------------------------------------
def ensure_stage_log(cfg: dict, name: str) -> str:
    """为根 logger 追加 logs_dir/{name}.log 文件处理器。

    重复调用（同一路径）不会重复添加处理器。
    输入:
        cfg: 配置字典；name: 日志名（如 upstream_train_void）。
    返回:
        str: 日志文件路径。
    """
    os.makedirs(cfg["logs_dir"], exist_ok=True)
    path = os.path.abspath(os.path.join(cfg["logs_dir"], f"{name}.log"))
    root = logging.getLogger()
    for h in root.handlers:
        if isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == path:
            return path
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                           "%Y-%m-%d %H:%M:%S"))
    root.addHandler(handler)
    return path


def append_evaluation_result(cfg: dict, row: dict) -> None:
    """把一行评估结果按统一列顺序追加到 evaluation_results.csv。

    输入:
        cfg: 配置字典；row: 结果字段字典（缺失列自动补空）。
    """
    path = get_evaluation_results_path(cfg)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    full = {c: row.get(c, "") for c in EVAL_COLUMNS}
    df = pd.DataFrame([full], columns=EVAL_COLUMNS)
    if os.path.exists(path):
        try:
            existing_cols = list(pd.read_csv(path, nrows=0).columns)
        except Exception:  # noqa: BLE001  空文件/损坏时退回标准列
            existing_cols = list(EVAL_COLUMNS)
        df = df.reindex(columns=existing_cols, fill_value="")
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)


def read_json(path: str, default: dict = None) -> dict:
    """读取 JSON，文件不存在或损坏时返回 default。"""
    if not os.path.exists(path):
        return {} if default is None else default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {} if default is None else default


def write_json(path: str, obj) -> None:
    """原子化程度较低的 JSON 写入（目录不存在时自动创建）。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def sha1_hash(obj) -> str:
    """对任意可 json 序列化对象求 SHA1（用于 config/data 哈希）。"""
    raw = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def file_stat_hash(path: str) -> str:
    """文件 (size, mtime) 哈希；用于判断依赖文件是否变化。"""
    if not os.path.exists(path):
        return "missing:" + path
    return sha1_hash({"size": os.path.getsize(path),
                      "mtime": int(os.path.getmtime(path))})


# ---------------------------------------------------------------------------
# 指纹分组：阶段相关参数分组，用于状态自动识别
# ---------------------------------------------------------------------------
_SHARED_ARCH_KEYS = ("d_model", "n_heads", "n_layers", "dropout", "seq_length")


def get_fingerprint_groups(cfg: dict) -> dict:
    """返回各阶段指纹相关的参数分组字典。

    分组规则：
    - load：输入路径/文件名/时间范围（决定 01/02 拆表结果）；
    - data：预处理相关（窗口、词表阈值、划分比例、pt_dtype 等）；
    - upstream_train：up_* 训练参数 + 共享结构参数 + 上游损失参数 + 种子；
    - downstream_train：down_* 训练参数 + 共享结构参数 + HPO 搜索空间 +
      最终重训设置 + 上游依赖 + 种子。
    输入:
        cfg: 配置字典。
    返回:
        dict: 分组字典。
    """
    shared = {k: cfg[k] for k in _SHARED_ARCH_KEYS}

    fp = {}
    fp["load"] = {
        "data_dir": cfg["data_dir"],
        "dynamic_file": cfg["dynamic_file"],
        "label_file": cfg["label_file"],
        "static_file": cfg["static_file"],
        "drug_file": cfg["drug_file"],
        "upstream_start": cfg["upstream_start"],
        "upstream_end": cfg["upstream_end"],
        "downstream_start": cfg["downstream_start"],
        "downstream_end": cfg["downstream_end"],
    }
    fp["data"] = {
        "upstream_start": cfg["upstream_start"],
        "upstream_end": cfg["upstream_end"],
        "downstream_start": cfg["downstream_start"],
        "downstream_end": cfg["downstream_end"],
        "data_cutoff": cfg["data_cutoff"],
        "prediction_window_days": cfg["prediction_window_days"],
        "rare_threshold": cfg["rare_threshold"],
        "test_ratio": cfg["test_ratio"],
        "upstream_val_ratio": cfg["upstream_val_ratio"],
        "random_seed": cfg["random_seed"],
        "label_agg": cfg["label_agg"],
        "seq_length": cfg["seq_length"],
        "pt_dtype": cfg["pt_dtype"],
    }
    fp["upstream_train"] = dict(shared)
    fp["upstream_train"].update({
        "upstream_method": cfg["upstream_method"],
        "up_loss_weighting": cfg["up_loss_weighting"],
        "up_token_loss_weighting": cfg["up_token_loss_weighting"],
        "up_select_metric": cfg["up_select_metric"],
        "up_mask_ratio": cfg["up_mask_ratio"],
        "up_tau": cfg["up_tau"],
        "up_log_sigma_clip": cfg["up_log_sigma_clip"],
        "up_void_ratio": cfg["up_void_ratio"],
        "up_amcpc_steps": cfg["up_amcpc_steps"],
        "lambda_mask": cfg["lambda_mask"],
        "lambda_void": cfg["lambda_void"],
        "lambda_amcpc": cfg["lambda_amcpc"],
        "lambda_ts2vec": cfg["lambda_ts2vec"],
        "lambda_tstcc": cfg["lambda_tstcc"],
        "up_lr": cfg["up_lr"],
        "up_weight_decay": cfg["up_weight_decay"],
        "up_batch_size": cfg["up_batch_size"],
        "up_epochs": cfg["up_epochs"],
        "up_patience": cfg["up_patience"],
        "up_grad_clip": cfg["up_grad_clip"],
        "up_scheduler": cfg["up_scheduler"],
        "up_warmup_epochs": cfg["up_warmup_epochs"],
        "up_grad_accum_steps": cfg["up_grad_accum_steps"],
        "up_amp": cfg["up_amp"],
        "up_amp_init_scale": cfg["up_amp_init_scale"],
        "seed_list": cfg["seed_list"],
    })
    fp["downstream_train"] = dict(shared)
    fp["downstream_train"].update({
        "upstream_method": cfg["upstream_method"],
        "downstream_model": cfg["downstream_model"],
        "downstream_arch": cfg["downstream_arch"],
        "downstream_init": cfg["downstream_init"],
        "freeze_upstream": cfg["freeze_upstream"],
        "down_lr": cfg["down_lr"],
        "down_weight_decay": cfg["down_weight_decay"],
        "down_batch_size": cfg["down_batch_size"],
        "down_epochs": cfg["down_epochs"],
        "down_patience": cfg["down_patience"],
        "down_grad_clip": cfg["down_grad_clip"],
        "down_scheduler": cfg["down_scheduler"],
        "down_warmup_epochs": cfg["down_warmup_epochs"],
        "down_grad_accum_steps": cfg["down_grad_accum_steps"],
        "down_amp": cfg["down_amp"],
        "down_amp_init_scale": cfg["down_amp_init_scale"],
        "down_dropout": cfg["down_dropout"],
        "down_loss": cfg["down_loss"],
        "down_hpo_max_epochs": cfg["down_hpo_max_epochs"],
        "hpo_strategy": cfg["hpo_strategy"],
        "hpo_trials": cfg["hpo_trials"],
        "no_hpo": cfg["no_hpo"],
        "hpo_fold_ratio": cfg["hpo_fold_ratio"],
        "cv_folds": cfg["cv_folds"],
        "hpo_lr_grid": cfg["hpo_lr_grid"],
        "hpo_dropout_grid": cfg["hpo_dropout_grid"],
        "hpo_wd_grid": cfg["hpo_wd_grid"],
        "hpo_bs_grid": cfg["hpo_bs_grid"],
        "hpo_primary_metric": cfg["hpo_primary_metric"],
        "final_epochs_mode": cfg["final_epochs_mode"],
        "final_fixed_epochs": cfg["final_fixed_epochs"],
        "final_watch_ratio": cfg["final_watch_ratio"],
        "seed_list": cfg["seed_list"],
    })
    return fp


if __name__ == "__main__":
    print(json.dumps(get_config(), indent=2, ensure_ascii=False, default=str))
