# -*- coding: utf-8 -*-
"""
test_upstream.py
================

上游模型评估：加载 upstream_{method}_seed{seed}_best.pt，在 06_upstream_val.pt
上评估（val_loss / token_acc / value_mae），指标写入日志与 evaluation_results.csv。

复用 train_upstream.validate，保证训练与评估的指标口径完全一致。

依赖: os, logging, numpy, torch, param_setting, model, train_upstream
"""

import os
import logging
import datetime

import numpy as np
import torch

import param_setting
import model as M
import train_upstream as tu

logger = logging.getLogger(__name__)


def load_model(cfg, seed=None):
    """加载上游 best checkpoint，返回 (model, val_dict, ckpt_path)。"""
    pre = M.load_preprocess_params(cfg)
    vocab_size = max(pre["token_vocab"].values()) + 1
    val_path = os.path.join(cfg["save_dir"], "06_upstream_val.pt")
    val_dict = tu.load_tensor_dict(val_path)
    d_static = val_dict["S"].shape[1] if "S" in val_dict else 0
    model = M.build_upstream_model(cfg, vocab_size, d_static, pre["token_freq"])
    ckpt_path = param_setting.get_upstream_ckpt_path(cfg, seed, best=True)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"上游 checkpoint 不存在: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt.get("model_state_dict", ckpt),
                                                strict=False)
    if missing or unexpected:
        logger.warning("[STEP TEST_UP.1] 权重加载 missing=%d unexpected=%d",
                       len(missing), len(unexpected))
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    return model.to(device), val_dict, ckpt_path


def evaluate_upstream(model, loader, device, cfg):
    """在验证集上评估上游，返回 val_loss/token_acc/value_mae 等。"""
    return tu.validate(model, loader, device, cfg)


def append_evaluation_result(cfg, stage, method, dataset, metrics, seed="mean"):
    """追加一行到 evaluation_results.csv。"""
    row = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "stage": stage, "upstream_method": method, "downstream_model": "",
        "downstream_arch": "", "seed": seed, "dataset": dataset,
        "loss": metrics.get("val_loss", metrics.get("loss", np.nan)),
        "token_acc": metrics.get("token_acc", np.nan),
        "value_mae": metrics.get("value_mae", np.nan),
        "best_hparams": "{}", "notes": "upstream_val",
    }
    param_setting.append_evaluation_result(cfg, row)


def main(cfg: dict) -> None:
    """上游评估主入口（对 cfg["seed_list"] 全部种子逐一评估）。"""
    method = cfg["upstream_method"]
    param_setting.ensure_stage_log(cfg, f"upstream_test_{method}")
    results = []
    for seed in cfg["seed_list"]:
        print(f"[STEP TEST_UP.1] upstream_method={method} seed={seed} "
              f"加载上游最佳模型...")
        model, val_dict, ckpt_path = load_model(cfg, seed)
        device = next(model.parameters()).device
        loader = tu.build_dataloader(val_dict, cfg, shuffle=False)
        m = evaluate_upstream(model, loader, device, cfg)
        print(f"[STEP TEST_UP.2] 评估上游验证集... loss={m['val_loss']:.4f} "
              f"token_acc={m.get('token_acc', float('nan')):.4f} "
              f"value_mae={m.get('value_mae', float('nan')):.4f}")
        append_evaluation_result(cfg, "upstream_test", method, "upstream_val", m,
                                seed=seed)
        results.append(float(m["val_loss"]))
    if results:
        std = np.std(results) if len(results) > 1 else 0.0
        print(f"[STEP TEST_UP.3] 多种子汇总 mean_loss={np.mean(results):.4f} ± {std:.4f}")


if __name__ == "__main__":
    cfg = param_setting.get_config()
    logging.basicConfig(level=getattr(logging, cfg["log_level"].upper(), logging.INFO),
                        format="%(asctime)s [%(levelname)s] %(message)s")
    main(cfg)
