# -*- coding: utf-8 -*-
"""
test_downstream.py
==================

下游模型在 08_downstream_test.pt 上的一次性评估。

流程：
  1. 加载 downstream_{model}_seed{seed}_best.pt（含 best_hparams，用于对齐 batch/正则）；
  2. 在 08 上评估一次（严格只此一处读 08）；
  3. 患者级 cluster Bootstrap 得到 95% CI；
  4. 多种子取平均；
  5. 写 save/test_audit.json（审计测试集使用次数）；
  6. 可选 --eval_new_patients_subset 评估新患者子集。

依赖: os, logging, datetime, numpy, torch, scipy, sklearn,
      param_setting, model, train_downstream
"""

import os
import logging
import datetime

import numpy as np
import torch
from scipy import stats as sp_stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import param_setting
import model as M
import train_downstream as td_train

logger = logging.getLogger(__name__)


def load_model(cfg, seed=None):
    """加载下游 best checkpoint。

    返回:
        (model, test_dict, ckpt_path, best_hparams)
    """
    pre = M.load_preprocess_params(cfg)
    vocab_size = max(pre["token_vocab"].values()) + 1
    test_path = os.path.join(cfg["save_dir"], "08_downstream_test.pt")
    test_dict = td_train.load_tensor_dict(test_path)
    d_static = test_dict["S"].shape[1] if "S" in test_dict else 0
    upstream_ckpt = param_setting.get_upstream_ckpt_path(cfg, best=True)
    ckpt_path = param_setting.get_downstream_ckpt_path(cfg, seed, best=True)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"下游 checkpoint 不存在: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    best_hparams = ckpt.get("best_hparams") or {}
    dropout = float(best_hparams.get("down_dropout", cfg["down_dropout"]))
    model = M.build_downstream_model(cfg, vocab_size, d_static, upstream_ckpt,
                                     dropout=dropout)
    missing, unexpected = model.load_state_dict(ckpt.get("model_state_dict", ckpt),
                                                strict=False)
    if missing or unexpected:
        logger.warning("[STEP TEST_DOWN.1] 权重加载 missing=%d unexpected=%d",
                       len(missing), len(unexpected))
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    return model.to(device), test_dict, ckpt_path, best_hparams


def evaluate_downstream(model, loader, device, cfg):
    """推理并返回 (y_true, y_pred, patient_ids)。"""
    model.eval()
    preds, gts, pids = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = M.move_batch_to_device(batch, device)
            preds.append(M.to_numpy(model(batch).float()))
            gts.append(M.to_numpy(batch["y"].float()))
            pids.extend(list(batch["patient_id"]))
    return np.concatenate(gts), np.concatenate(preds), pids


def evaluate_metrics(y_true, y_pred):
    """点估计指标：MAE/RMSE/R2（sklearn）+ Pearson/Spearman（scipy）。"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    try:
        r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan")
    except Exception:  # noqa: BLE001
        r2 = float("nan")
    pearson = spearman = float("nan")
    if len(y_true) > 2 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        try:
            pearson = float(sp_stats.pearsonr(y_true, y_pred)[0])
        except Exception:  # noqa: BLE001
            pass
        try:
            spearman = float(sp_stats.spearmanr(y_true, y_pred)[0])
        except Exception:  # noqa: BLE001
            pass
    return {"mae": mae, "rmse": rmse, "r2": r2, "pearson": pearson,
            "spearman": spearman}


def patient_level_bootstrap(y_true, y_pred, patient_ids, cfg):
    """患者级 cluster Bootstrap 95% CI。

    - 按 patient_id 分组重采样（同一患者的样本整体进出）；
    - 固定随机种子 cfg["random_seed"] + 1；
    - 每次 bootstrap 前检查 y_true.std()/y_pred.std() < 1e-6，跳过并计数；
    - n_valid < 0.9*B 时打印警告；
    - R2 用 sklearn r2_score 并捕获异常。
    返回:
        (ci, n_valid)，ci 为 {metric: (low, high)}。
    """
    B = int(cfg["bootstrap_n"])
    rng = np.random.RandomState(int(cfg["random_seed"]) + 1)
    pids = np.asarray(patient_ids)
    keys = list(np.unique(pids))
    groups = {p: np.where(pids == p)[0] for p in keys}
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    boot = {k: [] for k in ("mae", "rmse", "r2", "pearson", "spearman")}
    n_valid = 0
    for _ in range(B):
        chosen = rng.choice(keys, size=len(keys), replace=True)
        idx = np.concatenate([groups[p] for p in chosen])
        yt, yp = y_true[idx], y_pred[idx]
        if yt.std() < 1e-6 or yp.std() < 1e-6:
            continue
        n_valid += 1
        m = evaluate_metrics(yt, yp)
        for k in boot:
            boot[k].append(m[k])
    if n_valid < 0.9 * B:
        logger.warning("[STEP TEST_DOWN.3] Bootstrap 有效样本 %d < 0.9*%d，CI 可能不稳",
                       n_valid, B)
    ci = {}
    for k, vals in boot.items():
        arr = np.asarray(vals, dtype=float)
        arr = arr[~np.isnan(arr)]
        ci[k] = ((float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))
                 if len(arr) else (np.nan, np.nan))
    return ci, n_valid


def eval_new_patients_subset(model, cfg, best_hparams=None):
    """可选：评估新患者子集（08 中出现但未出现在 07 trainval 的患者）。"""
    try:
        tv = td_train.load_tensor_dict(
            os.path.join(cfg["save_dir"], "07_downstream_trainval.pt"))
        test = td_train.load_tensor_dict(
            os.path.join(cfg["save_dir"], "08_downstream_test.pt"))
        seen = set(tv["patient_id"])
        mask = np.array([p not in seen for p in test["patient_id"]], dtype=bool)
        if mask.sum() == 0:
            print("[STEP TEST_DOWN.5] 新患者子集为空，跳过")
            return
        # 避开 numpy->torch 桥接（部分 torch/numpy 版本组合不可用）
        mask_t = torch.tensor(mask.tolist(), dtype=torch.bool)
        sub = {}
        for k, v in test.items():
            if torch.is_tensor(v):
                sub[k] = v[mask_t] if v.dim() >= 1 and v.shape[0] == len(mask) else v
            elif isinstance(v, list):
                sub[k] = [v[i] for i in np.where(mask)[0]]
            else:
                sub[k] = v
        bs = int((best_hparams or {}).get("down_batch_size", cfg["down_batch_size"]))
        device = next(model.parameters()).device
        loader = torch.utils.data.DataLoader(td_train._DS(sub), batch_size=bs,
                                             shuffle=False)
        yt, yp, _ = evaluate_downstream(model, loader, device, cfg)
        m = evaluate_metrics(yt, yp)
        print(f"[STEP TEST_DOWN.5] 新患者子集 n={int(mask.sum())} "
              f"MAE={m['mae']:.4f} RMSE={m['rmse']:.4f}")
    except Exception as exc:  # noqa: BLE001
        logger.warning("[STEP TEST_DOWN.5] 新患者子集评估失败: %s", exc)


def append_test_audit(cfg, model_name, metrics):
    """写 test_audit.json；同一模型在同一测试集评估 >1 次时打印警告。"""
    path = os.path.join(cfg["save_dir"], "test_audit.json")
    audit = param_setting.read_json(path)
    key = f"{model_name}@{cfg['upstream_method']}"
    prev = int(audit.get(key, 0))
    if prev >= 1:
        logger.warning("[STEP TEST_DOWN.5] 模型 %s 在测试集上已评估 %d 次（应只评估一次）",
                       key, prev + 1)
    audit[key] = prev + 1
    audit["last_updated"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    audit["last_metrics"] = metrics
    param_setting.write_json(path, audit)


def append_evaluation_result(cfg, stage, method, model_name, metrics, ci, seed, n_valid):
    """追加测试集点估计 + CI 到 evaluation_results.csv。"""
    row = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "stage": stage, "upstream_method": method, "downstream_model": model_name,
        "downstream_arch": cfg.get("downstream_arch", ""), "seed": seed,
        "dataset": "downstream_test",
        "mae": metrics["mae"], "mae_ci_low": ci["mae"][0], "mae_ci_high": ci["mae"][1],
        "rmse": metrics["rmse"], "rmse_ci_low": ci["rmse"][0],
        "rmse_ci_high": ci["rmse"][1],
        "r2": metrics["r2"], "r2_ci_low": ci["r2"][0], "r2_ci_high": ci["r2"][1],
        "pearson": metrics["pearson"], "pearson_ci_low": ci["pearson"][0],
        "pearson_ci_high": ci["pearson"][1],
        "spearman": metrics["spearman"], "spearman_ci_low": ci["spearman"][0],
        "spearman_ci_high": ci["spearman"][1],
        "bootstrap_n": n_valid, "bootstrap_unit": cfg["bootstrap_unit"],
        "best_hparams": "{}", "notes": "downstream_test_once",
    }
    param_setting.append_evaluation_result(cfg, row)


def main(cfg: dict) -> None:
    """下游测试主入口（对 cfg["seed_list"] 全部种子逐一评估，测试集只读一次）。"""
    model_name = cfg["downstream_model"]
    param_setting.ensure_stage_log(cfg, f"downstream_test_{model_name}")
    all_metrics = []
    for seed in cfg["seed_list"]:
        print(f"[STEP TEST_DOWN.1] downstream_model={model_name} seed={seed} "
              f"加载最佳模型...")
        model, test_dict, ckpt_path, best_hparams = load_model(cfg, seed)
        device = next(model.parameters()).device
        bs = int(best_hparams.get("down_batch_size", cfg["down_batch_size"]))
        loader = torch.utils.data.DataLoader(td_train._DS(test_dict), batch_size=bs,
                                             shuffle=False)
        yt, yp, pids = evaluate_downstream(model, loader, device, cfg)
        m = evaluate_metrics(yt, yp)
        print(f"[STEP TEST_DOWN.2] 测试集评估... MAE={m['mae']:.4f} RMSE={m['rmse']:.4f} "
              f"R2={m['r2']:.4f} Pearson={m['pearson']:.4f} Spearman={m['spearman']:.4f}")
        print(f"[STEP TEST_DOWN.3] 患者级聚类 Bootstrap... B={cfg['bootstrap_n']} "
              f"patients={len(set(pids))}")
        ci, n_valid = patient_level_bootstrap(yt, yp, pids, cfg)
        print(f"[STEP TEST_DOWN.3] MAE={m['mae']:.3f} "
              f"[{ci['mae'][0]:.3f}, {ci['mae'][1]:.3f}] RMSE={m['rmse']:.3f} "
              f"R2={m['r2']:.3f} Pearson={m['pearson']:.3f} "
              f"Spearman={m['spearman']:.3f} n_valid={n_valid}")
        append_evaluation_result(cfg, "downstream_test", cfg["upstream_method"],
                                 model_name, m, ci, seed, n_valid)
        all_metrics.append(m)
        if cfg.get("eval_new_patients_subset", False):
            eval_new_patients_subset(model, cfg, best_hparams)
    if all_metrics:
        mean_mae = float(np.mean([x["mae"] for x in all_metrics]))
        std_mae = (float(np.std([x["mae"] for x in all_metrics]))
                   if len(all_metrics) > 1 else 0.0)
        print(f"[STEP TEST_DOWN.4] 多种子汇总... MAE={mean_mae:.4f} ± {std_mae:.4f}")
        append_test_audit(cfg, model_name, {"mean_mae": mean_mae, "std_mae": std_mae})
    print("[STEP TEST_DOWN.5] 测试集审计已写入 test_audit.json")


if __name__ == "__main__":
    cfg = param_setting.get_config()
    logging.basicConfig(level=getattr(logging, cfg["log_level"].upper(), logging.INFO),
                        format="%(asctime)s [%(levelname)s] %(message)s")
    main(cfg)
