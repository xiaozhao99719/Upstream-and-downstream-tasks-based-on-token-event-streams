# -*- coding: utf-8 -*-
"""
train_downstream.py
====================

下游监督回归模型训练脚本（完整七环节）：
  读取 07_downstream_trainval.pt -> expanding window CV（日期分组 + 患者级边界对齐）
  -> HPO 折与评估折分离 -> grid/random HPO -> 评估折无偏重跑
  -> 全量地标重训最终模型（留 final_watch_ratio 仅监控）-> 多种子。

HPO/CV 结果按配置指纹缓存到 state/downstream_state_{model}.json：
main.py 按 "downstream_train:{model}:{seed}" 逐种子调用时，只有第一次真正做 HPO，
其余种子直接复用，避免 HPO 被每种子重复执行。

断点续跑（断电/中断后可继续）：
- HPO 逐候选落盘到 state/hpo_progress_{model}.json，重跑时跳过已完成的候选；
- 最终重训每个 epoch 保存 last.pt，重跑时按 epoch 续训（恢复优化器/调度器/AMP 状态）；
- 关闭方式：--no_resume；--force 会忽略全部断点强制重跑。

严格禁止读取 08_downstream_test.pt；测试集只在 test_downstream.py 评估一次。
训练参数全部来自 down_*，与上游 up_* 完全独立。

依赖: os, json, math, random, itertools, logging, datetime, numpy, pandas, torch,
      sklearn.model_selection.TimeSeriesSplit, sklearn.metrics
"""

import os
import json
import random
import itertools
import logging
import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import param_setting
import model as M

logger = logging.getLogger(__name__)

_TENSOR_KEYS = ("token_ids", "values", "value_masks", "attn_masks",
                "abs_ages", "interval_univ", "interval_same", "S")


class _DS(Dataset):
    """下游 Dataset：按索引取张量 + 标签 y。"""

    def __init__(self, td, indices=None):
        self.td = td
        self.indices = list(range(len(td["patient_id"]))) if indices is None else list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        out = {k: self.td[k][idx] for k in _TENSOR_KEYS if k in self.td}
        out["y"] = self.td["y"][idx]
        out["patient_id"] = self.td["patient_id"][idx]
        return out


def set_seed(cfg, seed=None):
    """固定随机种子（默认 cfg["random_seed"]）。"""
    if seed is None:
        seed = cfg["random_seed"]
    param_setting.set_random_seed(seed)
    return seed


def load_tensor_dict(path, map_location="cpu"):
    """加载 .pt 张量字典。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"张量文件不存在: {path}")
    return torch.load(path, map_location=map_location, weights_only=False)


def build_downstream_dataloader(td, indices, cfg, shuffle):
    """构建下游 DataLoader（默认 batch size 来自 down_batch_size）。"""
    return DataLoader(_DS(td, indices), batch_size=cfg["down_batch_size"],
                      shuffle=shuffle, num_workers=cfg["num_workers"], drop_last=False)


def evaluate_metrics(y_true, y_pred):
    """计算 MAE/RMSE/R2（sklearn，统一口径）。"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
    }


def _use_amp(cfg, device) -> bool:
    """混合精度仅在 CUDA 上启用。"""
    return bool(cfg["down_amp"]) and device.type == "cuda"


def _parse_landmark_date(lid: str):
    """从 landmark_id 'patient_id_YYYY-MM-DD' 解析日期。"""
    try:
        return pd.Timestamp(str(lid).rsplit("_", 1)[-1])
    except Exception:  # noqa: BLE001
        return pd.Timestamp("1900-01-01")


def make_time_series_folds(landmark_ids, patient_ids, cfg):
    """按 landmark 日期升序做 expanding window CV，并做患者级边界对齐。

    以"天"为最小不可分割单元切分：train 的所有日期严格早于 val；
    随后把 val 中出现的患者其全部地标从 train 移除（患者级边界对齐），
    保证同一患者不会同时出现在 train 与 val（避免患者级信息泄漏）。

    返回:
        list[(train_idx, val_idx)]，每折为 numpy 数组索引（原始数组下标）。
    """
    dates = pd.Series([_parse_landmark_date(x) for x in landmark_ids])
    uniq_dates = np.array(sorted(dates.unique()))
    n_groups = len(uniq_dates)
    n_splits = int(cfg["cv_folds"])
    if n_groups <= n_splits:
        raise ValueError(f"唯一日期数({n_groups})不足，无法生成 {n_splits} 折 CV")
    in_dates = lambda ds: dates.isin(ds).to_numpy()  # noqa: E731
    pat = np.asarray([str(p) for p in patient_ids])
    tscv = TimeSeriesSplit(n_splits=n_splits)
    folds = []
    for tr_g, va_g in tscv.split(np.arange(n_groups)):
        tr_idx = np.where(in_dates(uniq_dates[tr_g]))[0]
        va_idx = np.where(in_dates(uniq_dates[va_g]))[0]
        va_pat = set(pat[va_idx].tolist())
        if va_pat:
            keep = np.array([p not in va_pat for p in pat[tr_idx]])
            tr_idx = tr_idx[keep]
        folds.append((tr_idx, va_idx))
    print(f"[STEP TRAIN_DOWN.2] 生成 expanding window CV(按日期组+患者级对齐)... "
          f"folds={len(folds)}")
    for i, (tr, va) in enumerate(folds):
        print(f"  fold {i+1}: train={len(tr)} val={len(va)} "
              f"train_pat={len(set(pat[tr]))} val_pat={len(set(pat[va]))} "
              f"(train≤{dates.iloc[tr].max().date()} / val≥{dates.iloc[va].min().date()})")
    return folds


def check_fold_validity(folds, landmark_ids, patient_ids, cfg=None):
    """校验每折：非空、train 最大日期严格早于 val 最小日期、患者不重叠。"""
    dates = pd.Series([_parse_landmark_date(x) for x in landmark_ids])
    pat = np.asarray([str(p) for p in patient_ids])
    for i, (tr, va) in enumerate(folds):
        if len(tr) == 0 or len(va) == 0:
            raise ValueError(f"fold {i+1} 为空折（患者级对齐后 train={len(tr)}, "
                             f"val={len(va)}）")
        if dates.iloc[tr].max() >= dates.iloc[va].min():
            raise ValueError(
                f"fold {i+1} 时间顺序不合法：train.max("
                f"{dates.iloc[tr].max().date()}) >= val.min({dates.iloc[va].min().date()})")
        overlap = set(pat[tr].tolist()) & set(pat[va].tolist())
        if overlap:
            raise ValueError(f"fold {i+1} 患者级泄漏：{len(overlap)} 名患者同时出现在 "
                             f"train 与 val")
    return True


def split_hpo_eval_folds(folds, cfg):
    """前 hpo_fold_ratio 折做 HPO，其余折做无偏评估。"""
    n_hpo = max(1, int(round(len(folds) * cfg["hpo_fold_ratio"])))
    hpo_folds = folds[:n_hpo]
    eval_folds = folds[n_hpo:]
    if not eval_folds:
        eval_folds = folds[-1:]
        hpo_folds = folds[:-1] if len(folds) > 1 else folds
    print(f"[STEP TRAIN_DOWN.2.0] HPO 折={len(hpo_folds)}，评估折={len(eval_folds)}")
    return hpo_folds, eval_folds


def _build_optimizer_scheduler(model, cfg, epochs, steps_per_epoch):
    """构建 AdamW + 学习率调度（down_* 参数；cosine 含 warmup）。"""
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["down_lr"],
                            weight_decay=cfg["down_weight_decay"])
    total = max(1, int(epochs) * max(1, steps_per_epoch))
    warm = max(0, int(cfg["down_warmup_epochs"]) * max(1, steps_per_epoch))
    sched = None
    if cfg["down_scheduler"] == "cosine":
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, param_setting.cosine_with_warmup_lambda(total, warm))
    elif cfg["down_scheduler"] == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=max(1, cfg["down_patience"] // 2),
            min_lr=1e-6)
    # 在调度器构造完成后再挂 step 计数器（调度器构造时会包装 optimizer.step）
    param_setting.attach_step_tracker(opt)
    return opt, sched


def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, cfg):
    """训练一个 epoch（梯度累积 + 梯度裁剪 + AMP），返回平均训练损失。"""
    model.train()
    accum = max(1, int(cfg["down_grad_accum_steps"]))
    amp = _use_amp(cfg, device)
    losses = []
    skipped = 0
    optimizer.zero_grad(set_to_none=True)
    n_steps = len(loader)
    for step, batch in enumerate(loader):
        batch = M.move_batch_to_device(batch, device)
        with param_setting.autocast_ctx(device, amp):
            pred = model(batch)
            loss = M.downstream_regression_loss(pred, batch["y"], cfg)
        scaler.scale(loss / accum).backward()
        if (step + 1) % accum == 0 or (step + 1) == n_steps:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["down_grad_clip"])
            steps_before = param_setting.optimizer_step_count(optimizer)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if param_setting.did_optimizer_step(optimizer, steps_before):
                if scheduler is not None and cfg["down_scheduler"] != "plateau":
                    scheduler.step()
            else:
                skipped += 1
        losses.append(float(loss.detach()))
    if skipped:
        logger.warning("[STEP TRAIN_DOWN] 本 epoch 有 %d 次更新被 AMP 跳过（梯度溢出），"
                       "学习率未推进；若频繁出现建议调小初始 scale 或 --down_no_amp", skipped)
    return float(np.mean(losses)) if losses else float("nan")


def validate(model, loader, device, cfg):
    """验证：返回 {'loss', 'mae', 'rmse', 'r2'}。"""
    model.eval()
    amp = _use_amp(cfg, device)
    preds, gts = [], []
    with torch.no_grad():
        for batch in loader:
            batch = M.move_batch_to_device(batch, device)
            with param_setting.autocast_ctx(device, amp):
                pred = model(batch)
            preds.append(M.to_numpy(pred.float()))
            gts.append(M.to_numpy(batch["y"].float()))
    if not preds:
        return {"loss": float("nan"), "mae": float("nan"),
                "rmse": float("nan"), "r2": float("nan")}
    m = evaluate_metrics(np.concatenate(gts), np.concatenate(preds))
    m["loss"] = m["mae"]
    return m


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best_metric,
                    patience_counter, cfg, seed, phase="final", fold=0,
                    best_hparams=None, cv_fold_indices=None, history=None,
                    config_hash=None, data_hash=None, log_sigmas=None):
    """保存下游 checkpoint（字段遵循断点续训规范）。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "epoch": epoch, "fold": fold, "phase": phase,
        "best_metric": best_metric, "patience_counter": patience_counter,
        "config_hash": config_hash, "data_hash": data_hash,
        "upstream_method": cfg.get("upstream_method"),
        "downstream_model": cfg.get("downstream_model"),
        "downstream_arch": cfg.get("downstream_arch"),
        "seed": seed, "best_hparams": best_hparams,
        "cv_fold_indices": cv_fold_indices, "history": history,
        "log_sigmas": log_sigmas,
        "rng_state": param_setting.get_rng_state(),
        "cfg": cfg,
    }
    torch.save(ckpt, path)


def load_checkpoint(path, map_location="cpu"):
    """加载下游 checkpoint；不存在或损坏时返回 None。"""
    if not os.path.exists(path):
        return None
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[STEP TRAIN_DOWN] 断点损坏，忽略: %s (%s)", path, exc)
        return None


def _fit_one_fold(cfg, td, train_idx, val_idx, hparams, vocab_size, d_static,
                  upstream_ckpt, seed=42, max_epochs=None):
    """在单个折上训练并返回 {'best_mae', 'best_epoch'}。"""
    set_seed(cfg, seed)
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    dropout = float(hparams.get("down_dropout", cfg["down_dropout"]))
    model = M.build_downstream_model(cfg, vocab_size, d_static, upstream_ckpt,
                                     dropout=dropout).to(device)
    bs = int(hparams.get("down_batch_size", cfg["down_batch_size"]))
    epochs = int(max_epochs or cfg["down_epochs"])
    tr_loader = DataLoader(_DS(td, train_idx), batch_size=bs, shuffle=True,
                           num_workers=cfg["num_workers"])
    va_loader = DataLoader(_DS(td, val_idx), batch_size=bs, shuffle=False,
                           num_workers=cfg["num_workers"])
    opt, sched = _build_optimizer_scheduler(model, cfg, epochs, max(1, len(tr_loader)))
    scaler = param_setting.make_grad_scaler(device, _use_amp(cfg, device),
                                           init_scale=cfg.get("down_amp_init_scale"))
    best_mae, best_epoch, patience = float("inf"), 0, 0
    for ep in range(1, epochs + 1):
        steps_before = param_setting.optimizer_step_count(opt)
        train_one_epoch(model, tr_loader, opt, sched, scaler, device, cfg)
        va = validate(model, va_loader, device, cfg)
        if (sched is not None and cfg["down_scheduler"] == "plateau"
                and param_setting.did_optimizer_step(opt, steps_before)):
            sched.step(va["mae"])
        if va["mae"] < best_mae - 1e-9:
            best_mae, best_epoch, patience = float(va["mae"]), ep, 0
        else:
            patience += 1
            if patience >= int(cfg["down_patience"]):
                break
    return {"best_mae": float(best_mae), "best_epoch": int(best_epoch)}


def search_hyperparameters(td, hpo_folds, cfg, vocab_size, d_static, upstream_ckpt):
    """HPO：grid/random 搜索，选 val_mae 最小配置。

    逐候选把结果落盘到 state/hpo_progress_{model}.json：
    中断后重跑时，若缓存键（参数分组 + 依赖文件）一致，则跳过已完成的候选，
    避免整段 HPO 重算。

    返回:
        dict: best_hparams（down_lr / down_dropout / down_weight_decay /
              down_batch_size / best_epoch_median / mean_val_mae）。
    """
    print(f"[STEP TRAIN_DOWN.2.1] HPO 开始... strategy={cfg['hpo_strategy']} "
          f"trials={cfg['hpo_trials']}")
    grid = list(itertools.product(cfg["hpo_lr_grid_parsed"], cfg["hpo_dropout_grid_parsed"],
                                 cfg["hpo_wd_grid_parsed"], cfg["hpo_bs_grid_parsed"]))
    if not grid:
        raise ValueError("HPO 搜索空间为空，请检查 hpo_*_grid 参数")
    if cfg["hpo_strategy"] == "random":
        random.Random(cfg["random_seed"]).shuffle(grid)
        grid = grid[:max(1, int(cfg["hpo_trials"]))]
    progress_path = param_setting.get_hpo_progress_path(cfg)
    cache_key = _hpo_cache_key(cfg)
    done_rows = {}
    if param_setting.resume_enabled(cfg):
        progress = param_setting.read_json(progress_path)
        if progress.get("cache_key") == cache_key:
            for row in progress.get("results", []):
                try:
                    done_rows[int(row["config_id"])] = row
                except (KeyError, TypeError, ValueError):
                    continue
    if done_rows:
        print(f"[STEP TRAIN_DOWN.2.1] HPO 断点续跑：复用已完成候选 "
              f"{len(done_rows)}/{len(grid)} 个（{progress_path}）")
    results = []
    for cid, (lr, do, wd, bs) in enumerate(grid):
        if cid in done_rows:
            results.append(done_rows[cid])
            continue
        hparams = {"down_lr": float(lr), "down_dropout": float(do),
                   "down_weight_decay": float(wd), "down_batch_size": int(bs)}
        maes, eps = [], []
        for tr, va in hpo_folds:
            res = _fit_one_fold(cfg, td, tr, va, hparams, vocab_size, d_static,
                                upstream_ckpt, seed=cfg["random_seed"],
                                max_epochs=cfg["down_hpo_max_epochs"])
            maes.append(res["best_mae"])
            eps.append(res["best_epoch"])
        mean_mae = float(np.mean(maes))
        std_mae = float(np.std(maes)) if len(maes) > 1 else 0.0
        results.append({"config_id": cid, "lr": lr, "dropout": do, "weight_decay": wd,
                        "batch_size": bs, "mean_val_mae": mean_mae,
                        "std_val_mae": std_mae,
                        "best_epoch_median": int(np.median(eps))})
        _update_progress(progress_path, cache_key=cache_key, results=results)
        print(f"[STEP TRAIN_DOWN.2.2] 候选{cid} lr={lr} dropout={do} wd={wd} bs={bs} "
              f"mae={mean_mae:.4f}±{std_mae:.4f}")
    df = pd.DataFrame(results).sort_values("config_id").reset_index(drop=True)
    hpo_path = param_setting.get_hpo_results_path(cfg)
    os.makedirs(os.path.dirname(hpo_path), exist_ok=True)
    df.to_csv(hpo_path, index=False)
    best_row = df.sort_values("mean_val_mae").iloc[0].to_dict()
    best_hparams = {
        "down_lr": float(best_row["lr"]), "down_dropout": float(best_row["dropout"]),
        "down_weight_decay": float(best_row["weight_decay"]),
        "down_batch_size": int(best_row["batch_size"]),
        "best_epoch_median": int(best_row["best_epoch_median"]),
        "mean_val_mae": float(best_row["mean_val_mae"]),
    }
    print(f"[STEP TRAIN_DOWN.2.3] 最佳配置 lr={best_hparams['down_lr']} "
          f"dropout={best_hparams['down_dropout']} wd={best_hparams['down_weight_decay']} "
          f"bs={best_hparams['down_batch_size']} "
          f"mean_val_mae={best_hparams['mean_val_mae']:.4f}")
    return best_hparams


def run_cv_with_best_hparams(td, eval_folds, best_hparams, cfg, vocab_size, d_static,
                             upstream_ckpt):
    """用最佳配置在评估折上重跑，得到无偏指标与早停轮数中位数。

    折结果逐折落盘（与 HPO 共用进度文件），中断后只补算缺失的折。
    """
    print("[STEP TRAIN_DOWN.3] 评估折重跑...")
    progress_path = param_setting.get_hpo_progress_path(cfg)
    cv_key = param_setting.sha1_hash({"hpo": _hpo_cache_key(cfg),
                                      "hparams": best_hparams,
                                      "data": _data_hash(cfg)})
    done_folds = {}
    if param_setting.resume_enabled(cfg):
        progress = param_setting.read_json(progress_path)
        if progress.get("cv_cache_key") == cv_key:
            for row in progress.get("cv_results", []):
                try:
                    done_folds[int(row["fold"])] = row
                except (KeyError, TypeError, ValueError):
                    continue
    if done_folds:
        print(f"[STEP TRAIN_DOWN.3] 断点续跑：复用已完成评估折 "
              f"{len(done_folds)}/{len(eval_folds)} 个")
    maes, epochs_list = [], []
    for i, (tr, va) in enumerate(eval_folds):
        fold_no = i + 1
        if fold_no in done_folds:
            row = done_folds[fold_no]
            res = {"best_mae": float(row["best_mae"]),
                   "best_epoch": int(row["best_epoch"])}
        else:
            res = _fit_one_fold(cfg, td, tr, va, best_hparams, vocab_size, d_static,
                                upstream_ckpt, seed=cfg["random_seed"])
            done_folds[fold_no] = {"fold": fold_no, "best_mae": res["best_mae"],
                                   "best_epoch": res["best_epoch"]}
            _update_progress(progress_path, cv_cache_key=cv_key,
                             cv_results=[done_folds[k] for k in sorted(done_folds)])
        maes.append(res["best_mae"])
        epochs_list.append(res["best_epoch"])
        print(f"  fold={fold_no}/{len(eval_folds)} val_mae={res['best_mae']:.4f} "
              f"early_stop_epoch={res['best_epoch']}")
    return {"mean_mae": float(np.mean(maes)),
            "std_mae": float(np.std(maes)) if len(maes) > 1 else 0.0,
            "epochs_median": int(np.median(epochs_list))}


def retrain_final_model(td, best_hparams, epochs, cfg, vocab_size, d_static,
                        upstream_ckpt, seed):
    """全量地标重训最终模型（留 final_watch_ratio 仅监控），返回 watch_mae。"""
    n = len(td["patient_id"])
    n_watch = max(1, int(round(n * cfg["final_watch_ratio"]))) if n > 1 else 0
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    watch_idx = np.sort(perm[:n_watch])
    train_idx = np.sort(perm[n_watch:])
    if len(train_idx) == 0:
        raise ValueError("最终重训训练集为空，请减小 final_watch_ratio")
    print(f"[STEP TRAIN_DOWN.4] 全量重训最终模型... n_landmarks={n} epochs={epochs} "
          f"watch_ratio={cfg['final_watch_ratio']} seed={seed}")
    set_seed(cfg, seed)
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    dropout = float(best_hparams.get("down_dropout", cfg["down_dropout"]))
    model = M.build_downstream_model(cfg, vocab_size, d_static, upstream_ckpt,
                                     dropout=dropout).to(device)
    bs = int(best_hparams.get("down_batch_size", cfg["down_batch_size"]))
    tr_loader = DataLoader(_DS(td, train_idx), batch_size=bs, shuffle=True,
                           num_workers=cfg["num_workers"])
    wa_loader = DataLoader(_DS(td, watch_idx), batch_size=bs, shuffle=False,
                           num_workers=cfg["num_workers"])
    opt, sched = _build_optimizer_scheduler(model, cfg, epochs, max(1, len(tr_loader)))
    scaler = param_setting.make_grad_scaler(device, _use_amp(cfg, device),
                                           init_scale=cfg.get("down_amp_init_scale"))
    best_path = param_setting.get_downstream_ckpt_path(cfg, seed, best=True)
    last_path = param_setting.get_downstream_ckpt_path(cfg, seed, best=False)
    cfg_hash, data_hash = _cfg_hash(cfg), _data_hash(cfg)
    hp_hash = param_setting.sha1_hash(best_hparams)

    start_epoch = 1
    prev_train, prev_watch, worsen = float("inf"), float("inf"), 0
    history, last_watch = [], float("nan")
    ckpt = load_checkpoint(last_path) if param_setting.resume_enabled(cfg) else None
    if (ckpt and ckpt.get("phase") == "final"
            and ckpt.get("downstream_model") == cfg["downstream_model"]
            and int(ckpt.get("seed", -1)) == int(seed)
            and ckpt.get("config_hash") == cfg_hash
            and ckpt.get("data_hash") == data_hash
            and param_setting.sha1_hash(ckpt.get("best_hparams") or {}) == hp_hash):
        model.load_state_dict(ckpt["model_state_dict"])
        if ckpt.get("optimizer_state_dict"):
            opt.load_state_dict(ckpt["optimizer_state_dict"])
        if sched is not None and ckpt.get("scheduler_state_dict"):
            sched.load_state_dict(ckpt["scheduler_state_dict"])
        if ckpt.get("scaler_state_dict"):
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        last_watch = float(ckpt.get("best_metric", float("nan")))
        history = list(ckpt.get("history") or [])
        if history:
            prev_train = float(history[-1].get("train", float("inf")))
            prev_watch = float(history[-1].get("watch_mae", float("inf")))
        # 恢复随机状态，使续训后的训练轨迹与不中断时一致
        param_setting.set_rng_state(ckpt.get("rng_state"))
        if start_epoch > int(epochs):
            print(f"[STEP TRAIN_DOWN.4] 断点续训：seed={seed} 的最终重训已完成 "
                  f"{start_epoch - 1}/{epochs} 轮，跳过训练直接保存结果")
        else:
            print(f"[STEP TRAIN_DOWN.4] 断点续训：检测到 seed={seed} 的 last.pt，"
                  f"已完成 {start_epoch - 1}/{epochs} 轮，从第 {start_epoch} 轮继续")

    for ep in range(start_epoch, int(epochs) + 1):
        steps_before = param_setting.optimizer_step_count(opt)
        tr_loss = train_one_epoch(model, tr_loader, opt, sched, scaler, device, cfg)
        wa = validate(model, wa_loader, device, cfg)
        last_watch = float(wa["mae"])
        if (sched is not None and cfg["down_scheduler"] == "plateau"
                and param_setting.did_optimizer_step(opt, steps_before)):
            sched.step(wa["mae"])
        history.append({"epoch": ep, "train": tr_loss, "watch_mae": last_watch})
        if ep % 10 == 0 or ep == int(epochs):
            print(f"[STEP TRAIN_DOWN.4] Epoch {ep} train={tr_loss:.4f} "
                  f"watch={last_watch:.4f}")
        if tr_loss < prev_train and last_watch > prev_watch:
            worsen += 1
            if worsen >= 3:
                logger.warning("[STEP TRAIN_DOWN.4] train 持续下降而 watch 上升≥3轮"
                               "（可能过拟合）")
                worsen = 0
        else:
            worsen = 0
        prev_train, prev_watch = tr_loss, last_watch
        save_checkpoint(last_path, model, opt, sched, scaler, ep, last_watch, 0, cfg,
                        seed, phase="final", best_hparams=best_hparams,
                        history=history, config_hash=cfg_hash, data_hash=data_hash)
    save_checkpoint(best_path, model, opt, sched, scaler, int(epochs), last_watch, 0,
                    cfg, seed, phase="final", best_hparams=best_hparams,
                    history=history, config_hash=cfg_hash, data_hash=data_hash)
    print(f"[STEP TRAIN_DOWN.4] seed={seed} 最终模型已保存 best/last，"
          f"watch_mae={last_watch:.4f}")
    return last_watch


def append_evaluation_result(cfg, stage, method, model_name, dataset, metrics,
                             seed="mean", extra=""):
    """追加一行到 evaluation_results.csv（统一列由 param_setting 维护）。"""
    row = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "stage": stage, "upstream_method": method, "downstream_model": model_name,
        "downstream_arch": cfg.get("downstream_arch", ""), "seed": seed,
        "dataset": dataset,
        "mae": metrics.get("mae", np.nan), "rmse": metrics.get("rmse", np.nan),
        "r2": metrics.get("r2", np.nan),
        "best_hparams": json.dumps(metrics.get("best_hparams", {}), default=str),
        "notes": extra,
    }
    param_setting.append_evaluation_result(cfg, row)


def _hpo_cache_key(cfg: dict) -> str:
    """HPO/CV 缓存键：下游参数分组 + 数据分组 + 依赖文件状态。"""
    groups = param_setting.get_fingerprint_groups(cfg)
    dep = {
        "trainval": param_setting.file_stat_hash(
            os.path.join(cfg["save_dir"], "07_downstream_trainval.pt")),
        "upstream_ckpt": param_setting.file_stat_hash(
            param_setting.get_upstream_ckpt_path(cfg, best=True)),
    }
    return param_setting.sha1_hash({"down": groups["downstream_train"],
                                   "data": groups["data"], "dep": dep})


def _cfg_hash(cfg: dict) -> str:
    """下游训练配置哈希（用于最终重训断点续跑匹配）。"""
    return param_setting.sha1_hash(
        param_setting.get_fingerprint_groups(cfg)["downstream_train"])


def _data_hash(cfg: dict) -> str:
    """下游数据文件哈希（size+mtime）。"""
    return param_setting.file_stat_hash(
        os.path.join(cfg["save_dir"], "07_downstream_trainval.pt"))


def _update_progress(path: str, **kv) -> None:
    """合并写入断点续跑进度文件（保留其它字段，避免互相覆盖）。"""
    data = param_setting.read_json(path)
    data.update(kv)
    param_setting.write_json(path, data)


def main(cfg: dict) -> None:
    """下游训练主入口（HPO/CV 只做一次并缓存，逐种子重训最终模型）。"""
    model_name = cfg["downstream_model"]
    seeds = param_setting.resolve_seeds(cfg)
    param_setting.ensure_stage_log(cfg, f"downstream_train_{model_name}")
    print(f"[STEP TRAIN_DOWN.1] downstream_model={model_name} seeds={seeds} "
          f"加载 07 张量...")
    print(f"[STEP TRAIN_DOWN.1] 断点续跑："
          f"{'开启' if param_setting.resume_enabled(cfg) else '关闭'}"
          f"（HPO 逐候选落盘、最终重训按 epoch 续训，配置/数据一致时自动跳过已完成部分）")
    td = load_tensor_dict(os.path.join(cfg["save_dir"], "07_downstream_trainval.pt"))
    print(f"[STEP TRAIN_DOWN.1] 地标数={len(td['landmark_id'])}")
    pre = M.load_preprocess_params(cfg)
    vocab_size = max(pre["token_vocab"].values()) + 1
    d_static = td["S"].shape[1] if "S" in td else 0
    upstream_ckpt = param_setting.get_upstream_ckpt_path(cfg, best=True)
    if cfg["downstream_init"] == "upstream" and not os.path.exists(upstream_ckpt):
        raise FileNotFoundError(
            f"downstream_init=upstream 但未找到上游 checkpoint: {upstream_ckpt}；"
            f"请先运行上游训练阶段")

    state_path = param_setting.get_downstream_state_path(cfg)
    state = param_setting.read_json(state_path)
    cache_key = _hpo_cache_key(cfg)
    if (not cfg.get("force", False) and state.get("hpo_cache_key") == cache_key
            and state.get("best_hparams") and state.get("cv_result")):
        best_hparams = state["best_hparams"]
        cv_res = state["cv_result"]
        print("[STEP TRAIN_DOWN.2] 复用已缓存的 HPO/CV 结果（配置与依赖文件未变化）")
    else:
        folds = make_time_series_folds(td["landmark_id"], td["patient_id"], cfg)
        check_fold_validity(folds, td["landmark_id"], td["patient_id"], cfg)
        hpo_folds, eval_folds = split_hpo_eval_folds(folds, cfg)
        if cfg["no_hpo"] or cfg["hpo_strategy"] == "none":
            best_hparams = {"down_lr": cfg["down_lr"],
                            "down_dropout": cfg["down_dropout"],
                            "down_weight_decay": cfg["down_weight_decay"],
                            "down_batch_size": cfg["down_batch_size"],
                            "best_epoch_median": int(cfg["down_hpo_max_epochs"])}
            print("[STEP TRAIN_DOWN.2.1] --no_hpo，使用 down_* 默认超参数")
        else:
            best_hparams = search_hyperparameters(td, hpo_folds, cfg, vocab_size,
                                                  d_static, upstream_ckpt)
        cv_res = run_cv_with_best_hparams(td, eval_folds, best_hparams, cfg,
                                          vocab_size, d_static, upstream_ckpt)
        state["hpo_cache_key"] = cache_key
        state["best_hparams"] = best_hparams
        state["cv_result"] = cv_res
        param_setting.write_json(state_path, state)

    append_evaluation_result(cfg, "downstream_train", cfg["upstream_method"],
                             model_name, "downstream_cv",
                             {"mae": cv_res["mean_mae"], "best_hparams": best_hparams},
                             seed="mean", extra="cv_fold_mean")

    if cfg["final_epochs_mode"] == "cv_median":
        final_epochs = int(best_hparams.get("best_epoch_median", cv_res["epochs_median"]))
    else:
        final_epochs = int(cfg["final_fixed_epochs"])
    final_epochs = max(1, final_epochs)
    watch_maes = []
    for seed in seeds:
        wm = retrain_final_model(td, best_hparams, final_epochs, cfg, vocab_size,
                                 d_static, upstream_ckpt, seed)
        watch_maes.append(wm)
        append_evaluation_result(cfg, "downstream_train", cfg["upstream_method"],
                                 model_name, "downstream_trainval",
                                 {"best_hparams": best_hparams}, seed=seed,
                                 extra="final_model")
        state = param_setting.read_json(state_path)
        state.setdefault("final_fit", {})[str(seed)] = {"watch_mae": float(wm),
                                                        "epochs": final_epochs}
        param_setting.write_json(state_path, state)

    std_w = float(np.std(watch_maes)) if len(watch_maes) > 1 else 0.0
    print(f"[STEP TRAIN_DOWN.5] 多种子汇总... 最终轮数={final_epochs} "
          f"mean_watch_mae={float(np.mean(watch_maes)):.4f} ± {std_w:.4f}")


if __name__ == "__main__":
    cfg = param_setting.get_config()
    logging.basicConfig(level=getattr(logging, cfg["log_level"].upper(), logging.INFO),
                        format="%(asctime)s [%(levelname)s] %(message)s")
    main(cfg)
