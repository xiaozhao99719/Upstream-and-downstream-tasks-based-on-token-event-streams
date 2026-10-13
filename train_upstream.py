# -*- coding: utf-8 -*-
"""
train_upstream.py
=================

上游自监督对比学习训练脚本。

流程：
  1. 读取 save/05_upstream_train.pt 与 save/06_upstream_val.pt；
  2. 按 cfg["upstream_method"] 构建 UpstreamModel；
  3. 对本次需要训练的种子（main.py 传入 cfg["_current_seed"] 时只训该种子，
     直接运行本脚本时训练 cfg["seed_list"] 全部种子）独立训练；
  4. 每轮验证（val_loss / token_acc / value_mae），按 up_select_metric 早停；
  5. 支持断点续训（last.pt，config_hash/data_hash/方法/种子匹配时才续训）；
  6. 指标追加到 save/evaluation_results.csv，历史写入 state JSON。

严格不读取 08_downstream_test.pt。
训练参数全部来自 up_*，与下游 down_* 完全独立。

依赖: os, json, logging, datetime, numpy, torch, torch.utils.data
"""

import os
import json
import logging
import datetime

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import param_setting
import model as M

logger = logging.getLogger(__name__)

_TENSOR_KEYS = ("token_ids", "values", "value_masks", "attn_masks",
                "abs_ages", "interval_univ", "interval_same", "S")
_LOSS_TERMS = ("mask_token", "mask_value", "void", "amcpc", "ts2vec", "tstcc")


class _TensorDictDataset(Dataset):
    """把张量字典按样本索引包装为 Dataset。"""

    def __init__(self, tensor_dict: dict):
        self.tensor_dict = tensor_dict
        self.n = len(tensor_dict["patient_id"])

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        out = {k: self.tensor_dict[k][idx] for k in _TENSOR_KEYS
               if k in self.tensor_dict}
        out["patient_id"] = self.tensor_dict["patient_id"][idx]
        return out


def set_seed(cfg: dict, seed=None) -> int:
    """固定随机种子（默认 cfg["random_seed"]）。"""
    if seed is None:
        seed = cfg["random_seed"]
    param_setting.set_random_seed(seed)
    return seed


def load_tensor_dict(path: str, map_location: str = "cpu") -> dict:
    """加载 .pt 张量字典。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"张量文件不存在: {path}")
    return torch.load(path, map_location=map_location, weights_only=False)


def build_dataloader(tensor_dict: dict, cfg: dict, shuffle: bool) -> DataLoader:
    """构建上游 DataLoader（batch size 来自 up_batch_size）。"""
    ds = _TensorDictDataset(tensor_dict)
    return DataLoader(ds, batch_size=cfg["up_batch_size"], shuffle=shuffle,
                      num_workers=cfg["num_workers"], drop_last=False)


def _use_amp(cfg: dict, device: torch.device) -> bool:
    """混合精度仅在 CUDA 上启用（CPU 上 fp16/bf16 无收益且有兼容风险）。"""
    return bool(cfg["up_amp"]) and device.type == "cuda"


def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, cfg,
                    log_sigmas=None):
    """训练一个 epoch（支持梯度累积、梯度裁剪、AMP）。

    返回:
        dict: 各损失分量/监控指标的 epoch 均值。
    """
    model.train()
    accum = max(1, int(cfg["up_grad_accum_steps"]))
    amp = _use_amp(cfg, device)
    totals, n_batches = {}, 0
    skipped = 0
    optimizer.zero_grad(set_to_none=True)
    n_steps = len(loader)
    for step, batch in enumerate(loader):
        batch = M.move_batch_to_device(batch, device)
        with param_setting.autocast_ctx(device, amp):
            out = model(batch)
            loss = out["loss_total"] / accum
        scaler.scale(loss).backward()
        is_last = (step + 1) == n_steps
        if (step + 1) % accum == 0 or is_last:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["up_grad_clip"])
            steps_before = param_setting.optimizer_step_count(optimizer)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            # 官方推荐顺序：optimizer.step() 之后 scheduler.step()。
            # GradScaler 溢出会跳过更新，此时不推进 LR（否则调度与真实更新次数错位）。
            if param_setting.did_optimizer_step(optimizer, steps_before):
                if scheduler is not None and cfg["up_scheduler"] != "plateau":
                    scheduler.step()
            else:
                skipped += 1
        for k, v in out.items():
            if torch.is_tensor(v):
                totals[k] = totals.get(k, 0.0) + float(v.detach())
        n_batches += 1
    if skipped:
        logger.warning("[STEP TRAIN_UP] 本 epoch 有 %d 次更新被 AMP 跳过（梯度溢出），"
                       "学习率未推进；若频繁出现建议调小初始 scale 或 --up_no_amp", skipped)
    return {k: v / max(1, n_batches) for k, v in totals.items()}


def validate(model, loader, device, cfg):
    """验证：返回 val_loss 及各损失分量/监控指标（按样本数加权平均）。

    评估期固定随机性（mask/增强采样可复现），使 val 曲线跨轮可比。
    """
    model.eval()
    amp = _use_amp(cfg, device)
    rng_state = torch.get_rng_state()
    torch.manual_seed(int(cfg["random_seed"]))
    gen = torch.Generator()
    gen.manual_seed(int(cfg["random_seed"]))
    sums, n = {}, 0
    with torch.no_grad():
        for batch in loader:
            batch = M.move_batch_to_device(batch, device)
            with param_setting.autocast_ctx(device, amp):
                out = model(batch, generator=gen)
            B = batch["token_ids"].size(0)
            for k, v in out.items():
                if torch.is_tensor(v):
                    sums[k] = sums.get(k, 0.0) + float(v.detach()) * B
            n += B
    torch.set_rng_state(rng_state)
    if n == 0:
        return {"val_loss": float("nan")}
    ret = {k: v / n for k, v in sums.items()}
    ret["val_loss"] = ret.pop("loss_total", float("nan"))
    return ret


def _select_metric(va: dict, cfg: dict) -> float:
    """按 up_select_metric 取出用于早停/最优模型选择的标量。"""
    if cfg["up_select_metric"] == "val_loss":
        return float(va.get("val_loss", float("inf")))
    return float(va.get("mask_token", 0.0)) + float(va.get("mask_value", 0.0))


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch,
                    best_metric, patience_counter, cfg, seed, log_sigmas,
                    phase="train", fold=0, best_hparams=None, cv_fold_indices=None,
                    history=None, config_hash=None, data_hash=None, last_val=None):
    """保存 checkpoint（字段遵循断点续训规范）。"""
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
        "cv_fold_indices": cv_fold_indices,
        "history": history,
        "last_val": last_val,
        "log_sigmas": ({k: v.detach().cpu() for k, v in log_sigmas.items()}
                       if log_sigmas else None),
        "rng_state": param_setting.get_rng_state(),
        "cfg": cfg,
    }
    torch.save(ckpt, path)


def load_checkpoint(path, map_location="cpu"):
    """加载 checkpoint；不存在或损坏时返回 None。"""
    if not os.path.exists(path):
        return None
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[STEP TRAIN_UP] 断点损坏，忽略: %s (%s)", path, exc)
        return None


def _config_hash(cfg: dict) -> str:
    """上游训练配置哈希（用于断点续训匹配）。"""
    return param_setting.sha1_hash(param_setting.get_fingerprint_groups(cfg)["upstream_train"])


def _data_hash(cfg: dict) -> str:
    """上游数据文件哈希（size+mtime）。"""
    return param_setting.sha1_hash({
        "train": param_setting.file_stat_hash(
            os.path.join(cfg["save_dir"], "05_upstream_train.pt")),
        "val": param_setting.file_stat_hash(
            os.path.join(cfg["save_dir"], "06_upstream_val.pt")),
    })


def append_evaluation_result(cfg, stage, method, model_name, dataset, metrics,
                             seed="mean", extra=None):
    """追加一行到 evaluation_results.csv（统一列由 param_setting 维护）。"""
    row = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "stage": stage, "upstream_method": method, "downstream_model": model_name,
        "downstream_arch": "", "seed": seed, "dataset": dataset,
        "loss": metrics.get("loss", np.nan),
        "token_acc": metrics.get("token_acc", np.nan),
        "value_mae": metrics.get("value_mae", np.nan),
        "best_hparams": json.dumps(metrics.get("best_hparams", {}), default=str),
        "notes": (extra or ""),
    }
    param_setting.append_evaluation_result(cfg, row)


def write_history_to_state(state_path: str, seed: int, history: list,
                           best_metric: float, select_metric_name: str) -> dict:
    """把某个种子的训练历史合并写入 state JSON，并返回多种子汇总。

    按 seed 分键保存，避免多种子多次调用时覆盖彼此的历史。
    """
    state = param_setting.read_json(state_path)
    state.setdefault("history", {})[f"seed{seed}"] = history
    state.setdefault("seeds", {})[str(seed)] = {"best_metric": float(best_metric),
                                                "select_metric": select_metric_name}
    vals = [float(v["best_metric"]) for v in state["seeds"].values()]
    summary = {
        "select_metric": select_metric_name,
        "n_seeds": len(vals),
        "mean_best_metric": float(np.mean(vals)) if vals else float("nan"),
        "std_best_metric": float(np.std(vals)) if len(vals) > 1 else 0.0,
    }
    state["multi_seed_summary"] = summary
    state["updated_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    param_setting.write_json(state_path, state)
    return summary


def _build_optimizer_scheduler(model, cfg, steps_per_epoch):
    """构建 AdamW + 学习率调度（cosine 含 warmup；plateau 按轮更新）。"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["up_lr"],
                                  weight_decay=cfg["up_weight_decay"])
    total = max(1, int(cfg["up_epochs"]) * max(1, steps_per_epoch))
    warm = max(0, int(cfg["up_warmup_epochs"]) * max(1, steps_per_epoch))
    scheduler = None
    if cfg["up_scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, param_setting.cosine_with_warmup_lambda(total, warm))
    elif cfg["up_scheduler"] == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5,
            patience=max(1, cfg["up_patience"] // 2), min_lr=1e-6)
    # 在调度器构造完成后再挂 step 计数器（调度器构造时会包装 optimizer.step）
    param_setting.attach_step_tracker(optimizer)
    return optimizer, scheduler


def _train_one_seed(cfg, train_dict, val_dict, vocab_size, d_static, token_freq, seed):
    """训练单个种子的上游模型，返回 (best_metric, history, last_val)。"""
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    set_seed(cfg, seed)
    model = M.build_upstream_model(cfg, vocab_size, d_static, token_freq).to(device)
    print(f"[STEP TRAIN_UP.2] 构建模型... 参数量={M.count_parameters(model)} "
          f"激活损失={model.loss_keys} 损失权重={cfg['up_loss_weighting']} seed={seed}")

    train_loader = build_dataloader(train_dict, cfg, shuffle=True)
    val_loader = build_dataloader(val_dict, cfg, shuffle=False)
    optimizer, scheduler = _build_optimizer_scheduler(model, cfg, max(1, len(train_loader)))
    scaler = param_setting.make_grad_scaler(device, _use_amp(cfg, device))

    best_path = param_setting.get_upstream_ckpt_path(cfg, seed, best=True)
    last_path = param_setting.get_upstream_ckpt_path(cfg, seed, best=False)
    cfg_hash, data_hash = _config_hash(cfg), _data_hash(cfg)

    start_epoch, best_metric, patience_counter, history = 1, float("inf"), 0, []
    last_val = {}
    if param_setting.resume_enabled(cfg):
        ckpt = load_checkpoint(last_path)
    else:
        ckpt = None
        if os.path.exists(last_path):
            print("[STEP TRAIN_UP.1] 断点续训已关闭（--no_resume/--force），从第 1 轮重新训练")
    if (ckpt and ckpt.get("upstream_method") == cfg["upstream_method"]
            and int(ckpt.get("seed", -1)) == int(seed)
            and ckpt.get("config_hash") == cfg_hash
            and ckpt.get("data_hash") == data_hash):
        model.load_state_dict(ckpt["model_state_dict"])
        if ckpt.get("optimizer_state_dict"):
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler is not None and ckpt.get("scheduler_state_dict"):
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if ckpt.get("scaler_state_dict"):
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_metric = float(ckpt.get("best_metric", float("inf")))
        patience_counter = int(ckpt.get("patience_counter", 0))
        history = list(ckpt.get("history") or [])
        last_val = dict(ckpt.get("last_val") or {})
        # 恢复随机状态，使续训后的训练轨迹与不中断时一致
        param_setting.set_rng_state(ckpt.get("rng_state"))
        if start_epoch > int(cfg["up_epochs"]):
            print(f"[STEP TRAIN_UP.1] 断点续训：last.pt 显示 seed={seed} 已完成 "
                  f"{start_epoch - 1}/{cfg['up_epochs']} 轮，跳过训练直接汇总")
        else:
            print(f"[STEP TRAIN_UP.1] 断点续训：检测到 seed={seed} 的 last.pt，"
                  f"已完成 {start_epoch - 1}/{cfg['up_epochs']} 轮，从第 {start_epoch} 轮继续"
                  f"（best_metric={best_metric:.4f}）")
    elif ckpt is not None:
        print("[STEP TRAIN_UP.1] 断点与当前配置/数据/方法/种子不匹配，忽略断点，从第 1 轮训练")

    for epoch in range(start_epoch, int(cfg["up_epochs"]) + 1):
        steps_before = param_setting.optimizer_step_count(optimizer)
        tr = train_one_epoch(model, train_loader, optimizer, scheduler, scaler,
                             device, cfg)
        va = validate(model, val_loader, device, cfg)
        last_val = va
        cur_metric = _select_metric(va, cfg)
        cur_lr = optimizer.param_groups[0]["lr"]
        terms = " ".join(f"{k}={tr[k]:.4f}" for k in _LOSS_TERMS if k in tr)
        print(f"[STEP TRAIN_UP.3] Epoch {epoch}/{cfg['up_epochs']} "
              f"total={tr.get('loss_total', 0):.4f} {terms} "
              f"val_loss={va.get('val_loss', float('nan')):.4f} "
              f"select={cur_metric:.4f} token_acc={va.get('token_acc', 0):.4f} "
              f"value_mae={va.get('value_mae', 0):.4f} lr={cur_lr:.2e}")
        history.append({"epoch": epoch,
                        "train": {k: float(v) for k, v in tr.items()},
                        "val": {k: float(v) for k, v in va.items()},
                        "lr": cur_lr, "select_metric": cur_metric})

        if cur_metric < best_metric:
            best_metric = cur_metric
            patience_counter = 0
            save_checkpoint(best_path, model, optimizer, scheduler, scaler, epoch,
                            best_metric, patience_counter, cfg, seed,
                            model.log_sigmas, phase="train", history=history,
                            config_hash=cfg_hash, data_hash=data_hash,
                            last_val={k: float(v) for k, v in va.items()})
        else:
            patience_counter += 1
        save_checkpoint(last_path, model, optimizer, scheduler, scaler, epoch,
                        best_metric, patience_counter, cfg, seed,
                        model.log_sigmas, phase="train", history=history,
                        config_hash=cfg_hash, data_hash=data_hash,
                        last_val={k: float(v) for k, v in va.items()})
        if (cfg["up_scheduler"] == "plateau"
                and param_setting.did_optimizer_step(optimizer, steps_before)):
            scheduler.step(cur_metric)
        if patience_counter >= int(cfg["up_patience"]):
            logger.info("[STEP TRAIN_UP.3] 早停于 epoch %d (seed=%d)", epoch, seed)
            break

    print(f"[STEP TRAIN_UP.4] seed={seed} 完成，"
          f"best_{cfg['up_select_metric']}={best_metric:.4f}")
    return best_metric, history, last_val


def main(cfg: dict) -> None:
    """上游训练主入口。"""
    up = cfg["upstream_method"]
    seeds = param_setting.resolve_seeds(cfg)
    param_setting.ensure_stage_log(cfg, f"upstream_train_{up}")
    print(f"[STEP TRAIN_UP.1] upstream_method={up} seeds={seeds} 加载上游张量...")
    print(f"[STEP TRAIN_UP.1] 断点续跑："
          f"{'开启' if param_setting.resume_enabled(cfg) else '关闭'}"
          f"（自动识别 last.pt：配置/数据/方法/种子一致则从上次 epoch 继续）")
    train_path = os.path.join(cfg["save_dir"], "05_upstream_train.pt")
    val_path = os.path.join(cfg["save_dir"], "06_upstream_val.pt")
    train_dict = load_tensor_dict(train_path)
    val_dict = load_tensor_dict(val_path)
    print(f"[STEP TRAIN_UP.1] train={len(train_dict['patient_id'])}, "
          f"val={len(val_dict['patient_id'])}")

    pre = M.load_preprocess_params(cfg)
    vocab_size = max(pre["token_vocab"].values()) + 1
    d_static = train_dict["S"].shape[1] if "S" in train_dict else 0
    token_freq = pre["token_freq"]

    state_path = param_setting.get_upstream_state_path(cfg)
    best_metrics = []
    for seed in seeds:
        bm, history, last_val = _train_one_seed(
            cfg, train_dict, val_dict, vocab_size, d_static, token_freq, seed)
        best_metrics.append(bm)
        write_history_to_state(state_path, seed, history, bm, cfg["up_select_metric"])
        append_evaluation_result(
            cfg, "upstream_train", up, "", "upstream_val",
            {"loss": float(last_val.get("val_loss", np.nan)),
             "token_acc": float(last_val.get("token_acc", np.nan)),
             "value_mae": float(last_val.get("value_mae", np.nan))},
            seed=seed, extra="seed_run")

    mean_m = float(np.mean(best_metrics))
    std_m = float(np.std(best_metrics)) if len(best_metrics) > 1 else 0.0
    print(f"[STEP TRAIN_UP.5] 多种子汇总... mean_{cfg['up_select_metric']}="
          f"{mean_m:.4f} ± {std_m:.4f}")
    append_evaluation_result(cfg, "upstream_train", up, "", "upstream_val",
                             {"loss": mean_m}, seed="mean", extra="multi_seed_mean")


if __name__ == "__main__":
    cfg = param_setting.get_config()
    logging.basicConfig(level=getattr(logging, cfg["log_level"].upper(), logging.INFO),
                        format="%(asctime)s [%(levelname)s] %(message)s")
    main(cfg)
