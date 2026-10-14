# -*- coding: utf-8 -*-
"""
model.py
========

上游对比学习编码器/对比头 + 下游回归模型 + 损失函数 + 数据增强。

张量字典约定（来自 data_process.tensors_to_pt_and_csv）：
- token_ids:     [B, L] int64   （0=PAD, 1=UNK, 2=RARE, 3=MASK，>=4 为真实 token）
- values:        [B, L] float32  （连续值，已 Min-Max 归一化）
- value_masks:   [B, L] int8    （1=该位置有连续值，0=类别/缺失）
- attn_masks:    [B, L] int8    （1=真实 token，0=PAD）
- abs_ages:      [B, L] float32  （中心化年龄，年）
- interval_univ:[B, L] float32  （log1p 距上次任意就诊天数）
- interval_same: [B, L] float32  （log1p 距上次同类 token 天数）
- S:             [B, d_static] float32（静态 one-hot 特征）

上游方法：void / amcpc / ts2vec / tstcc / all（all 为联合激活）。
下游模型：strats / duett / gru_d / lstm / transformer。

训练参数分层：上游损失权重使用 up_* / lambda_*，下游回归头 Dropout 使用
down_dropout，编码器结构参数（d_model / n_heads / n_layers / dropout）上下游共享。

依赖: os, math, logging, random, numpy, torch
"""

import os
import math
import logging
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

PAD_TOKEN_ID = 0
UNK_TOKEN_ID = 1
RARE_TOKEN_ID = 2
MASK_TOKEN_ID = 3     # 上游 masked event modeling 的占位符（真实 token 从 4 起）


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------
def load_preprocess_params(cfg: dict) -> dict:
    """加载 preprocess_params.pt，提取训练需要的字段。

    输入:
        cfg: 配置字典。
    返回:
        dict: 含 token_vocab / rare_mapping / numeric_minmax / age_center /
              token_freq；token_freq 缺失时返回空 dict 并打印警告。
    """
    path = os.path.join(cfg["save_dir"], "preprocess_params.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到预处理参数: {path}")
    params = torch.load(path, map_location="cpu", weights_only=False)
    out = {
        "token_vocab": params.get("token_vocab", {}),
        "rare_mapping": params.get("rare_mapping", {}),
        "numeric_minmax": params.get("numeric_minmax", {}),
        "age_center": float(params.get("age_center", 0.0)),
        "token_freq": params.get("token_freq", None),
    }
    if not out["token_freq"]:
        logger.warning("[STEP MODEL.0] preprocess_params 中无 token_freq 字段，"
                       "token 损失将不做频率加权")
        out["token_freq"] = {}
    return out


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """统计模型参数量。

    输入:
        model: nn.Module；trainable_only: 只统计可训练参数。
    返回:
        int: 参数个数。
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def handle_nan_values(values: torch.Tensor, value_masks: torch.Tensor):
    """统一 NaN 处理（缺失/极端值不落盘插补，下游消费时执行）。

    value = torch.nan_to_num(value, nan=0.0)
    value_masks = value_masks * (~torch.isnan(values_original))

    输入:
        values: [B, L] 连续值张量；value_masks: [B, L] (int/float) 掩码。
    返回:
        (values_filled, masks_refined)：同 shape。
    """
    values_orig = values
    values_filled = torch.nan_to_num(values, nan=0.0)
    nan_mask = (~torch.isnan(values_orig)).to(value_masks.dtype)
    masks_refined = value_masks.to(values_filled.dtype) * nan_mask
    return values_filled, masks_refined


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    """把一个 batch 张量字典搬到 device，并统一执行 NaN 处理。

    非张量字段（patient_id / landmark_id 等 list 元数据）原样保留。
    输入:
        batch: DataLoader collate 出的 dict；device: 目标设备。
    返回:
        dict: 搬到 device 且处理过 NaN 的 batch。
    """
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    if "values" in out and "value_masks" in out:
        out["values"], out["value_masks"] = handle_nan_values(
            out["values"], out["value_masks"])
    return out


def to_numpy(x):
    """把张量 / 列表安全转换为 numpy 数组。

    torch 与 numpy 2.x 若不匹配，``tensor.numpy()`` 会抛
    ``RuntimeError: Numpy is not available``；这里退化为 ``np.asarray(t.tolist())``，
    保证指标计算在任意 torch/numpy 组合下都能进行。
    """
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        x = x.detach().cpu()
        try:
            return x.numpy()
        except Exception:  # noqa: BLE001
            return np.asarray(x.tolist())
    return np.asarray(x)


# ---------------------------------------------------------------------------
# 时间/数值/静态投影
# ---------------------------------------------------------------------------
class TimeFeatureProjection(nn.Module):
    """三路时间特征（abs_age_years / interval_univ / interval_same）-> d_model。"""

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(3, d_model)

    def forward(self, abs_ages, interval_univ, interval_same):
        x = torch.stack([abs_ages, interval_univ, interval_same], dim=-1)
        return self.proj(x)


class ValueProjection(nn.Module):
    """连续值投影：value_masks 对应的标量 value -> d_model。"""

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(1, d_model)

    def forward(self, values, value_masks):
        v = values.unsqueeze(-1) * value_masks.unsqueeze(-1).to(values.dtype)
        return self.proj(v)


class StaticProjection(nn.Module):
    """静态特征 -> d_model，可广播加到每个 token 位置。"""

    def __init__(self, d_static: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_static, d_model) if d_static > 0 else None

    def forward(self, S):
        if self.proj is None:
            return None
        return self.proj(S)


def _head_dropout(cfg: dict, dropout=None) -> float:
    """下游头 Dropout：显式值 > cfg["down_dropout"]。"""
    if dropout is not None:
        return float(dropout)
    return float(cfg.get("down_dropout", cfg.get("dropout", 0.2)))


# ---------------------------------------------------------------------------
# 上游事件编码器
# ---------------------------------------------------------------------------
class UpstreamEventEncoder(nn.Module):
    """Token Embedding + Value Projection + Time Projection + Static Projection
    + TransformerEncoder。

    输出 hidden_states [B,L,d] 与 pooled [B,d]（attn_mask 加权平均）。
    """

    def __init__(self, vocab_size: int, d_static: int, d_model: int = 128,
                 n_heads: int = 4, n_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_TOKEN_ID)
        self.value_proj = ValueProjection(d_model)
        self.time_proj = TimeFeatureProjection(d_model)
        self.static_proj = StaticProjection(d_static, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu",
            norm_first=True)
        # norm_first=True（Pre-LN，训练更稳）与 nested tensor 快速路径互斥，
        # 显式关闭 nested tensor 以避免 torch 每次构建模型都发一条无用告警。
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers,
                                            enable_nested_tensor=False)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, token_ids, values, value_masks, attn_masks,
                abs_ages, interval_univ, interval_same, S):
        """编码序列。

        返回:
            hidden_states: [B,L,d_model]；pooled: [B,d_model]。
        """
        h = self.token_emb(token_ids)
        h = h + self.value_proj(values, value_masks)
        h = h + self.time_proj(abs_ages, interval_univ, interval_same)
        s_emb = self.static_proj(S)
        if s_emb is not None:
            h = h + s_emb.unsqueeze(1)
        h = self.dropout(h)
        # padding mask：True 位置被忽略
        src_key_padding_mask = (attn_masks == 0)
        hidden = self.encoder(h, src_key_padding_mask=src_key_padding_mask)
        hidden = self.norm(hidden)
        # masked pooled：只对真实 token 平均
        mask_f = attn_masks.to(hidden.dtype).unsqueeze(-1)
        pooled = (hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        return hidden, pooled


# ---------------------------------------------------------------------------
# 上游对比头（温度 tau 可配置）
# ---------------------------------------------------------------------------
class MaskedEventModelingHead(nn.Module):
    """Masked Event Modeling：token 分类 + value 回归。"""

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.token_head = nn.Linear(d_model, vocab_size)
        self.value_head = nn.Linear(d_model, 1)

    def forward(self, hidden):
        # value 是预处理归一化后的 [0,1] 值，sigmoid 将无界线性输出约束到 (0,1)，
        # 避免 MSE 被发散输出放大导致 mask_value 损失失稳。
        return self.token_head(hidden), torch.sigmoid(self.value_head(hidden)).squeeze(-1)


class VoidEpochContrastiveHead(nn.Module):
    """Void Epoch 对比：把 void epoch 表示与真实表示做 InfoNCE。"""

    def __init__(self, d_model: int, tau: float = 0.1):
        super().__init__()
        self.tau = float(tau)
        self.proj = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(),
                                  nn.Linear(d_model, d_model))

    def forward(self, pooled_real, pooled_void):
        z_r = F.normalize(self.proj(pooled_real), dim=-1)
        z_v = F.normalize(self.proj(pooled_void), dim=-1)
        logits = torch.matmul(z_r, z_v.t()) / self.tau
        labels = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, labels)


class AMCPCHead(nn.Module):
    """Autoregressive Multi-step Contrastive Predictive Coding：预测未来 K 步表示。"""

    def __init__(self, d_model: int, k_steps: int = 3, tau: float = 0.1):
        super().__init__()
        self.k_steps = int(k_steps)
        self.tau = float(tau)
        self.preds = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(self.k_steps)])

    def forward(self, hidden):
        # 对每个预测步，用当前表示预测未来时刻表示（对比 InfoNCE）
        losses = []
        B, L, D = hidden.shape
        for k in range(1, self.k_steps + 1):
            if L <= k + 1:
                continue
            cur = hidden[:, :-k, :]
            fut = hidden[:, k:, :]
            pred = self.preds[k - 1](cur)
            z1 = F.normalize(pred.reshape(-1, D), dim=-1)
            z2 = F.normalize(fut.reshape(-1, D), dim=-1)
            logits = torch.matmul(z1, z2.t()) / self.tau
            labels = torch.arange(z1.size(0), device=z1.device)
            losses.append(F.cross_entropy(logits, labels))
        if not losses:
            return hidden.new_zeros(())
        return torch.stack(losses).mean()


class TS2VecHead(nn.Module):
    """TS2Vec 时序对比：同序列两个增强视图的层级对比。"""

    def __init__(self, d_model: int, tau: float = 0.1):
        super().__init__()
        self.tau = float(tau)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, hidden_view1, hidden_view2):
        z1 = F.normalize(self.proj(hidden_view1.mean(dim=1)), dim=-1)
        z2 = F.normalize(self.proj(hidden_view2.mean(dim=1)), dim=-1)
        logits = torch.matmul(z1, z2.t()) / self.tau
        labels = torch.arange(z1.size(0), device=z1.device)
        return F.cross_entropy(logits, labels)


class TSTCCHead(nn.Module):
    """TSTCC 时间对比 + 示例对比（简化联合）。"""

    def __init__(self, d_model: int, tau: float = 0.1):
        super().__init__()
        self.tau = float(tau)
        self.time_proj = nn.Linear(d_model, d_model)
        self.instance_proj = nn.Linear(d_model, d_model)

    def forward(self, pooled_aug1, pooled_aug2):
        z1 = F.normalize(self.time_proj(pooled_aug1), dim=-1)
        z2 = F.normalize(self.time_proj(pooled_aug2), dim=-1)
        logits = torch.matmul(z1, z2.t()) / self.tau
        labels = torch.arange(z1.size(0), device=z1.device)
        return F.cross_entropy(logits, labels)


def amcpc_select_steps(seq_len: int, cfg: dict, history: dict = None) -> int:
    """按序列长度自适应选择 AMCPC 预测步长 K。

    K 从 {1, ..., cfg["up_amcpc_steps"]} 中选取：
    - 依据序列长度：K = clip(floor(log2(seq_len)) - 1, 1, K_max)；
    - 若提供历史使用次数 history（{K: 出现次数}）且候选 K 的出现次数 < 3，
      则退化为 K=1：样本过少时多步预测的负样本质量不可靠。
    输入:
        seq_len: 序列长度；cfg: 配置；history: 可选的历史使用次数统计。
    返回:
        int: 选中的预测步长 K。
    """
    k_max = max(1, int(cfg.get("up_amcpc_steps", 3)))
    k = int(min(k_max, max(1, int(math.floor(math.log2(max(2, int(seq_len))))) - 1)))
    if history is not None and int(history.get(k, 0)) < 3:
        k = 1
    return k


# ---------------------------------------------------------------------------
# 损失加权
# ---------------------------------------------------------------------------
def uncertainty_weighted_loss(loss_dict: dict, log_sigmas: nn.ParameterDict,
                             clip: float = None):
    """Kendall 不确定性加权：L = sum_i ( L_i / (2 sigma_i^2) + log sigma_i )。

    实现为 L_i * exp(-2*log_sigma_i) + log_sigma_i（等价形式，常数因子被吸收进
    可学习的 sigma 中）。注意：这是"训练用代理目标"，其数值可以为负
    （log sigma 项在 sigma<1 时为负），**不能当作可解释的损失值上报**；
    监控/记录请使用 loss_raw（各项损失的 λ 加权和）。

    clip: 可选，把 log_sigma 截断到 [-clip, clip]，防止 sigma 过小导致
    权重 exp(-2*log_sigma) 爆炸（进而引起混合精度梯度溢出）。

    输入:
        loss_dict: {name: scalar tensor}；log_sigmas: 可学习 log sigma 参数。
    返回:
        (total_loss, per_term_weighted)。
    """
    rng = None
    if clip is not None:
        rng = (-float(clip), float(clip))
    total = None
    weighted = {}
    for name, val in loss_dict.items():
        if name not in log_sigmas:
            continue
        ls = log_sigmas[name]
        if rng is not None:
            ls = torch.clamp(ls, rng[0], rng[1])
        term = val * torch.exp(-2.0 * ls) + ls
        weighted[name] = float(term.detach())
        total = term if total is None else total + term
    if total is None:
        total = torch.zeros((), device=next(iter(loss_dict.values())).device)
    return total, weighted


def gradnorm_weighted_loss(loss_dict: dict, shared_params, cfg: dict):
    """GradNorm 动态加权：按各损失对共享参数的梯度范数反比调整 λ。

    实现（单步、无需额外优化器）：
        G_i = || ∂L_i / ∂θ_shared ||_2
        λ_i = mean_j G_j / G_i，再整体归一化使 mean_i λ_i = 1
    梯度范数小的任务获得更大权重，抑制多任务训练中的梯度失衡。
    注意：需在开启梯度（训练模式）下调用；每步额外做 K 次反传，
    开销约 K 倍，仅在 up_loss_weighting=gradnorm 时启用。

    输入:
        loss_dict: {name: scalar tensor}；shared_params: 共享参数；cfg: 配置。
    返回:
        (total_loss, weights)。
    """
    names = list(loss_dict.keys())
    default = {n: 1.0 for n in names}
    params = [p for p in shared_params if p.requires_grad]
    weights = default
    if params and torch.is_grad_enabled():
        try:
            gnorms = {}
            for n in names:
                L = loss_dict[n]
                if (not torch.is_tensor(L)) or (not L.requires_grad):
                    continue
                grads = torch.autograd.grad(L, params, retain_graph=True,
                                            allow_unused=True)
                s = 0.0
                for g in grads:
                    if g is not None:
                        s += float(g.detach().pow(2).sum())
                if s > 0:
                    gnorms[n] = math.sqrt(s)
            if gnorms:
                mean_g = float(np.mean(list(gnorms.values())))
                raw = {n: (mean_g / gnorms[n] if n in gnorms else 1.0) for n in names}
                wsum = sum(raw.values())
                if wsum > 0:
                    weights = {n: raw[n] * len(names) / wsum for n in names}
        except Exception as exc:  # noqa: BLE001
            logger.warning("[STEP MODEL] GradNorm 权重计算失败，退回均匀权重: %s", exc)
            weights = default
    total = None
    for n in names:
        term = loss_dict[n] * weights[n]
        total = term if total is None else total + term
    if total is None:
        total = torch.zeros((), device=next(iter(loss_dict.values())).device)
    return total, weights


# ---------------------------------------------------------------------------
# 损失函数
# ---------------------------------------------------------------------------
def masked_event_modeling_loss(token_logits, value_pred, token_ids, values,
                               value_masks, mask_positions, cfg, token_freq=None):
    """Masked event modeling 损失（token CE + value MSE）与监控指标。

    token 交叉熵按 w_i = 1/sqrt(freq_i+1)（或 freq_inv）加权，
    在被 mask 位置上做权重归一化的加权平均；cfg["up_token_loss_weighting"] 控制。

    输入:
        token_logits: [B,L,V]；value_pred: [B,L]；token_ids: [B,L]；
        values/value_masks: [B,L]；mask_positions: [B,L] bool；
        cfg: 配置；token_freq: {token_id: count}。
    返回:
        dict: {"mask_token", "mask_value", "token_acc", "value_mae"}。
    """
    B, L, V = token_logits.shape
    logits_flat = token_logits.reshape(-1, V)
    targets_flat = token_ids.reshape(-1).long()
    ce_flat = F.cross_entropy(logits_flat, targets_flat, reduction="none")
    preds_flat = logits_flat.argmax(dim=-1)

    sel = mask_positions.reshape(-1).bool()
    if int(sel.sum()) > 0:
        ce_sel = ce_flat[sel]
        tgt_sel = targets_flat[sel]
        mode = str(cfg.get("up_token_loss_weighting", "freq_inv_sqrt"))
        if mode != "none" and token_freq:
            freq = torch.tensor(
                [float(token_freq.get(int(i), 1)) for i in tgt_sel.tolist()],
                dtype=torch.float32, device=ce_flat.device)
            w = 1.0 / torch.sqrt(freq + 1.0) if mode == "freq_inv_sqrt" else 1.0 / (freq + 1.0)
            w = w / w.mean().clamp(min=1e-8)
            token_loss = (ce_sel * w).sum() / w.sum().clamp(min=1e-8)
        else:
            token_loss = ce_sel.mean()
        token_acc = (preds_flat[sel] == tgt_sel).float().mean()
    else:
        token_loss = ce_flat.new_zeros(())
        token_acc = ce_flat.new_zeros(())

    # value MSE / MAE（仅被 mask 且 value_mask>0 的位置）
    vm = mask_positions.bool() & (value_masks > 0)
    if int(vm.sum()) > 0:
        value_loss = F.mse_loss(value_pred[vm], values[vm])
        value_mae = (value_pred[vm] - values[vm]).abs().mean()
    else:
        value_loss = value_pred.new_zeros(())
        value_mae = value_pred.new_zeros(())
    return {"mask_token": token_loss, "mask_value": value_loss,
            "token_acc": token_acc.detach(), "value_mae": value_mae.detach()}


def void_epoch_contrastive_loss(pooled_real, pooled_void, cfg, head=None):
    """Void epoch 对比损失（InfoNCE）。

    head 为 None 时按 cfg 临时构建（仅用于独立调用/单元测试）；
    训练路径请传入 UpstreamModel 中已注册的 head，避免参数不被优化。
    """
    if head is None:
        head = VoidEpochContrastiveHead(
            pooled_real.size(-1), tau=cfg.get("up_tau", 0.1)).to(pooled_real.device)
        logger.warning("[STEP MODEL] void_epoch_contrastive_loss 未传入 head，使用临时头")
    return head(pooled_real, pooled_void)


def amcpc_loss(hidden, cfg, head=None):
    """AMCPC 损失（K 由 amcpc_select_steps 自适应选取）。"""
    if head is None:
        k = amcpc_select_steps(hidden.size(1), cfg)
        head = AMCPCHead(hidden.size(-1), k_steps=k,
                         tau=cfg.get("up_tau", 0.1)).to(hidden.device)
        logger.warning("[STEP MODEL] amcpc_loss 未传入 head，使用临时头")
    return head(hidden)


def ts2vec_loss(hidden_v1, hidden_v2, cfg, head=None):
    """TS2Vec 损失。"""
    if head is None:
        head = TS2VecHead(hidden_v1.size(-1),
                          tau=cfg.get("up_tau", 0.1)).to(hidden_v1.device)
        logger.warning("[STEP MODEL] ts2vec_loss 未传入 head，使用临时头")
    return head(hidden_v1, hidden_v2)


def tstcc_loss(pooled_aug1, pooled_aug2, cfg, head=None):
    """TSTCC 损失。"""
    if head is None:
        head = TSTCCHead(pooled_aug1.size(-1),
                         tau=cfg.get("up_tau", 0.1)).to(pooled_aug1.device)
        logger.warning("[STEP MODEL] tstcc_loss 未传入 head，使用临时头")
    return head(pooled_aug1, pooled_aug2)


def downstream_regression_loss(y_pred, y, cfg):
    """下游回归损失（cfg["down_loss"]：l1 / smooth_l1 / mse）。

    数值稳定性说明：AMP(fp16) 下 mse 对残差做平方会显著放大梯度，若回归目标
    （如透析龄/临床指标）未标准化到较小量级，极易触发 GradScaler 梯度溢出。
    这里对目标做逐 batch 稳健缩放（除以中位绝对偏差 MAD + eps）以稳定量级，
    等价于对损失做自适应缩放，不改变 argmin 方向；l1 本身量级温和，不做缩放。
    """
    mode = str(cfg.get("down_loss", "l1"))
    y = y.float()
    y_pred = y_pred.float()
    if mode == "l1":
        return F.l1_loss(y_pred, y)
    if mode == "mse":
        diff = y_pred - y
        scale = torch.median(torch.abs(y - torch.median(y))).clamp(min=1e-3) + 1e-6
        return F.mse_loss(diff / scale, torch.zeros_like(diff))
    # smooth_l1
    return F.smooth_l1_loss(y_pred, y, beta=0.1)


# ---------------------------------------------------------------------------
# 数据增强
# ---------------------------------------------------------------------------
_SEQ_KEYS = ("token_ids", "values", "value_masks", "attn_masks",
             "abs_ages", "interval_univ", "interval_same")


def _clone_tensor_dict(tensor_dict: dict) -> dict:
    """浅拷贝 + 张量 clone（不修改原对象）。"""
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in tensor_dict.items()}


def _token_dropout(out: dict, p: float) -> dict:
    """随机把真实 token 置为 PAD，并同步 attn_mask（原地作用于 out）。"""
    ids = out["token_ids"]
    attn = out["attn_masks"].bool()
    drop = (torch.rand_like(ids.float()) < p) & attn
    ids[drop] = PAD_TOKEN_ID
    out["attn_masks"] = (~drop).to(out["attn_masks"].dtype) * out["attn_masks"]
    return out


def _visit_segments(attn_1d: torch.Tensor, interval_1d: torch.Tensor):
    """按 visit 切分序列：[start, end) 列表；interval>0 的位置标记新 visit。"""
    L = int(attn_1d.numel())
    starts = []
    for i in range(L):
        if not bool(attn_1d[i]):
            continue
        if not starts or float(interval_1d[i]) > 1e-6:
            starts.append(i)
    segs = []
    for j, s in enumerate(starts):
        e = starts[j + 1] if j + 1 < len(starts) else L
        if e > s:
            segs.append((s, e))
    return segs


def augment_tokens_strong(tensor_dict: dict, cfg: dict) -> dict:
    """强增强：token dropout(p=0.1) + 时间抖动(±1 visit) + 值缩放(×0.9~1.1)
    + visit 内 token 置换。

    输入:
        tensor_dict: batch 张量字典；cfg: 配置。
    返回:
        增强后的 tensor_dict（新 dict，不修改原对象）。
    """
    out = _clone_tensor_dict(tensor_dict)
    L = out["token_ids"].shape[-1]
    out = _token_dropout(out, 0.1)

    # 值缩放
    if "values" in out:
        out["values"] = out["values"] * (0.9 + 0.2 * torch.rand_like(out["values"]))

    # 时间抖动（±1 visit）：按样本随机整体前移/后移一位，裁剪到 >= 0
    for key in ("interval_univ", "interval_same"):
        if key not in out:
            continue
        shifted = out[key].clone()
        for b in range(out[key].shape[0]):
            if random.random() < 0.5:
                continue
            step = 1 if random.random() < 0.5 else -1
            shifted[b] = torch.roll(out[key][b], shifts=step, dims=0).clamp(min=0.0)
        out[key] = shifted

    # visit 内 token 置换
    for b in range(out["token_ids"].shape[0]):
        segs = _visit_segments(out["attn_masks"][b].bool(), out["interval_univ"][b])
        for s, e in segs:
            n = e - s
            if n < 2:
                continue
            perm = torch.randperm(n, device=out["token_ids"].device)
            for key in _SEQ_KEYS:
                if key in out and out[key].shape[-1] == L:
                    out[key][b, s:e] = out[key][b, s:e][perm]
    return out


def augment_tokens_weak(tensor_dict: dict, cfg: dict) -> dict:
    """弱增强：仅 token dropout(p=0.1)。"""
    return _token_dropout(_clone_tensor_dict(tensor_dict), 0.1)


def crop_sequence(tensor_dict: dict, ratio_range=(0.7, 1.0)) -> dict:
    """随机 crop 序列（TS2Vec 用）：保留连续片段并右对齐。"""
    L = tensor_dict["token_ids"].shape[-1]
    lo, hi = ratio_range
    keep = max(1, int(L * (lo + (hi - lo) * random.random())))
    start = random.randint(0, L - keep)
    out = {}
    for k, v in tensor_dict.items():
        if torch.is_tensor(v) and v.shape[-1] == L:
            cropped = v[..., start:start + keep]
            out[k] = F.pad(cropped, (L - keep, 0))
        else:
            out[k] = v
    return out


def insert_void_epochs(tensor_dict: dict, cfg: dict) -> dict:
    """插入 void epoch：每个样本随机选 int(void_ratio * n_tokens) 个真实位置，
    将其 token 置为 RARE、value/value_mask 清零，作为"空 epoch"表示。

    输入:
        tensor_dict: batch 张量字典；cfg: 配置。
    返回:
        含 void 扰动的 batch（新 dict）。
    """
    out = _clone_tensor_dict(tensor_dict)
    attn = out["attn_masks"].bool()
    n_tokens = attn.sum(dim=-1)
    n_void = (n_tokens.float() * float(cfg.get("up_void_ratio", 0.15))).long()
    B, L = out["token_ids"].shape
    for b in range(B):
        idx = torch.where(attn[b])[0]
        if len(idx) == 0 or int(n_void[b]) == 0:
            continue
        pick = idx[torch.randperm(len(idx), device=idx.device)[:int(n_void[b])]]
        out["token_ids"][b, pick] = RARE_TOKEN_ID
        out["values"][b, pick] = 0.0
        out["value_masks"][b, pick] = 0
    return out


# ---------------------------------------------------------------------------
# 上游模型（聚合编码器 + 激活对比头）
# ---------------------------------------------------------------------------
class UpstreamModel(nn.Module):
    """根据 cfg["upstream_method"] 激活对应对比头，forward 返回损失字典。

    返回的 dict 同时包含损失项（参与加权）与监控指标（token_acc / value_mae）；
    参与加权的损失项由 self.loss_keys 显式列出，避免监控指标被误当作损失。
    """

    def __init__(self, cfg: dict, vocab_size: int, d_static: int, token_freq=None):
        super().__init__()
        self.cfg = cfg
        self.method = cfg["upstream_method"]
        self.mask_ratio = float(cfg.get("up_mask_ratio", 0.15))
        self.loss_weighting = cfg.get("up_loss_weighting", "uncertainty")
        self.log_sigma_clip = float(cfg.get("up_log_sigma_clip", 2.0))
        tau = float(cfg.get("up_tau", 0.1))
        self.encoder = UpstreamEventEncoder(
            vocab_size=vocab_size, d_static=d_static,
            d_model=cfg["d_model"], n_heads=cfg["n_heads"],
            n_layers=cfg["n_layers"], dropout=cfg["dropout"])
        self.mem_head = MaskedEventModelingHead(cfg["d_model"], vocab_size)
        self.use_void = self.method in ("void", "all")
        self.use_amcpc = self.method in ("amcpc", "all")
        self.use_ts2vec = self.method in ("ts2vec", "all")
        self.use_tstcc = self.method in ("tstcc", "all")
        # 对比头注册为子模块（避免 forward 中临时 new 导致参数不被优化）
        if self.use_void:
            self.void_head = VoidEpochContrastiveHead(cfg["d_model"], tau=tau)
        if self.use_amcpc:
            self.amcpc_k = amcpc_select_steps(cfg["seq_length"], cfg)
            self.amcpc_head = AMCPCHead(cfg["d_model"], k_steps=self.amcpc_k, tau=tau)
        if self.use_ts2vec:
            self.ts2vec_head = TS2VecHead(cfg["d_model"], tau=tau)
        if self.use_tstcc:
            self.tstcc_head = TSTCCHead(cfg["d_model"], tau=tau)
        # 所有方法默认带 masked event modeling 主任务
        self.token_freq = token_freq or {}
        self.loss_keys = ["mask_token", "mask_value"]
        if self.use_void:
            self.loss_keys.append("void")
        if self.use_amcpc:
            self.loss_keys.append("amcpc")
        if self.use_ts2vec:
            self.loss_keys.append("ts2vec")
        if self.use_tstcc:
            self.loss_keys.append("tstcc")
        # 不确定性加权可学习 log_sigma（仅损失项）
        self.log_sigmas = nn.ParameterDict(
            {n: nn.Parameter(torch.zeros(())) for n in self.loss_keys})

    def make_mask_positions(self, attn_masks, generator=None):
        """随机 mask ratio 比例的真实 token，返回 bool 掩码 [B,L]。

        传入 generator 可让评估阶段的 mask 采样可复现，使 val_loss 跨轮可比。
        """
        attn = attn_masks.bool()
        mask_pos = torch.zeros_like(attn, dtype=torch.bool)
        n_total = int(attn.sum().item())
        if n_total == 0:
            return mask_pos
        n_mask = max(1, int(self.mask_ratio * n_total))
        rows, cols = torch.where(attn)
        perm = torch.randperm(rows.numel(), generator=generator)[:n_mask].to(attn.device)
        mask_pos[rows[perm], cols[perm]] = True
        return mask_pos

    def mask_input(self, batch: dict, mask_pos: torch.Tensor) -> dict:
        """把被选中的位置"挖空"：token->MASK、value/value_mask 清零。

        这是 masked event modeling 的关键一步：如果直接把原始 token 喂进编码器，
        模型可以从自己的输入 embedding 里直接"抄答案"（token_acc 会迅速到 1.0、
        损失趋 0），自监督预训练就退化成恒等映射。attn_mask 保持不变——位置仍是
        一次真实就诊，只是"内容未知"，需要模型依据上下文预测。
        """
        out = dict(batch)
        ids = batch["token_ids"].clone()
        vals = batch["values"].clone()
        vmasks = batch["value_masks"].clone()
        ids[mask_pos] = MASK_TOKEN_ID
        vals[mask_pos] = 0.0
        vmasks[mask_pos] = 0
        out["token_ids"] = ids
        out["values"] = vals
        out["value_masks"] = vmasks
        return out

    def _raw_loss(self, loss_dict: dict):
        """各项损失的 λ 加权和：非负、可解释，用于打印/记录（不参与反向）。"""
        w = self._fixed_loss_weights()
        total = None
        for k, v in loss_dict.items():
            term = v * w.get(k, 1.0)
            total = term if total is None else total + term
        return total

    def _fixed_loss_weights(self) -> dict:
        """fixed 模式下的 λ。"""
        return {"mask_token": self.cfg["lambda_mask"],
                "mask_value": self.cfg["lambda_mask"],
                "void": self.cfg["lambda_void"],
                "amcpc": self.cfg["lambda_amcpc"],
                "ts2vec": self.cfg["lambda_ts2vec"],
                "tstcc": self.cfg["lambda_tstcc"]}

    def _combine_losses(self, loss_dict: dict):
        """按 cfg["up_loss_weighting"] 组合损失项，返回总损失。"""
        mode = self.loss_weighting
        if mode == "gradnorm" and not torch.is_grad_enabled():
            # 评估阶段（no_grad）无法求梯度范数，退回固定权重保证 val 指标可比。
            mode = "fixed"
        if mode == "uncertainty":
            total, _ = uncertainty_weighted_loss(loss_dict, self.log_sigmas,
                                                 clip=self.log_sigma_clip)
            return total
        if mode == "gradnorm":
            total, _ = gradnorm_weighted_loss(
                loss_dict, list(self.encoder.parameters()), self.cfg)
            return total
        weights = self._fixed_loss_weights()
        total = None
        for k, v in loss_dict.items():
            term = v * weights.get(k, 1.0)
            total = term if total is None else total + term
        return total

    def forward(self, batch: dict, generator=None) -> dict:
        """前向：返回 {损失项...，token_acc，value_mae，loss_total}。

        - loss_total：训练用代理目标（随 up_loss_weighting 变化，可为负，用于 backward）；
        - loss_raw  ：各项损失的 λ 加权和（非负、可解释，用于打印/记录/选模型参考）。

        输入:
            batch: 搬到 device 后的张量字典；generator: 可选 torch.Generator。
        返回:
            dict: 各损失分量 + 监控指标 + loss_raw + loss_total。
        """
        cfg = self.cfg
        # 1) masked event modeling：先把输入"挖空"，再预测被挖空位置
        mask_pos = self.make_mask_positions(batch["attn_masks"], generator)
        mem_batch = self.mask_input(batch, mask_pos)
        hidden_masked, _ = self.encoder(
            mem_batch["token_ids"], mem_batch["values"], mem_batch["value_masks"],
            mem_batch["attn_masks"], mem_batch["abs_ages"],
            mem_batch["interval_univ"], mem_batch["interval_same"], mem_batch["S"])
        token_logits, value_pred = self.mem_head(hidden_masked)
        mem = masked_event_modeling_loss(
            token_logits, value_pred, batch["token_ids"], batch["values"],
            batch["value_masks"], mask_pos, cfg, self.token_freq)
        out = {"mask_token": mem["mask_token"], "mask_value": mem["mask_value"]}

        # 对比头使用"未挖空"的干净表示（与 MEM 解耦，避免互相干扰）
        hidden = pooled = None
        if self.use_void or self.use_amcpc:
            hidden, pooled = self.encoder(
                batch["token_ids"], batch["values"], batch["value_masks"],
                batch["attn_masks"], batch["abs_ages"],
                batch["interval_univ"], batch["interval_same"], batch["S"])

        # 2) void epoch
        if self.use_void:
            void_batch = insert_void_epochs(batch, cfg)
            _, pooled_void = self.encoder(
                void_batch["token_ids"], void_batch["values"], void_batch["value_masks"],
                void_batch["attn_masks"], void_batch["abs_ages"],
                void_batch["interval_univ"], void_batch["interval_same"], void_batch["S"])
            out["void"] = self.void_head(pooled, pooled_void)

        # 3) amcpc
        if self.use_amcpc:
            out["amcpc"] = self.amcpc_head(hidden)

        # 4) ts2vec：两个 crop 视图
        if self.use_ts2vec:
            v1 = crop_sequence(batch)
            v2 = crop_sequence(batch)
            h1, _ = self.encoder(v1["token_ids"], v1["values"], v1["value_masks"],
                                 v1["attn_masks"], v1["abs_ages"],
                                 v1["interval_univ"], v1["interval_same"], v1["S"])
            h2, _ = self.encoder(v2["token_ids"], v2["values"], v2["value_masks"],
                                 v2["attn_masks"], v2["abs_ages"],
                                 v2["interval_univ"], v2["interval_same"], v2["S"])
            out["ts2vec"] = self.ts2vec_head(h1, h2)

        # 5) tstcc：弱增强双视图
        if self.use_tstcc:
            a1 = augment_tokens_weak(batch, cfg)
            a2 = augment_tokens_weak(batch, cfg)
            _, p1 = self.encoder(a1["token_ids"], a1["values"], a1["value_masks"],
                                 a1["attn_masks"], a1["abs_ages"],
                                 a1["interval_univ"], a1["interval_same"], a1["S"])
            _, p2 = self.encoder(a2["token_ids"], a2["values"], a2["value_masks"],
                                 a2["attn_masks"], a2["abs_ages"],
                                 a2["interval_univ"], a2["interval_same"], a2["S"])
            out["tstcc"] = self.tstcc_head(p1, p2)

        # 监控指标（不参与损失加权）
        out["token_acc"] = mem["token_acc"]
        out["value_mae"] = mem["value_mae"]
        # 加权汇总：loss_raw 供监控，loss_total 供反向
        loss_dict = {k: out[k] for k in self.loss_keys}
        out["loss_raw"] = self._raw_loss(loss_dict)
        out["loss_total"] = self._combine_losses(loss_dict)
        return out


# ---------------------------------------------------------------------------
# 下游模型
# ---------------------------------------------------------------------------
class _DownstreamBase(nn.Module):
    """下游公共基类：共享上游编码器结构 + 掩码池化 + 回归头。"""

    def __init__(self, cfg, vocab_size, d_static, dropout=None):
        super().__init__()
        self.cfg = cfg
        self.d_model = cfg["d_model"]
        self.d_static = d_static
        self.encoder = UpstreamEventEncoder(
            vocab_size, d_static, cfg["d_model"], cfg["n_heads"],
            cfg["n_layers"], cfg["dropout"])
        hid = max(1, cfg["d_model"] // 2)
        self.regressor = nn.Sequential(
            nn.Linear(cfg["d_model"], hid), nn.ReLU(),
            nn.Dropout(_head_dropout(cfg, dropout)), nn.Linear(hid, 1))

    def _encode(self, batch):
        """编码 batch，返回 (hidden, pooled)。"""
        return self.encoder(
            batch["token_ids"], batch["values"], batch["value_masks"],
            batch["attn_masks"], batch["abs_ages"],
            batch["interval_univ"], batch["interval_same"], batch["S"])

    def forward(self, batch):
        _, pooled = self._encode(batch)
        return self.regressor(pooled).squeeze(-1)


class STraTSModel(_DownstreamBase):
    """STraTS：复用上游编码器（Token/Value/Time/Static 投影 + Transformer），
    在其 pooled 表示上接回归头。"""


class DuETTModel(_DownstreamBase):
    """DuETT：在事件编码之上叠加时间维度注意力（双注意力）。"""

    def __init__(self, cfg, vocab_size, d_static, dropout=None):
        super().__init__(cfg, vocab_size, d_static, dropout)
        head_drop = _head_dropout(cfg, dropout)
        self.time_attn = nn.MultiheadAttention(
            cfg["d_model"], cfg["n_heads"], dropout=head_drop, batch_first=True)
        self.time_gate = nn.Linear(cfg["d_model"] * 2, cfg["d_model"])
        self.time_norm = nn.LayerNorm(cfg["d_model"])

    def forward(self, batch):
        hidden, pooled = self._encode(batch)
        key_pad = (batch["attn_masks"] == 0)
        h2, _ = self.time_attn(hidden, hidden, hidden, key_padding_mask=key_pad)
        # 门控 tanh 与掩码池化在 fp16 下精度/数值稳定性较差，强制 fp32 计算，
        # 避免 AMP 下梯度溢出或精度损失（结果再转回 hidden 的 dtype）。
        hidden32 = hidden.float()
        h2_32 = h2.float()
        mixed = torch.tanh(self.time_gate(torch.cat([hidden32, h2_32], dim=-1)))
        mixed = self.time_norm(mixed)
        mask_f = batch["attn_masks"].to(mixed.dtype).unsqueeze(-1)
        pooled = (mixed * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        pooled = pooled.to(hidden.dtype)
        return self.regressor(pooled).squeeze(-1)


class GRUDModel(nn.Module):
    """GRU-D（独立模型，不使用上游编码器）：输入级衰减 + 状态级衰减。

    由于输入是事件 Token 序列而非定长多变量序列，这里按 GRU-D 的思想实现：
    - 输入级衰减：h_in ← h_in * exp(-softplus(W_x·Δ))；
    - 状态级衰减：h_{t-1} ← h_{t-1} * exp(-softplus(W_h·Δ))；
    其中 Δ = interval_univ（log1p 天），Δ 越大衰减越强。
    """

    def __init__(self, cfg, vocab_size, d_static, dropout=None):
        super().__init__()
        d = cfg["d_model"]
        self.d_model = d
        self.d_static = d_static
        self.token_emb = nn.Embedding(vocab_size, d, padding_idx=PAD_TOKEN_ID)
        self.value_proj = ValueProjection(d)
        self.time_proj = TimeFeatureProjection(d)
        self.static_proj = StaticProjection(d_static, d)
        self.gru_cell = nn.GRUCell(d, d)
        self.decay_x = nn.Linear(1, 1)
        self.decay_h = nn.Linear(1, d)
        self.out_norm = nn.LayerNorm(d)
        hid = max(1, d // 2)
        self.regressor = nn.Sequential(
            nn.Linear(d + d_static, hid), nn.ReLU(),
            nn.Dropout(_head_dropout(cfg, dropout)), nn.Linear(hid, 1))

    def forward(self, batch):
        d = self.d_model
        attn = batch["attn_masks"].to(batch["token_ids"].device).float()
        delta = batch["interval_univ"].unsqueeze(-1)               # [B,L,1]
        gamma_x = torch.exp(-F.softplus(self.decay_x(delta)))      # [B,L,1]
        h_in = (self.token_emb(batch["token_ids"])
                + self.value_proj(batch["values"], batch["value_masks"])
                + self.time_proj(batch["abs_ages"], batch["interval_univ"],
                                 batch["interval_same"]))
        s_emb = self.static_proj(batch["S"])
        if s_emb is not None:
            h_in = h_in + s_emb.unsqueeze(1)
        h_in = h_in * gamma_x

        B, L, _ = h_in.shape
        h = torch.zeros(B, d, device=h_in.device, dtype=h_in.dtype)
        outs = []
        for t in range(L):
            gamma_h = torch.exp(-F.softplus(self.decay_h(delta[:, t, :])))  # [B,d]
            h_new = self.gru_cell(h_in[:, t, :], h * gamma_h)
            m = attn[:, t].unsqueeze(-1)
            h = m * h_new + (1.0 - m) * h
            outs.append(h)
        out = self.out_norm(torch.stack(outs, dim=1))              # [B,L,d]
        mask_f = attn.unsqueeze(-1)
        pooled = (out * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        pooled = torch.cat([pooled, batch["S"]], dim=-1)
        return self.regressor(pooled).squeeze(-1)


class LSTMModel(nn.Module):
    """LSTM + 掩码池化（独立模型，不使用上游编码器）。"""

    def __init__(self, cfg, vocab_size, d_static, dropout=None):
        super().__init__()
        d = cfg["d_model"]
        self.d_static = d_static
        self.token_emb = nn.Embedding(vocab_size, d, padding_idx=PAD_TOKEN_ID)
        self.value_proj = ValueProjection(d)
        self.time_proj = TimeFeatureProjection(d)
        self.static_proj = StaticProjection(d_static, d)
        self.lstm = nn.LSTM(d, d, batch_first=True, num_layers=1)
        self.out_norm = nn.LayerNorm(d)
        hid = max(1, d // 2)
        self.regressor = nn.Sequential(
            nn.Linear(d + d_static, hid), nn.ReLU(),
            nn.Dropout(_head_dropout(cfg, dropout)), nn.Linear(hid, 1))

    def forward(self, batch):
        attn = batch["attn_masks"].to(batch["token_ids"].device).float()
        h = (self.token_emb(batch["token_ids"])
             + self.value_proj(batch["values"], batch["value_masks"])
             + self.time_proj(batch["abs_ages"], batch["interval_univ"],
                              batch["interval_same"]))
        s_emb = self.static_proj(batch["S"])
        if s_emb is not None:
            h = h + s_emb.unsqueeze(1)
        out, _ = self.lstm(h)
        out = self.out_norm(out)
        mask_f = attn.unsqueeze(-1)
        pooled = (out * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        pooled = torch.cat([pooled, batch["S"]], dim=-1)
        return self.regressor(pooled).squeeze(-1)


class TransformerBaseline(nn.Module):
    """标准 Transformer + 掩码平均池化（结构与上游编码器一致，便于权重复用）。"""

    def __init__(self, cfg, vocab_size, d_static, dropout=None):
        super().__init__()
        self.encoder = UpstreamEventEncoder(
            vocab_size, d_static, cfg["d_model"], cfg["n_heads"],
            cfg["n_layers"], cfg["dropout"])
        hid = max(1, cfg["d_model"] // 2)
        self.regressor = nn.Sequential(
            nn.Linear(cfg["d_model"], hid), nn.ReLU(),
            nn.Dropout(_head_dropout(cfg, dropout)), nn.Linear(hid, 1))

    def forward(self, batch):
        _, pooled = self.encoder(
            batch["token_ids"], batch["values"], batch["value_masks"],
            batch["attn_masks"], batch["abs_ages"],
            batch["interval_univ"], batch["interval_same"], batch["S"])
        return self.regressor(pooled).squeeze(-1)


# ---------------------------------------------------------------------------
# 构建函数
# ---------------------------------------------------------------------------
def build_upstream_model(cfg: dict, vocab_size: int, d_static: int, token_freq=None):
    """根据 cfg["upstream_method"] 构建上游模型。

    输入:
        cfg: 配置；vocab_size: 词表大小；d_static: 静态特征维度；token_freq: 词频。
    返回:
        UpstreamModel。
    """
    model = UpstreamModel(cfg, vocab_size, d_static, token_freq)
    active = ["mask"]
    for name, flag in (("void", model.use_void), ("amcpc", model.use_amcpc),
                       ("ts2vec", model.use_ts2vec), ("tstcc", model.use_tstcc)):
        if flag:
            active.append(name)
    n_param = count_parameters(model)
    logger.info("[STEP MODEL.1] 构建上游模型 method=%s 参数量=%d 激活损失=%s "
                "损失权重=%s 对比温度=%.3f", cfg["upstream_method"], n_param,
                active, cfg["up_loss_weighting"], float(cfg.get("up_tau", 0.1)))
    return model


def load_upstream_encoder_weights(model: nn.Module, ckpt_path: str, cfg: dict,
                                  arch: str = "reuse") -> None:
    """从上游 checkpoint 加载编码器权重到下游模型。

    输入:
        model: 下游模型；ckpt_path: 上游 best checkpoint 路径；cfg: 配置；
        arch: reuse/full（full 模式表示下游自建编码器，不加载上游权重）。
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"上游 checkpoint 不存在: {ckpt_path}")
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        logger.warning("[STEP MODEL.2] 模型 %s 不含上游编码器，跳过权重加载",
                       type(model).__name__)
        return
    if arch == "full":
        logger.warning("[STEP MODEL.2] downstream_arch=full：忽略上游权重（等效 scratch）")
        return
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    enc_sd = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
    if not enc_sd:
        logger.warning("[STEP MODEL.2] %s 中未找到 encoder.* 权重，跳过加载", ckpt_path)
        return
    try:
        missing, unexpected = encoder.load_state_dict(enc_sd, strict=False)
    except RuntimeError as exc:
        raise RuntimeError(
            f"上游编码器权重与下游编码器结构不匹配（请检查 d_model/n_heads/n_layers/"
            f"dropout/词表/静态特征维度是否一致）: {exc}") from exc
    logger.info("[STEP MODEL.2] 加载上游编码器权重 %s (missing=%d, unexpected=%d)",
                ckpt_path, len(missing), len(unexpected))
    if cfg.get("freeze_upstream", False):
        for p in encoder.parameters():
            p.requires_grad = False
        logger.info("[STEP MODEL.2] 已冻结上游编码器（freeze_upstream=True）")


def build_downstream_model(cfg: dict, vocab_size: int, d_static: int,
                           upstream_ckpt: str = None, dropout=None):
    """根据 cfg["downstream_model"] 构建下游模型。

    输入:
        cfg: 配置；vocab_size/d_static: 维度；upstream_ckpt: 上游 best 路径
        （downstream_init=upstream 且 arch=reuse 时加载）；dropout: 下游头 Dropout
        覆盖值（HPO 用）。
    返回:
        下游 nn.Module。
    """
    name = cfg["downstream_model"]
    model_map = {
        "strats": STraTSModel, "duett": DuETTModel, "gru_d": GRUDModel,
        "lstm": LSTMModel, "transformer": TransformerBaseline,
    }
    if name not in model_map:
        raise ValueError(f"未知 downstream_model: {name}")
    model = model_map[name](cfg, vocab_size, d_static, dropout=dropout)

    if cfg["downstream_init"] == "upstream":
        if upstream_ckpt and os.path.exists(upstream_ckpt):
            load_upstream_encoder_weights(model, upstream_ckpt, cfg,
                                          arch=cfg["downstream_arch"])
        else:
            logger.warning("[STEP MODEL.3] 未找到上游 checkpoint（%s），"
                           "下游以随机初始化开始", upstream_ckpt)
    n_param = count_parameters(model)
    logger.info("[STEP MODEL.3] 构建下游模型 %s arch=%s init=%s 参数量=%d "
                "down_dropout=%.3f", name, cfg["downstream_arch"],
                cfg["downstream_init"], n_param, _head_dropout(cfg, dropout))
    return model
