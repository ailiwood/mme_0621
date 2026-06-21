"""
models/fusion/awaf.py — AWAF: Adaptive Weighted Attention Fusion

论文核心创新模块。样本级自适应加权注意力融合。

两段式设计:
  第一段: 轻量跨模态上下文增强 (Context Enhancement)
  第二段: 一阶模态摘要 + 二阶 Hadamard 交互项生成样本级权重 (Interaction Scoring)

输入:  h_t, h_a, h_v ∈ R^d     (三模态池化后摘要向量)
输出:  Z ∈ R^d                   (融合表示)
       w = [w_t, w_a, w_v]       (样本级三模态权重, sum(w) = 1)

支持 8 种融合模式:
  awaf               完整 AWAF
  awaf_no_context    去掉跨模态上下文增强
  awaf_no_interaction 去掉二阶交互项
  mean               等权平均
  concat             拼接 + MLP
  gated              普通门控融合
  fixed              全局固定可学习权重
  modality_dropout_off 关闭模态 dropout (在 awaf 基础上)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np


class AdaptiveWeightedAttentionFusion(nn.Module):
    """
    样本级自适应加权注意力融合 (AWAF)。

    第一段: 跨模态上下文增强
      对每个模态 h_m，以 h_m 为 query，其他模态为 key/value，
      得到上下文向量 c_m。
      增强表示: ĥ_m = LayerNorm(h_m + c_m)

    第二段: 二阶交互打分
      g_ta = ĥ_t ⊙ ĥ_a
      g_tv = ĥ_t ⊙ ĥ_v
      g_av = ĥ_a ⊙ ĥ_v
      e = MLP([ĥ_t, ĥ_a, ĥ_v, g_ta, g_tv, g_av]) ∈ R^3
      w = softmax(e / τ)
      Z = w_t·ĥ_t + w_a·ĥ_a + w_v·ĥ_v
    """

    def __init__(
        self,
        hidden_dim: int,
        fusion_mode: str = 'awaf',
        tau_init: float = 1.0,
        dropout: float = 0.1,
        use_modality_dropout: bool = True,
        modality_dropout_prob: float = 0.1,
        context_hidden_ratio: float = 0.5,
        eps: float = 1e-8,
        # === P6H-R 修复开关 ===
        use_modal_layernorm: bool = False,       # 模态输入 LayerNorm
        awaf_uniform_mix: float = 0.0,           # weight floor ε: w=(1-ε)*softmax+ε/3
        return_diagnostics: bool = False,         # 返回 L2 norm / entropy 诊断
    ):
        """
        Args:
            hidden_dim: 三模态摘要向量的维度 d
            fusion_mode: 融合模式
                'awaf' | 'awaf_no_context' | 'awaf_no_interaction' |
                'mean' | 'concat' | 'gated' | 'fixed'
            tau_init: 温度参数初始值 (≥0.1, 数值稳定)
            dropout: MLP 中的 dropout
            use_modality_dropout: 是否启用模态 dropout (仅在 awaf 系列模式下)
            modality_dropout_prob: 模态 dropout 概率
            context_hidden_ratio: 上下文增强中 attention 的 hidden dim 比例
            eps: 数值稳定常数
            use_modal_layernorm: [P6H-R] 在 AWAF 前对各模态独立 LayerNorm
            awaf_uniform_mix: [P6H-R] 权重均匀混合，防止单模态崩溃
            return_diagnostics: [P6H-R] 输出 L2 norm 和 entropy 诊断
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fusion_mode = fusion_mode
        self.use_modality_dropout = use_modality_dropout and ('awaf' in fusion_mode)
        self.modality_dropout_prob = modality_dropout_prob
        self.eps = eps
        self.awaf_uniform_mix = awaf_uniform_mix
        self.return_diagnostics = return_diagnostics

        # --- 可学习温度 τ ---
        # 用 softplus 保证 τ ≥ 0.1
        self.tau_raw = nn.Parameter(torch.tensor(self._inv_softplus(tau_init)))

        # === P6H-R: 模态输入 LayerNorm ===
        self.use_modal_layernorm = use_modal_layernorm
        if use_modal_layernorm:
            self.modal_ln_t = nn.LayerNorm(hidden_dim)
            self.modal_ln_a = nn.LayerNorm(hidden_dim)
            self.modal_ln_v = nn.LayerNorm(hidden_dim)

        # ============================================================
        # 第一段: 跨模态上下文增强 (仅在 awaf / awaf_no_interaction 模式下使用)
        # ============================================================
        self._use_context = fusion_mode not in ('awaf_no_context', 'mean', 'concat', 'gated', 'fixed')

        if self._use_context:
            context_hidden = max(int(hidden_dim * context_hidden_ratio), 16)
            # 对每个模态: query_proj, key_proj, value_proj (共享 key/value)
            self.context_query = nn.ModuleList([
                nn.Linear(hidden_dim, context_hidden, bias=False) for _ in range(3)
            ])
            self.context_key = nn.Linear(hidden_dim, context_hidden, bias=False)
            self.context_value = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.context_norm = nn.ModuleList([
                nn.LayerNorm(hidden_dim) for _ in range(3)
            ])

        # ============================================================
        # 第二段: 二阶交互打分
        # ============================================================
        self._use_interaction = fusion_mode not in ('awaf_no_interaction', 'mean', 'concat', 'gated', 'fixed')

        if fusion_mode == 'mean':
            pass  # 不需要任何参数
        elif fusion_mode == 'fixed':
            # 全局固定可学习权重 (log-space)
            self.global_weight_raw = nn.Parameter(torch.zeros(3))
        elif fusion_mode == 'concat':
            # 拼接 + MLP → 融合向量 (非权重加权)
            self.concat_fuse = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
        elif fusion_mode == 'gated':
            # 普通门控融合
            self.gate_proj = nn.Linear(hidden_dim * 3, 3)
            self.gate_fuse = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
        else:
            # awaf / awaf_no_context / awaf_no_interaction
            if self._use_interaction:
                # 输入: [h_t, h_a, h_v, g_ta, g_tv, g_av] = 3*hidden + 3*hidden = 6*hidden
                mlp_input_dim = hidden_dim * 6
            else:
                # 仅一阶: [h_t, h_a, h_v] = 3*hidden
                mlp_input_dim = hidden_dim * 3

            self.score_mlp = nn.Sequential(
                nn.Linear(mlp_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 3),  # 输出 3 维 → w_t, w_a, w_v
            )

    @staticmethod
    def _inv_softplus(x: float) -> float:
        """softplus 的反函数，用于初始化 tau_raw。"""
        import math
        if x <= 0:
            return -10.0
        return math.log(math.exp(x) - 1)

    @property
    def tau(self) -> torch.Tensor:
        """可学习温度 τ = softplus(tau_raw) + 0.1，保证 ≥ 0.1。"""
        return F.softplus(self.tau_raw) + 0.1

    def _context_enhance(
        self,
        h_list: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        跨模态上下文增强 (D018 修复)。

        对每个模态 m:
          c_m = Attention(q=h_m, k=other_modals, v=other_modals)  ← 排除自身
          ĥ_m = LayerNorm(h_m + c_m)
        """
        h_t, h_a, h_v = h_list  # 各 [B, d]
        B, d = h_t.shape
        h_stack = torch.stack([h_t, h_a, h_v], dim=1)  # [B, 3, d]

        # key/value: 所有模态共享
        keys = self.context_key(h_stack)    # [B, 3, d_k]
        values = self.context_value(h_stack) # [B, 3, d]

        # 构建 self-exclusion mask: [3, 3]，对角线为 -inf，其他为 0
        # 使得每个模态 query 的 attention 不包含自身
        exclude_self_mask = torch.full((3, 3), -float('inf'), device=h_stack.device)
        exclude_self_mask.fill_diagonal_(0.0)  # -inf on diagonal, 0 elsewhere
        # 实际需要的是 -inf on diagonal → softmax 后自身权重为 0
        actual_mask = torch.zeros(3, 3, device=h_stack.device)
        actual_mask.fill_diagonal_(-float('inf'))

        d_k = keys.size(-1)

        enhanced = []
        for m in range(3):
            # query: 当前模态
            q = self.context_query[m](h_stack[:, m, :])  # [B, d_k]

            # Attention(Q, K, V) — Q 仅对其他 2 个模态的 K,V
            attn_scores = torch.bmm(
                q.unsqueeze(1),       # [B, 1, d_k]
                keys.transpose(1, 2)  # [B, d_k, 3]
            ) / (d_k ** 0.5)  # [B, 1, 3]

            # 排除自身模态: 将 self-position 的 score 设为 -inf
            # actual_mask[m]: [3] with -inf at position m
            attn_scores = attn_scores + actual_mask[m].unsqueeze(0).unsqueeze(0)  # [B, 1, 3]

            attn_weights = F.softmax(attn_scores, dim=-1)  # [B, 1, 3]  (自身位置权重≈0)

            c_m = torch.bmm(attn_weights, values).squeeze(1)  # [B, d]

            # ĥ_m = LayerNorm(h_m + c_m)
            enhanced_m = self.context_norm[m](h_stack[:, m, :] + c_m)
            enhanced.append(enhanced_m)

        return enhanced[0], enhanced[1], enhanced[2]

    def _modality_dropout(
        self,
        h_list: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        模态 dropout (D017 修复): 训练时随机将某些模态置零，增强 AWAF 鲁棒性。

        **逐样本保证至少一个模态保留。**
        """
        if not self.training or not self.use_modality_dropout:
            return h_list

        h_t, h_a, h_v = [h.clone() for h in h_list]
        B = h_t.size(0)
        device = h_t.device

        # 每个模态独立 dropout: mask[b, 0] = 1 表示保留该样本的该模态
        masks = []
        for _ in range(3):
            m = torch.rand(B, 1, device=device) > self.modality_dropout_prob
            masks.append(m.float())  # [B, 1]

        # 逐样本检查: 若某样本三个 modal 都被 dropout，随机恢复一个
        masks_stacked = torch.cat(masks, dim=1)  # [B, 3]
        n_kept = masks_stacked.sum(dim=1)  # [B]

        zero_kept = (n_kept == 0)  # [B] 标记哪些样本三个模态全被 dropout

        if zero_kept.any():
            # 对全丢样本：随机恢复一个模态
            n_zero = zero_kept.sum().item()
            rescue_mod = torch.randint(0, 3, (n_zero,), device=device)  # [n_zero] 随机选模态

            for i, bidx in enumerate(torch.where(zero_kept)[0]):
                mod = rescue_mod[i].item()
                masks[mod][bidx] = 1.0

        return (
            h_t * masks[0],
            h_a * masks[1],
            h_v * masks[2],
        )

    def compute_entropy(self, weights: torch.Tensor) -> torch.Tensor:
        """计算 AWAF 权重 per-sample entropy: H = -sum(w * log(w + eps))。"""
        w = weights.clamp(min=self.eps)
        return -(w * torch.log(w)).sum(dim=-1)  # [B]

    @property
    def max_entropy(self) -> float:
        """三模态均匀分布 log(3) ≈ 1.099 — 最大可能 entropy。"""
        import math
        return math.log(3.0)

    def forward(
        self,
        h_t: torch.Tensor,
        h_a: torch.Tensor,
        h_v: torch.Tensor,
        return_weights: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            h_t: 文本模态摘要 [B, d]
            h_a: 音频模态摘要 [B, d]
            h_v: 视觉模态摘要 [B, d]
            return_weights: 是否返回权重

        Returns:
            dict:
                'Z':      融合表示 [B, d]
                'weights': 三模态权重 [B, 3] (w_t, w_a, w_v), sum=1
                'diagnostics': (if return_diagnostics) L2 norms, entropy
        """
        B, d = h_t.shape
        diagnostics = {}

        # === P6H-R: 模态输入 LayerNorm (在 dropout/context 之前) ===
        if self.use_modal_layernorm:
            l2_before_t = h_t.norm(dim=-1).mean()
            l2_before_a = h_a.norm(dim=-1).mean()
            l2_before_v = h_v.norm(dim=-1).mean()
            h_t = self.modal_ln_t(h_t)
            h_a = self.modal_ln_a(h_a)
            h_v = self.modal_ln_v(h_v)
            l2_after_t = h_t.norm(dim=-1).mean()
            l2_after_a = h_a.norm(dim=-1).mean()
            l2_after_v = h_v.norm(dim=-1).mean()
            if self.return_diagnostics:
                diagnostics['l2_before_ln'] = (float(l2_before_t.detach()), float(l2_before_a.detach()), float(l2_before_v.detach()))
                diagnostics['l2_after_ln'] = (float(l2_after_t.detach()), float(l2_after_a.detach()), float(l2_after_v.detach()))

        # ---- 模态 Dropout (仅训练时) ----
        h_t, h_a, h_v = self._modality_dropout((h_t, h_a, h_v))

        # ---- 第一段: 跨模态上下文增强 ----
        if self._use_context:
            ĥ_t, ĥ_a, ĥ_v = self._context_enhance((h_t, h_a, h_v))
        else:
            ĥ_t, ĥ_a, ĥ_v = h_t, h_a, h_v

        # ---- 第二段: 融合 ----
        if self.fusion_mode == 'mean':
            # 等权平均
            Z = (ĥ_t + ĥ_a + ĥ_v) / 3.0
            weights = torch.ones(B, 3, device=h_t.device) / 3.0

        elif self.fusion_mode == 'fixed':
            # 全局固定权重: w = softmax(w_raw)
            w = F.softmax(self.global_weight_raw, dim=-1)  # [3]
            weights = w.unsqueeze(0).expand(B, -1)  # [B, 3]
            Z = (weights[:, 0:1] * ĥ_t +
                 weights[:, 1:2] * ĥ_a +
                 weights[:, 2:3] * ĥ_v)

        elif self.fusion_mode == 'concat':
            # 拼接 + MLP (不产生独立权重)
            concat = torch.cat([ĥ_t, ĥ_a, ĥ_v], dim=-1)  # [B, 3*d]
            Z = self.concat_fuse(concat)
            # concat 模式权重不可解释，返回均匀占位
            weights = torch.ones(B, 3, device=h_t.device) / 3.0

        elif self.fusion_mode == 'gated':
            # 普通门控融合
            concat = torch.cat([ĥ_t, ĥ_a, ĥ_v], dim=-1)  # [B, 3*d]
            gates = F.softmax(self.gate_proj(concat), dim=-1)  # [B, 3]
            weighted = torch.cat([
                ĥ_t * gates[:, 0:1],
                ĥ_a * gates[:, 1:2],
                ĥ_v * gates[:, 2:3],
            ], dim=-1)  # [B, 3*d]
            Z = self.gate_fuse(weighted)
            weights = gates  # gated 权重也可视为模态贡献

        else:
            # awaf / awaf_no_context / awaf_no_interaction
            if self._use_interaction:
                # 二阶 Hadamard 交互项
                g_ta = ĥ_t * ĥ_a  # [B, d]
                g_tv = ĥ_t * ĥ_v  # [B, d]
                g_av = ĥ_a * ĥ_v  # [B, d]
                mlp_input = torch.cat([ĥ_t, ĥ_a, ĥ_v, g_ta, g_tv, g_av], dim=-1)  # [B, 6*d]
            else:
                mlp_input = torch.cat([ĥ_t, ĥ_a, ĥ_v], dim=-1)  # [B, 3*d]

            # MLP → e ∈ R^3
            e = self.score_mlp(mlp_input)  # [B, 3]

            # w = softmax(e / τ)
            tau_val = self.tau.clamp(min=0.1)
            logits = e / tau_val  # [B, 3]
            weights_raw = F.softmax(logits, dim=-1)  # [B, 3]

            # === P6H-R: Uniform mix (weight floor) ===
            if self.awaf_uniform_mix > 0.0:
                eps_mix = self.awaf_uniform_mix
                weights = (1.0 - eps_mix) * weights_raw + eps_mix / 3.0
            else:
                weights = weights_raw

            # === P6H-R: Entropy 诊断 ===
            if self.return_diagnostics:
                diagnostics['entropy'] = self.compute_entropy(weights).mean().detach()
                diagnostics['entropy_raw'] = self.compute_entropy(weights_raw).mean().detach()
                diagnostics['tau'] = tau_val.detach()

            # Z = w_t·ĥ_t + w_a·ĥ_a + w_v·ĥ_v
            Z = (weights[:, 0:1] * ĥ_t +
                 weights[:, 1:2] * ĥ_a +
                 weights[:, 2:3] * ĥ_v)  # [B, d]

        result = {'Z': Z}
        if return_weights:
            result['weights'] = weights
        if self.return_diagnostics and diagnostics:
            result['diagnostics'] = diagnostics
        return result


def save_awaf_weights_csv(
    weights: torch.Tensor,
    ids: list,
    filepath: str,
    modality_names: tuple = ('text', 'audio', 'vision'),
):
    """
    将 AWAF 权重保存为 CSV 文件。

    Args:
        weights: [N, 3] 三模态权重 (numpy 或 torch)
        ids:     样本 ID 列表
        filepath: 保存路径
        modality_names: 模态名称
    """
    if isinstance(weights, torch.Tensor):
        weights = weights.detach().cpu().numpy()
    weights = np.asarray(weights)

    import csv
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['sample_id', f'w_{modality_names[0]}',
                        f'w_{modality_names[1]}', f'w_{modality_names[2]}'])
        for i, sid in enumerate(ids):
            if i < len(weights):
                writer.writerow([sid, weights[i, 0], weights[i, 1], weights[i, 2]])


# ============================================================
# 单元验证
# ============================================================
if __name__ == '__main__':
    print("=== AWAF 随机张量 forward test ===\n")

    B, d = 4, 256
    h_t = torch.randn(B, d)
    h_a = torch.randn(B, d)
    h_v = torch.randn(B, d)

    modes = [
        'awaf', 'awaf_no_context', 'awaf_no_interaction',
        'mean', 'concat', 'gated', 'fixed',
    ]

    for mode in modes:
        awaf = AdaptiveWeightedAttentionFusion(
            hidden_dim=d,
            fusion_mode=mode,
            tau_init=1.0,
            dropout=0.1,
            use_modality_dropout=(mode == 'awaf'),
        )
        awaf.train()  # 测试 modality_dropout
        out = awaf(h_t, h_a, h_v)
        Z = out['Z']
        w = out['weights']

        print(f"  [{mode:25s}] Z: {list(Z.shape)}, weights: {list(w.shape)}", end="")

        # 验证
        if mode not in ('concat',):
            w_sum = w.sum(dim=-1)
            max_dev = (w_sum - 1.0).abs().max().item()
            print(f"  sum(w) max_dev: {max_dev:.2e}", end="")
            assert max_dev < 1e-4, f"sum(w) should be ~1.0, got {w_sum}"
        else:
            print(f"  (concat: no weight sum check)", end="")

        print("  ✅")

    # 测试 eval 模式 (modality_dropout 关闭)
    print("\n  测试 eval 模式 (modality_dropout off):")
    awaf_eval = AdaptiveWeightedAttentionFusion(d, fusion_mode='awaf')
    awaf_eval.eval()
    out_eval = awaf_eval(h_t, h_a, h_v)
    print(f"    weights: {out_eval['weights'][0].tolist()}")

    # 测试 save function
    import tempfile
    import os
    weights_np = out_eval['weights'].detach().cpu().numpy()
    tmp_path = os.path.join(tempfile.gettempdir(), 'awaf_test.csv')
    save_awaf_weights_csv(weights_np, ['s0', 's1', 's2', 's3'], tmp_path)
    print(f"    CSV saved to {tmp_path}")
    os.remove(tmp_path)

    print("\n=== AWAF forward test 全部通过 ===")
