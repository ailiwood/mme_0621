"""
models/encoders/slstm.py — sLSTM (Scalar LSTM) 时序编码器

纯 PyTorch 自实现，基于 Beck et al. (2024) xLSTM 论文公式推导。
不复制 NX-AI / AGPL-3.0 代码。

核心特征 (区别于标准 LSTM):
  1. 指数输入门:  i_t = exp(˜i_t)    (log-space 稳定计算)
  2. 指数遗忘门:  f_t = exp(˜f_t)    或 σ(˜f_t) 可选
  3. 归一化状态:  n_t = f_t · n_{t-1} + i_t   (与 c_t 同步累积)
  4. 稳定状态:    m_t = max(˜f_t + m_{t-1}, ˜i_t)   (防数值溢出)
  5. 隐藏输出:    h_t = o_t · (c_t / n_t)
  6. 输出门:      o_t = σ(˜o_t)

References:
  Beck, M., et al. (2024). xLSTM: Extended Long Short-Term Memory. NeurIPS.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class SLSTMCell(nn.Module):
    """
    sLSTM 单元（标量 LSTM）。

    输入 x_t: [B, D]
    输出 h_t: [B, H]
    内部状态: c_t [B, H], n_t [B, H], m_t [B, H]  (scalar per hidden unit)

    数值稳定策略:
      - 在 log-space 维护 stabilizer state m_t
      - 指数门在减去 m_t 后计算，保证 exp(arg) ≤ 1.0
      - NaN/Inf 保护: clamp pre-activation 范围
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        use_exp_forget: bool = True,   # True = exp forget gate, False = sigmoid forget gate
        clamp_preact: float = 20.0,    # clamp pre-activation 到 [-20, 20]
        eps: float = 1e-8,             # 防止除零
    ):
        """
        Args:
            input_dim: 输入特征维度 D
            hidden_dim: 隐藏状态维度 H
            use_exp_forget: 遗忘门是否用指数激活 (False → sigmoid)
            clamp_preact: 预激活值 clamp 范围，防止 exp 溢出
            eps: 小常数
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_exp_forget = use_exp_forget
        self.clamp_preact = clamp_preact
        self.eps = eps

        # 输入→门的线性投影 (4 gates: i, f, z, o)
        self.W = nn.Linear(input_dim, 4 * hidden_dim, bias=False)
        # 隐藏→门的循环投影
        self.R = nn.Linear(hidden_dim, 4 * hidden_dim, bias=False)
        # 偏置 (仅对 pre-activation)
        self.bias = nn.Parameter(torch.zeros(4 * hidden_dim))

        # 初始化
        self._reset_parameters()

    def _reset_parameters(self):
        """小值初始化，遗忘门偏置初始化为正以鼓励记忆。"""
        nn.init.xavier_uniform_(self.W.weight, gain=0.5)
        nn.init.orthogonal_(self.R.weight, gain=0.5)

        # 偏置初始化: i/z/o 小随机, f 偏正向 (鼓励保留信息)
        with torch.no_grad():
            self.bias.zero_()
            # 遗忘门偏置初始为 2.0~4.0 (sigmoid 区间偏高，exp 区间适中)
            f_bias_slot = self.bias.view(4, self.hidden_dim)[1]  # f gate = index 1
            nn.init.uniform_(f_bias_slot, 2.0, 4.0)

    def _split_gates(self, preact: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        拆分 4 个门的预激活值。

        Args:
            preact: [B, 4*H]

        Returns:
            ˜i, ˜f, ˜z, ˜o  each [B, H]
        """
        chunks = preact.chunk(4, dim=-1)  # 4 × [B, H]
        return chunks[0], chunks[1], chunks[2], chunks[3]

    def forward(
        self,
        x_t: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        单步 forward。

        Args:
            x_t:      当前输入 [B, D]
            state:    (h_{t-1}, c_{t-1}, n_{t-1}, m_{t-1}) 各 [B, H]
                      若为 None，初始化为 0

        Returns:
            h_t:      当前隐藏 [B, H]
            new_state: (h_t, c_t, n_t, m_t)
        """
        B = x_t.size(0)
        H = self.hidden_dim
        device = x_t.device

        # --- 初始化或拆解状态 ---
        if state is None:
            h_prev = torch.zeros(B, H, device=device, dtype=x_t.dtype)
            c_prev = torch.zeros(B, H, device=device, dtype=x_t.dtype)
            n_prev = torch.zeros(B, H, device=device, dtype=x_t.dtype)
            m_prev = torch.full((B, H), -float('inf'), device=device, dtype=x_t.dtype)
        else:
            h_prev, c_prev, n_prev, m_prev = state

        # --- 计算预激活值 ---
        # gate_preact = W(x_t) + R(h_{t-1}) + bias   →  [B, 4*H]
        gate_preact = self.W(x_t) + self.R(h_prev) + self.bias

        # clamp 防止 exp 溢出
        gate_preact = torch.clamp(gate_preact, -self.clamp_preact, self.clamp_preact)

        i_preact, f_preact, z_preact, o_preact = self._split_gates(gate_preact)
        # 各 [B, H]

        # --- Stabilizer: m_t = max(˜f_t + m_{t-1}, ˜i_t) ---
        # 处理 m_prev = -inf 的初始情况
        m_candidate = f_preact + m_prev  # [B, H]
        m_t = torch.maximum(m_candidate, i_preact)  # [B, H]

        # --- 计算稳定化的指数门 ---
        # i_t = exp(˜i_t - m_t)          ← ≤ 1.0
        # f_t = exp(˜f_t + m_{t-1} - m_t) ← ≤ 1.0
        i_t = torch.exp(i_preact - m_t)  # [B, H]

        if self.use_exp_forget:
            # 指数遗忘门: f_t = σ(˜f_t) ... 不对，是指数!
            # Actually: f_t = exp(˜f_t) 然后通过 m_t stabilizer
            f_t = torch.exp(m_candidate - m_t)  # [B, H]  ← stabilized exp forget gate
        else:
            # Sigmoid 遗忘门 (数值不需 stabilizer)
            f_t = torch.sigmoid(f_preact)  # [B, H]

        # --- 输入变换 ---
        z_t = torch.tanh(z_preact)  # [B, H]

        # --- 输出门 ---
        o_t = torch.sigmoid(o_preact)  # [B, H]

        # --- 状态更新 ---
        # c_t = f_t * c_{t-1} + i_t * z_t
        c_t = f_t * c_prev + i_t * z_t  # [B, H]

        # n_t = f_t * n_{t-1} + i_t  (normalizer)
        n_t = f_t * n_prev + i_t  # [B, H]

        # --- 隐藏输出 ---
        # h_t = o_t * (c_t / max(n_t, eps))
        h_t = o_t * (c_t / torch.clamp(n_t, min=self.eps))  # [B, H]

        # --- NaN/Inf 防护 ---
        if torch.isnan(h_t).any() or torch.isinf(h_t).any():
            # 回退到安全值
            h_t = torch.where(
                torch.isnan(h_t) | torch.isinf(h_t),
                h_prev,  # 用上一步的 h 替代
                h_t
            )
            c_t = torch.where(
                torch.isnan(c_t) | torch.isinf(c_t),
                c_prev,
                c_t
            )
            n_t = torch.where(
                torch.isnan(n_t) | torch.isinf(n_t),
                n_prev,
                n_t
            )

        return h_t, (h_t, c_t, n_t, m_t)


class SLSTMEncoder(nn.Module):
    """
    sLSTM 时序编码器。

    输入:  x [B, T, D], mask [B, T]
    输出:  H [B, T, H]  (所有时间步隐藏)
           pooled [B, H] (池化后摘要向量)

    支持多种池化: masked_mean / last_valid
    支持 T=1 (单向量输入)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
        pooling: str = 'masked_mean',  # 'masked_mean' | 'last_valid'
        use_exp_forget: bool = True,
    ):
        """
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏维度
            num_layers: sLSTM 层数
            dropout: 层间 dropout (仅 num_layers > 1 时有效)
            bidirectional: 是否双向 (若 True，输出维度 ×2)
            pooling: 池化策略
            use_exp_forget: 遗忘门使用指数激活
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.pooling = pooling
        self.output_dim = hidden_dim * (2 if bidirectional else 1)

        # 多层 sLSTM cells
        self.cells = nn.ModuleList()
        for layer in range(num_layers):
            in_dim = input_dim if layer == 0 else hidden_dim
            self.cells.append(SLSTMCell(in_dim, hidden_dim, use_exp_forget=use_exp_forget))

        # 可选双向反向 cell
        if bidirectional:
            self.cells_rev = nn.ModuleList()
            for layer in range(num_layers):
                in_dim = input_dim if layer == 0 else hidden_dim
                self.cells_rev.append(SLSTMCell(in_dim, hidden_dim, use_exp_forget=use_exp_forget))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def _unroll(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        cells: nn.ModuleList,
        reverse: bool = False,
    ) -> torch.Tensor:
        """
        沿时间维度展开 sLSTM。

        Args:
            x:     [B, T, D]
            mask:  [B, T]  (1=有效, 0=padding)
            cells: sLSTM 层列表
            reverse: 是否反向 (双向编码用)

        Returns:
            H: [B, T, H]
        """
        B, T, _ = x.shape
        H = self.hidden_dim
        device = x.device

        if reverse:
            x = x.flip(dims=[1])
            if mask is not None:
                mask = mask.flip(dims=[1])

        # 逐层处理
        layer_input = x
        for cell in cells:
            outputs = []
            state = None
            h_prev_cp = c_prev_cp = n_prev_cp = m_prev_cp = None

            for t in range(T):
                x_t = layer_input[:, t, :]  # [B, D]

                if mask is not None:
                    is_valid = mask[:, t].bool()  # [B]
                else:
                    is_valid = torch.ones(B, dtype=torch.bool, device=device)

                # 保存上一时刻状态（用于 mask=0 时恢复）
                if mask is not None and state is not None:
                    h_prev_cp, c_prev_cp, n_prev_cp, m_prev_cp = state

                h_t, state = cell(x_t, state)

                if mask is not None:
                    # --- mask 状态冻结 (D017) ---
                    # 对 padding 样本：恢复上一时刻完整状态 (h/c/n/m)
                    # 对 padding 样本：h_t 置零
                    invalid_mask = ~is_valid  # [B]

                    if invalid_mask.any() and state is not None and h_prev_cp is not None:
                        # 展开当前 state
                        h_new, c_new, n_new, m_new = state
                        h_prev_local, c_prev_local, n_prev_local, m_prev_local = (
                            h_prev_cp, c_prev_cp, n_prev_cp, m_prev_cp
                        )

                        # 逐样本替换：padding 样本用上一时刻状态，有效样本用新状态
                        inv_float = invalid_mask.float()
                        # h: [B, H]
                        h_new_fixed = (
                            h_new * is_valid.float().unsqueeze(-1) +
                            h_prev_local * inv_float.unsqueeze(-1)
                        )
                        # c/n/m 同理
                        c_new_fixed = (
                            c_new * is_valid.float().unsqueeze(-1) +
                            c_prev_local * inv_float.unsqueeze(-1)
                        )
                        n_new_fixed = (
                            n_new * is_valid.float().unsqueeze(-1) +
                            n_prev_local * inv_float.unsqueeze(-1)
                        )
                        m_new_fixed = torch.where(
                            is_valid.unsqueeze(-1),
                            m_new,
                            m_prev_local
                        )

                        h_t = h_new_fixed
                        state = (h_new_fixed, c_new_fixed, n_new_fixed, m_new_fixed)
                    else:
                        # 无 mask 的简单情况
                        h_t = h_t * is_valid.float().unsqueeze(-1)

                outputs.append(h_t)

            # [T, B, H] → [B, T, H]
            layer_output = torch.stack(outputs, dim=1)

            # 层间 dropout
            if cell != cells[-1]:
                layer_output = self.dropout(layer_output)

            layer_input = layer_output

        if reverse:
            layer_input = layer_input.flip(dims=[1])

        return layer_input

    def _pool(self, H: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        从 [B, T, H] 池化到 [B, H]。

        Args:
            H:    [B, T, H]
            mask: [B, T] (1=有效)

        Returns:
            pooled: [B, H]
        """
        if self.pooling == 'masked_mean':
            if mask is not None:
                # 加权平均：只对有效位置
                mask_expanded = mask.float().unsqueeze(-1)  # [B, T, 1]
                H_masked = H * mask_expanded
                pooled = H_masked.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            else:
                pooled = H.mean(dim=1)
        elif self.pooling == 'last_valid':
            if mask is not None:
                # 找每个样本最后一个有效位置
                lengths = mask.sum(dim=1).long() - 1  # [B] 0-indexed
                lengths = torch.clamp(lengths, min=0)
                pooled = H[torch.arange(H.size(0), device=H.device), lengths, :]
            else:
                pooled = H[:, -1, :]
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        return pooled

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_all: bool = True,
    ):
        """
        Args:
            x:          [B, T, D] 输入特征
            mask:       [B, T] mask (1=有效, 0=padding)，可选
            return_all: 是否返回全部时间步

        Returns:
            dict:
                'H':       [B, T, hidden_dim] 全部时间步 (return_all=True 时)
                'pooled':  [B, hidden_dim] 池化后摘要
        """
        B, T, D_in = x.shape

        # 前向
        H_fwd = self._unroll(x, mask, self.cells, reverse=False)  # [B, T, H]

        if self.bidirectional:
            H_rev = self._unroll(x, mask, self.cells_rev, reverse=True)  # [B, T, H]
            H_all = torch.cat([H_fwd, H_rev], dim=-1)  # [B, T, 2*H]
        else:
            H_all = H_fwd

        # 池化
        pooled = self._pool(H_all, mask)  # [B, output_dim]

        result = {'pooled': pooled}
        if return_all:
            result['H'] = H_all
        return result


# ============================================================
# 随机张量 forward test
# ============================================================
if __name__ == '__main__':
    print("=== sLSTM 随机张量 forward test ===\n")

    B, T, D, H = 2, 5, 128, 256
    device = 'cpu'

    # --- Test SLSTMCell ---
    print("1. SLSTMCell 单步测试")
    cell = SLSTMCell(D, H)
    x_t = torch.randn(B, D)
    h_t, state = cell(x_t)
    print(f"   Input:  {x_t.shape} → Hidden: {h_t.shape}")
    print(f"   h_t range: [{h_t.min().item():.4f}, {h_t.max().item():.4f}]")
    assert not torch.isnan(h_t).any(), "NaN in h_t!"
    print("   ✅ 通过\n")

    # --- Test SLSTMEncoder ---
    print("2. SLSTMEncoder 序列测试")
    encoder = SLSTMEncoder(D, H, num_layers=2, dropout=0.1, pooling='masked_mean')
    x = torch.randn(B, T, D)
    mask = torch.ones(B, T)
    mask[0, -2:] = 0  # 最后2步为 padding

    out = encoder(x, mask)
    print(f"   Input:  {x.shape}")
    print(f"   H:      {out['H'].shape}")
    print(f"   pooled: {out['pooled'].shape}")
    assert out['H'].shape == (B, T, H), f"Expected H shape {(B,T,H)}, got {out['H'].shape}"
    assert out['pooled'].shape == (B, H)
    assert not torch.isnan(out['pooled']).any(), "NaN in pooled!"
    print("   ✅ 通过\n")

    # --- Test T=1 ---
    print("3. T=1 兼容性测试")
    x_t1 = torch.randn(B, 1, D)
    mask_t1 = torch.ones(B, 1)
    out_t1 = encoder(x_t1, mask_t1)
    print(f"   Input:  {x_t1.shape}")
    print(f"   pooled: {out_t1['pooled'].shape}")
    assert out_t1['pooled'].shape == (B, H)
    print("   ✅ 通过\n")

    # --- Test bidirectional ---
    print("4. 双向 sLSTM 测试")
    encoder_bi = SLSTMEncoder(D, H, num_layers=1, bidirectional=True)
    out_bi = encoder_bi(x, mask)
    print(f"   pooled: {out_bi['pooled'].shape} (期望 {B}, {H*2})")
    assert out_bi['pooled'].shape == (B, H * 2)
    print("   ✅ 通过\n")

    # --- Test last_valid pooling ---
    print("5. last_valid pooling 测试")
    encoder_lv = SLSTMEncoder(D, H, num_layers=1, pooling='last_valid')
    out_lv = encoder_lv(x, mask)
    # mask[0] = [1,1,1,0,0], last_valid = index 2
    print(f"   pooled: {out_lv['pooled'].shape}")
    print("   ✅ 通过\n")

    # --- 参数统计 ---
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"6. 参数量: {n_params:,}")
    print(f"   Cell: {sum(p.numel() for p in cell.parameters()):,}")

    print("\n=== sLSTM forward test 全部通过 ===")
