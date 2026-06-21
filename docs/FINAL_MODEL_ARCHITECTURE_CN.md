# 最终模型架构说明（中文版）

**日期**: 2026-06-21
**版本**: v1.0.0-final
**仓库**: `ailiwood/mme_0621`

---

## 一、项目概述

本项目实现了两种多模态情感分析架构，分别用于 CMU-MOSEI 和 CMU-MOSI 数据集。两种架构的选择由**数据集规模决定**：MOSEI（13,239 训练样本）足以支持等权融合架构的学习；MOSI（1,284 训练样本）需要文本保持架构来防止塌缩。

| 数据集 | 架构名称 | 融合方式 | 状态 |
|--------|----------|----------|------|
| **MOSEI** | Canonical AWAF sLSTM | 自适应加权注意力融合 | 主要三模态证据 |
| **MOSI** | Text-Anchored Reliable Fusion | 文本锚点 + 门控修正 | 最终三模态候选 |

---

## 二、MOSEI：Canonical AWAF sLSTM（自适应加权注意力融合）

### 2.1 架构总览

```
输入层：
  Text:  原始文本 → RoBERTa-large + LoRA(r=16) → TextMLP(512→256) → h_t  [B, 256]
  Audio: COVAREP 74维 × 100帧 → Linear(74→256) → sLSTM(1层) → MaskedAttentionPool → h_a  [B, 256]
  Vision: OpenFace2 713维 × 50帧 → Linear(713→256) → sLSTM(1层) → MaskedAttentionPool → h_v  [B, 256]

融合层：AdaptiveWeightedAttentionFusion (AWAF)
  第一阶段：跨模态上下文增强 (Context Enhancement)
    对每个模态 h_m，以 h_m 为 query，其他模态为 key/value，得到上下文向量 c_m。
    增强表示：ĥ_m = LayerNorm(h_m + c_m)

  第二阶段：二阶交互打分 (Interaction Scoring)
    g_ta = ĥ_t ⊙ ĥ_a     # 文本-音频 Hadamard 交互
    g_tv = ĥ_t ⊙ ĥ_v     # 文本-视觉 Hadamard 交互
    g_av = ĥ_a ⊙ ĥ_v     # 音频-视觉 Hadamard 交互
    e = MLP([ĥ_t, ĥ_a, ĥ_v, g_ta, g_tv, g_av]) ∈ R³   # 三模态打分
    [w_t, w_a, w_v] = softmax(e / τ)                    # 样本级三模态权重

  第三阶段：加权融合
    z = w_t·ĥ_t + w_a·ĥ_a + w_v·ĥ_v    # 加权和，权重和为1

预测层：
  y_hat = Linear(256→128) → ReLU → Linear(128→1)    # 情感回归 [-3, +3]
```

### 2.2 关键设计决策

1. **sLSTM 时序编码器**：替换普通 LSTM/GRU。sLSTM (Scalar LSTM) 使用标量记忆单元和指数门控，在小批量训练中更稳定。

2. **Hadamard 交互项**：g_ta、g_tv、g_av 是 AWAF 最关键的设计。它们捕捉二阶跨模态关系（例如：文本的"happy"与音频的"高音调"同时出现时，交互项会产生强信号）。

3. **样本级动态权重**：每个样本都有独立的三模态权重 [w_t, w_a, w_v]，由 MLP 根据该样本的模态内容打分生成。

4. **温度参数 τ**：初始化为 3.0，控制 softmax 的锐度。较大的 τ 使权重更均匀（训练初期探索），随着训练 τ 可学习调整。

### 2.3 模型参数

| 组件 | 参数量 |
|------|--------|
| RoBERTa-large (冻结) | ~355M |
| LoRA (r=16, query+value) | ~0.6M |
| 音频投影 + sLSTM + Pool | ~0.3M |
| 视觉投影 + sLSTM + Pool | ~0.5M |
| AWAF 融合模块 | ~1.2M |
| 预测头 | ~0.1M |
| **可训练参数总计** | **~4.84M** |

### 2.4 相关代码文件

| 文件 | 内容 |
|------|------|
| `models/fusion/awaf.py` | AWAF 融合模块完整实现 |
| `models/encoders/slstm.py` | sLSTM 时序编码器 |
| `models/pooling/attention_pooling.py` | 掩码注意力池化 |
| `models/modules/minimal_lora.py` | LoRA 轻量适配器 |
| `models/textft_lora_xlstm_awaf_residual.py` | 主模型（mode: `canonical_text_audio_vision_awaf_slstm`）|

---

## 三、MOSI：Text-Anchored Reliable Fusion（文本锚定可靠性融合）

### 3.1 架构总览

```
输入层：
  Text:  原始文本 → RoBERTa-large + LoRA(r=16) → TextMLP(512→256) → h_t  [B, 256]
  Audio: data2vec 768维 × 100帧 → Linear(768→256) → sLSTM(1层) → MaskedAttentionPool → h_a  [B, 256]
  Vision: CLIP-L14 1024维 × 32帧 → Linear(1024→256) → sLSTM(1层) → MaskedAttentionPool → h_v  [B, 256]

融合层：TextAnchoredReliableFusion（文本锚定可靠性融合）
  
  第一步：文本锚点预测（不可移除）
    y_text = TextHead(h_t)                              # 文本独立预测 [B, 1]
    TextHead = Linear(256→128) → LayerNorm → GELU → Dropout → Linear(128→1)

  第二步：文本锚定的跨模态交互（仅文本-辅助模态交互）
    g_ta = h_t ⊙ h_a                                    # 文本-音频 Hadamard 交互 [B, 256]
    g_tv = h_t ⊙ h_v                                    # 文本-视觉 Hadamard 交互 [B, 256]
    注意：不计算 g_av（音频-视觉交互），所有交互以文本为锚点

  第三步：辅助模态修正量计算
    audio_input  = [h_t, h_a, g_ta]                     # 拼接 [B, 768]
    vision_input = [h_t, h_v, g_tv]                     # 拼接 [B, 768]
    
    delta_a = MLP_a(audio_input)                        # 音频修正量 [B, 1]
    delta_v = MLP_v(vision_input)                       # 视觉修正量 [B, 1]
    
    MLP 结构：Linear(768→128) → LN → GELU → Drop → Linear(128→64) → GELU → Drop → Linear(64→1)

  第四步：样本级可靠性门控
    r_a_logit = GateMLP_a(audio_input) + b_a            # 音频可靠性 logit [B, 1]
    r_v_logit = GateMLP_v(vision_input) + b_v            # 视觉可靠性 logit [B, 1]
    r_a = sigmoid(r_a_logit)                             # 音频可靠性门 [B, 1]，范围 (0, 1)
    r_v = sigmoid(r_v_logit)                             # 视觉可靠性门 [B, 1]，范围 (0, 1)
    
    GateMLP 结构：Linear(768→64) → LN → GELU → Drop → Linear(64→1)
    初始偏置 b_a = b_v = -2.0 → sigmoid(-2.0) ≈ 0.12
    
  第五步：有界最终预测
    α_a = max_alpha × sigmoid(raw_alpha_a)              # 音频缩放因子（有界）
    α_v = max_alpha × sigmoid(raw_alpha_v)              # 视觉缩放因子（有界）
    
    contribution_a = α_a × r_a × delta_a                # 音频总贡献
    contribution_v = α_v × r_v × delta_v                # 视觉总贡献
    
    y_hat = clamp(y_text + contribution_a + contribution_v, -3.0, +3.0)
```

### 3.2 关键设计决策

1. **文本锚点不可移除**：y_text 始终直接贡献到最终预测。不同于 AWAF 的 softmax 加权（所有模态权重和为1，必须竞争），文本锚点提供"地板"性能保证。

2. **可靠性门初始化接近零**：初始偏置 b=-2.0 确保训练开始时 r_a ≈ 0.12, r_v ≈ 0.12。模型初始行为接近纯文本预测，随着训练逐步学会信任有用的辅助模态修正。

3. **文本锚定的交互**：只计算 g_ta 和 g_tv（文本与各辅助模态的交互），不计算 g_av（音频-视觉交互）。这基于诊断发现：MOSI 上音频和视觉单独均无有效信号（均低于多数类基线），交叉交互只会引入更多噪声。

4. **有界修正**：delta 通过 MLP 输出但不加约束，但 contribution = α × r × delta 天然被 r∈(0,1) 和 α∈(0,1) 限制。同时 y_hat clamp 到 [-3, +3] 防止异常值。

5. **P6K 权重初始化**：文本分支和音频分支使用 P6K text+audio conservative checkpoint 初始化（在 MOSI v3_T40 特征上训练，ACC2=88.72%）。视觉分支和融合模块随机初始化。

### 3.3 为什么 Canonical AWAF 在 MOSI 上塌缩

Canonical AWAF 使用 softmax 强制三模态权重求和为 1：
```
[w_t, w_a, w_v] = softmax(scorer(h_t, h_a, h_v) / τ)
z = w_t·h_t + w_a·h_a + w_v·h_v
```

在 MOSI 的 1,284 训练样本上：
- 文本单模态可达 86.11%（P6AF-G1 探针验证）
- 音频单模态仅 56.94%（低于多数类基线 59.8%）
- 视觉单模态仅 57.41%（低于多数类基线）

Softmax 强制模型为三个模态分配权重。即使 AWAF 想给文本高权重（如 0.8），仍需给音频和视觉各分配约 0.1。在小样本下，弱模态的噪声通过加权和污染了融合输出，导致最终预测（42–44%）远低于纯文本（86%）。

**Text-Anchored Fusion 的解决方案**：
```
y_hat = y_text + α_a·r_a·delta_a + α_v·r_v·delta_v
```
- y_text 始终 100% 贡献（锚点不移除）
- 辅助模态只能通过门控添加修正
- 如果辅助模态不可靠，r_a、r_v 可以学到接近 0 → 模型自动退化为文本单模态

### 3.4 模型参数

| 组件 | 参数量 |
|------|--------|
| RoBERTa-large (冻结) | ~355M |
| LoRA (r=16, query+value) | ~0.6M |
| TextMLP + TextHead | ~0.2M |
| 音频投影 + sLSTM + Pool | ~0.8M |
| 视觉投影 + sLSTM + Pool | ~0.5M |
| 音频修正 MLP + 门控 MLP | ~0.6M |
| 视觉修正 MLP + 门控 MLP | ~0.6M |
| 缩放因子等 | ~0.05M |
| **可训练参数总计** | **~4.16M** |

### 3.5 相关代码文件

| 文件 | 内容 |
|------|------|
| `models/fusion/text_anchored_reliable_fusion.py` | 文本锚定融合模块完整实现 |
| `models/encoders/slstm.py` | sLSTM 时序编码器 |
| `models/pooling/attention_pooling.py` | 掩码注意力池化 |
| `models/modules/minimal_lora.py` | LoRA 轻量适配器 |
| `models/textft_lora_xlstm_awaf_residual.py` | 主模型（mode: `canonical_mosi_text_anchored_tav`）|

---

## 四、共享组件

两个数据集共享以下基础组件：

| 组件 | 文件 | 用途 |
|------|------|------|
| RoBERTa + LoRA | `models/modules/minimal_lora.py` | 文本特征提取 |
| sLSTM 编码器 | `models/encoders/slstm.py` | 时序建模（音频+视觉） |
| 掩码注意力池化 | `models/pooling/attention_pooling.py` | 变长序列池化 |
| 多模态数据集 | `data/textft_multimodal_dataset.py` | 数据加载与预处理 |
| 评估指标 | `utils/metrics.py` | ACC2, F1, MAE, Corr, ACC7 |

---

## 五、特征版本差异

两个数据集使用了**不同来源**的特征，必须在论文中明确标注：

| 模态 | MOSI | MOSEI |
|------|------|-------|
| 视觉特征源 | CLIP-L14 | OpenFace2 |
| 视觉维度 | 1024 | 713 |
| 视觉帧数 | 32 | 50 |
| 音频特征源 | data2vec | COVAREP |
| 音频维度 | 768 | 74 |
| 音频帧数 | 100 | 100 |
| 文本处理 | RoBERTa-large（相同） | RoBERTa-large（相同） |

**重要说明**：
- MOSI 使用现代 Transformer 特征（CLIP-L14 1024d 视觉 + data2vec 768d 音频）
- MOSEI 使用传统手工特征（OpenFace2 713d 视觉 + COVAREP 74d 音频）
- 两者不可直接进行数值比较
- 特征版本的差异必须在论文中明确说明
