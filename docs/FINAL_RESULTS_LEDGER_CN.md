# 最终实验结果总账（中文版）

**日期**: 2026-06-21
**版本**: v1.0.0-final
**仓库**: `ailiwood/mme_0621`

---

## MOSI — 最终实验结果

### 主模型：Text-Anchored Reliable TAV Fusion（3-seed 冻结）

| Seed | ACC2_Non0 | F1_Non0 | MAE | Corr | ACC7 | 状态 |
|------|-----------|---------|-----|------|------|------|
| 42 | 87.65% | 85.03% | 0.668 | 0.837 | 48.10% | ✅ |
| 2024 | 87.50% | 85.51% | 0.699 | 0.816 | 45.48% | ✅ |
| 3407 | 88.26% | 86.03% | 0.644 | 0.852 | 48.54% | ✅ |
| **均值 ± 标准差** | **87.80% ± 0.33** | **85.52% ± 0.41** | **0.670 ± 0.022** | **0.835 ± 0.015** | **47.37% ± 1.34** | — |

- **架构**: Text-Anchored Reliable Fusion (`canonical_mosi_text_anchored_tav`)
- **初始化**: P6K text+audio checkpoint (seed 42)
- **特征**: `mosi_tav_v1` (data2vec 768d 音频 + CLIP-L14 1024d 视觉)
- **参数**: 4.16M 可训练（总计 ~359.5M，RoBERTa 参数冻结）
- **训练协议**: max 35 epoch, early stop patience 8, min 10 epoch
- **状态**: `final_candidate` — 三 seed 冻结门通过，可进入论文主表
- **对应配置**: `configs/experiments/p6aj_mosi_87_recovery/P1_p6k_init_tav_s{seed}.yaml`
- **对应输出**: `outputs/P6AJ_mosi_87_recovery/P1_p6k_init_tav_s{seed}_*/`
- **重新评估诊断**: `outputs/P6AK_mosi_final_ablation/re_evaluated_s{seed}/`

### 消融矩阵（5 变体 × 3 seeds, 35 epoch 协议）

| 变体 | 描述 | ACC2 均值 | Δ vs F0 | 状态 |
|------|------|-----------|---------|------|
| **A1** | 纯文本（无音频/视觉） | **88.11% ± 0.55** | **+0.31** | paper_ready |
| F0 | 完整 TAV（文本+音频+视觉） | 87.80% ± 0.33 | — | final_candidate |
| A4 | 无可靠性门（r_a=r_v=1） | 87.75% ± 0.56 | -0.05 | paper_ready |
| A5 | 无 Hadamard 交互 | 87.70% ± 0.26 | -0.10 | paper_ready |
| A2 | 无音频修正 | 87.55% ± 0.29 | -0.25 | paper_ready |
| A3 | 无视觉修正 | 87.55% ± 0.29 | -0.25 | paper_ready |

- **消融限制**: 所有效应在 seed 波动范围内（F0 std=0.33pp）。辅助模态贡献边际。
- **对应配置**: `configs/experiments/p6ak_mosi_final_ablation/A{1-5}_*_s{seed}.yaml`
- **对应输出**: `outputs/P6AK_mosi_final_ablation/A{1-5}_*/`

### 历史参考

| 模型 | ACC2 | 状态 | 说明 |
|------|------|------|------|
| P6K T+A conservative | 88.72% | `historical_reference` | 不同特征版本 (v3_T40)，仅 T+A，非 TAV，非 AWAF |
| P6AD Canonical AWAF | 42.23% | `collapsed_evidence` | softmax 塌缩，保留为诊断证据 |
| P6AF Route A Canonical AWAF | 44.21% | `collapsed_evidence` | 塌缩复现，保留为诊断证据 |

---

## MOSEI — 最终实验结果

### 主模型：Canonical AWAF sLSTM（12-epoch, seed 42）

| 指标 | 数值 | 状态 |
|------|------|------|
| ACC2_Non0 | 87.98% | `aligned_or_mixed_tav_reference` |
| F1_Non0 | 90.49% | — |
| MAE | 0.533 | — |
| Corr | 0.785 | — |
| ACC7 | 53.92% | — |
| 可训练参数 | 4.84M | — |
| 最佳 Epoch | 11 (val ACC2 = 87.72%) | — |

- **架构**: Canonical AWAF sLSTM (`canonical_text_audio_vision_awaf_slstm`)
- **AWAF 权重**: w_t=0.481, w_a=0.264, w_v=0.254
- **特征**: OpenFace2 713d 视觉 + COVAREP 74d 音频 + RoBERTa-large 文本
- **状态说明**: 使用 20,680 样本（非严格 18,571 complete-case）。2,109 样本存在视觉缺失（1,326）或音频 Inf（783）。已重新分类为 `aligned_or_mixed_tav_reference`。严格 cohort (18,571) 已构建但尚未重新训练。
- **对应配置**: `configs/canonical/mosei/control_awaf_slstm_s42.yaml`
- **对应输出**: `outputs/P6AA_tav/mosei/main/control_awaf_slstm_s42_s42_20260620_223047/`

### 消融矩阵（7 变体, 4-epoch, seed 42）

| 变体 | 描述 | ACC2 | Δ vs F0 | 95% CI | 显著性 |
|------|------|------|---------|--------|--------|
| **F3** | **全局静态可学习权重** | **79.30%** | **+0.98** | [+0.15, +1.79] | ✅ p<0.05 |
| F4 | 固定等权 1/3 | 79.30% | +0.98 | [+0.15, +1.79] | ✅ p<0.05 |
| F2 | T+V（无音频） | 78.72% | +0.40 | [-1.37, +0.58] | ❌ n.s. |
| F0 | TAV AWAF sLSTM | 78.32% | — | — | 控制组 |
| E1 | LSTM 替换 sLSTM | 77.78% | -0.54 | [-0.49, +1.58] | ❌ n.s. |
| F1 | T+A（无视觉） | 76.65% | -1.67 | [+0.61, +2.73] | ✅ p<0.05 |
| F5 | 无 Hadamard 交互 | 75.96% | -2.36 | [+1.49, +3.25] | ✅ p<0.05 |

- **因果贡献排序（4-epoch）**:
  1. Hadamard 交互项: **+2.37pp** ✅ (最大效应)
  2. 视觉模态: **+1.67pp** ✅
  3. 全局静态 > 动态 AWAF: **+0.97pp** ✅
  4. sLSTM vs LSTM: +0.54pp ❌ (不显著)
  5. 音频模态: -0.39pp ❌ (不显著)
- **对应配置**: `configs/canonical/mosei/*.yaml`
- **对应输出**: `outputs/P6AB_tav_ablation/mosei/*/`

---

## 实验状态分类定义

| 状态 | 英文 | 定义 | MOSI 示例 | MOSEI 示例 |
|------|------|------|-----------|-----------|
| 最终候选 | `final_candidate` | 冻结、多 seed、可进入论文主表 | F0 (3-seed) | — |
| 论文可用 | `paper_ready` | 单 seed 或消融、证据完整 | A1-A5 消融 | 4-epoch 消融 |
| 对齐引用 | `aligned_or_mixed_tav_reference` | 有效 TAV 但非严格 complete-case | — | 12-epoch 主模型 |
| 历史参考 | `historical_reference` | 不同配置/特征的有效结果 | P6K 88.72% | — |
| 塌缩证据 | `collapsed_evidence` | 塌缩，保留为诊断证据 | P6AD/P6AF | — |
| 不可比较 | `not_comparable` | 不同 cohort/protocol | — | — |

---

## 特征版本对照

| 模态 | MOSI | MOSEI |
|------|------|-------|
| 视觉来源 | CLIP-L14 (Transformer 深度特征) | OpenFace2 (传统面部动作单元) |
| 视觉维度 | 1024 | 713 |
| 视觉帧数 | 32 | 50 |
| 音频来源 | data2vec (Transformer 深度特征) | COVAREP (传统声学特征) |
| 音频维度 | 768 | 74 |
| 音频帧数 | 100 | 100 |
| 文本处理 | RoBERTa-large + LoRA (相同) | RoBERTa-large + LoRA (相同) |
| 数据版本 | `mosi_tav_v1` | `mosei_tav_openface2_v1` |
| Cohort ID | `mosi_official_tav_complete_case_v1` | `mosei_official_tav_intersection_v1` |
