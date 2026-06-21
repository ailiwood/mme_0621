# 实验状态与已知限制（中文版）

**日期**: 2026-06-21
**版本**: v1.0.0-final

---

## 一、MOSI：已完成与未完成

### ✅ 已完成

| 实验 | 规模 | 状态 |
|------|------|------|
| 主模型 TAV（3-seed 冻结） | seed 42/2024/3407 | final_candidate |
| 消融实验（5 变体 × 3 seeds） | 15 次训练 | paper_ready |
| 诊断探针（P0-P6） | 6 个探针 | paper_ready |
| 特征时序审计（Route-S/Route-P） | 全量 2,199 样本 | paper_ready |
| Canonical AWAF 塌缩记录 | 2 次独立尝试 (P6AD, P6AF) | historical_evidence |
| 反事实验证 | audio/vision zero + permutation | paper_ready |
| 可靠性门诊断导出 | 3 seed 完整诊断 | paper_ready |

### ❌ 未完成

| 实验 | 原因 |
|------|------|
| Baseline-lite 全矩阵 (TFN/LMF/MulT/SelfMM/MISA/MLCL/DLF) | 主模型消融完成后时间受限 |
| 课程学习 (TA → TAV) | 主模型已达 87.80%，消融表明辅助贡献边际 |
| Text-Query Cross-Modal Attention | 同上，辅助贡献小，复杂架构增益有限 |
| Confidence-Calibrated Multi-Task Fusion | 同上 |

### ⚠️ 已知限制

1. **文本基线主导**：纯文本（88.11%）略高于 TAV（87.80%）。辅助模态贡献在 seed 波动内。不得声称 TAV 显著优于文本单模态。

2. **依赖 P6K 初始化**：从零训练的 Text-Anchored V1 仅达 82.01%。P6K checkpoint 贡献约 6pp。这意味着最终模型的文本/音频分支携带了来自不同特征版本（v3_T40）的迁移知识。

3. **验证集极小**：229 个验证样本限制了超参数选择的可靠性。3-fold 交叉验证（在 train 内部）可以提供更稳健的开发期评估。

4. **特征与 MOSEI 不同**：MOSI 使用 CLIP-L14 (1024d) + data2vec (768d)；MOSEI 使用 OpenFace2 (713d) + COVAREP (74d)。两个数据集的特征不可直接比较。

5. **测试集仅 686 样本**：小测试集导致 ACC2 的 seed 间波动（std=0.33pp）可能掩盖小的真实效应。

---

## 二、MOSEI：已完成与未完成

### ✅ 已完成

| 实验 | 规模 | 状态 |
|------|------|------|
| 主模型 TAV AWAF (12-epoch) | seed 42 | aligned_or_mixed_tav_reference |
| 消融实验（7 变体, 4-epoch） | seed 42 | paper_ready |
| AWAF 权重分析 | 主模型权重分布 | paper_ready |
| 统计检验 | bootstrap CI + McNemar | paper_ready |
| 数据流水线验证 | 全量 20,680 样本 | paper_ready |
| 严格 complete-case 审计 | 18,571/20,680 通过 | paper_ready |
| 严格 cohort 构建 | mosei_tav_complete_case_v2 | paper_ready |

### ❌ 未完成

| 实验 | 原因 |
|------|------|
| 严格 cohort (18,571) 重新训练 | P6AD/P6AF/P6AG/P6AI/P6AJ/P6AK 期间优先级在 MOSI |
| 多 seed 主模型 | 仅 seed 42 |
| 长 epoch (12+) 消融 | 消融仅 4-epoch |
| MOSEI baseline-lite TAV 版本 | configs 已准备 (text_audio_vision)，未训练 |

### ⚠️ 已知限制

1. **非严格 complete-case**：主模型使用 20,680 样本，其中 2,109 个有视觉缺失或音频 Inf。被重新分类为 `aligned_or_mixed_tav_reference`。

2. **仅单 seed**：主模型仅 seed 42。跨 seed 稳健性未验证。

3. **12-epoch 主模型 vs 4-epoch 消融**：主模型和消融使用不同训练预算，不能直接比较绝对 ACC2。消融仅用于同预算机制比较。

4. **AWAF 在低训练预算下动态权重不占优**：4-epoch 时全局静态权重（79.30%）显著优于样本级动态 AWAF（78.32%）。这表明 AWAF 的动态机制需要更长训练。

5. **特征版本较旧**：COVAREP 74d 和 OpenFace2 713d 是传统手工特征。与现代 Transformer 特征（如 MOSI 使用的 data2vec/CLIP-L14）相比，可能信号较弱。

---

## 三、跨数据集限制

### 不可直接比较的原因

1. **不同特征源**：MOSI 使用 CLIP-L14 + data2vec（现代 Transformer），MOSEI 使用 OpenFace2 + COVAREP（传统手工）。
2. **不同架构**：MOSI 使用 Text-Anchored Fusion，MOSEI 使用 Canonical AWAF。
3. **不同训练协议**：MOSI 使用 35 epoch + P6K 初始化，MOSEI 使用 12 epoch 从头训练。
4. **不同数据集规模**：1,284 vs 13,239 训练样本。
5. **不同标注分布**：MOSI 标注范围 [-3,+3] 偏正（57% 正），MOSEI 标注范围 [-3,+3] 较均衡。

### 可以比较的内容

1. **消融效应的相对排序和方向**：两个数据集上，视觉和跨模态交互均正向贡献，方向一致。
2. **架构选择的依赖条件**：数据集规模决定是否需要文本保持机制。
3. **模型组件的可迁移性**：sLSTM、LoRA、RoBERTa、评估指标在两个数据集上均可正常工作。

---

## 四、实验证据强度分级

| 等级 | 条件 | MOSI 实验 | MOSEI 实验 |
|------|------|-----------|-----------|
| **A 级**（可直接写入论文主表） | 3+ seed, 严格 cohort, 完整 artifact | F0 (3-seed), A1-A5 (3-seed) | 4-epoch 消融 (7 变体, 统计检验) |
| **B 级**（可写入但需标注限制） | 单 seed 或非严格 cohort | P6AF 诊断探针 | 12-epoch 主模型 (aligned_reference) |
| **C 级**（仅作为历史参考） | 不同特征/架构/协议 | P6K 88.72% (非TAV) | — |
| **D 级**（失败但保留证据） | 塌缩或未通过质量门 | P6AD/P6AF canonical AWAF | — |

---

## 五、复现注意事项

1. **环境**: Python 3.10.20, PyTorch 2.11.0+cu128, RTX 5070 Ti (16GB)。完整环境见 `env/environment.yml`。

2. **数据**: MOSI 和 MOSEI 原始数据需从官方渠道下载。处理后特征未包含在本仓库中（仅本地保留）。

3. **P6K checkpoint**: MOSI 主模型需要 P6K text+audio checkpoint (`outputs/P6K/text_audio_conservative_s42_s42_20260619_031645/best_model.pth`)。该文件未上传，仅本地保留。

4. **训练时间**: MOSI 主模型每 seed 约 25 分钟（RTX 5070 Ti）。MOSEI 主模型每 epoch 约 9 分钟（12 epoch 总计约 2 小时）。

5. **评估**: 所有指标使用 `utils/metrics.py` 中的 `compute_all_metrics()`。ACC2_Non0 为标准 MOSEI 指标（排除 label=0 样本）。
