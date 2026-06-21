# README_data.md — 数据与特征版本说明

> 生成时间：2026-06-16 (P1)  
> 最后更新：待 MOSEI 特征提取完成后更新

---

## 一、数据集概览

| 数据集 | 任务类型 | 标签空间 | 样本总数 | 论文使用量 |
|--------|----------|----------|----------|------------|
| CMU-MOSI | 情感强度回归 | [-3, +3] | 2,199 clips (93 videos) | 全量 100% |
| CMU-MOSEI | 情感强度回归 | [-3, +3] | 22,856 segments | 全量或按标注使用 |

---

## 二、特征版本

### 2.1 当前特征版本：TMDC-v1

| 模态 | 编码器模型 | 特征维度 | 数据类型 | 备注 |
|------|-----------|:--:|------|------|
| Text | DeBERTa-large (microsoft/deberta-large) | 1024 | float32 | |
| Audio | wav2vec2-large-960h (facebook/wav2vec2-large-960h) | 1024 | float32 | |
| Vision | **CLIP-ViT-B/32** (openai/clip-vit-base-patch32) + fixed random projection (seed=42) to 1024d | 1024 | float32 | ⚠ 目录名 manet_UTT 为历史命名，**不代表 MANet**。实际模型为 CLIP-ViT-B/32 (768d) → 固定随机矩阵投影 (768→1024) |

### 2.2 科学性边界（决策 D015）

- **当前特征为 clip-level 单向量 (T=1)**，不是真正的时间序列特征
- P2 将其作为 T=1 输入用于工程 smoke test
- **不得将 T=1 smoke test 结果写成 "sLSTM 长程时序建模有效" 的论文证据**
- dataset 和模型接口已设计为兼容未来 [T, D] 序列输入，后续有真正序列特征时可无缝切换

### 2.2 特征固化状态

| 数据集 | 状态 | 提取日期 | Split | 路径 |
|--------|:--:|------|------|------|
| MOSI | ✅ 已固化 | 2026-06-12 | train=1284, val=229, test=686 | `tmdc_adapter/features/mosi/` |
| MOSEI | ❌ 未提取 | — | — | 待 P2 前自建 |

---

## 三、MOSI 数据说明

### 3.1 原始数据位置

```
data/mosi/
├── label.csv              # 标签文件（2199条）
├── Frames/<video_id>/     # 视频帧 (JPG)
├── Raw/<video_id>/        # 原始视频 (MP4)
└── wav/<video_id>/        # 音频 (WAV)
```

### 3.2 TMDC 特征位置

```
tmdc_adapter/features/mosi/
├── pkls/CMUMOSI_features_raw_2way.pkl    # TMDC 7-tuple (split+label+id)
├── deberta-large-4-UTT/<uid>.npy         # Text 1024d
├── wav2vec-large-c-UTT/<uid>.npy         # Audio 1024d
└── manet_UTT/<uid>.npy                   # Vision 1024d (CLIP→proj)
```

### 3.3 P2 数据加载方式

P2 将创建 `data/dataset.py`，从 TMDC 特征加载三模态数据，格式为：
```python
{
    'text': Tensor (1, 1024),    # 单 clip 为单向量，可重复到固定帧数
    'audio': Tensor (1, 1024),
    'vision': Tensor (1, 1024),
    'label': Tensor (1,),        # 回归值 [-3, 3]
}
```

注意：TMDC 特征是 per-clip 单一向量，不是序列特征。这适合 AWAF（AWAF 操作 per-sample 摘要向量），但 sLSTM 编码器需要序列输入。需要在 P2 中决定：(a) 将单向量重复为固定长度伪序列；(b) 调整为支持单向量输入。

---

## 四、MOSEI 数据说明

### 4.1 原始数据位置

```
data/CMU-MOSEI/CMU-MOSEI-20230514T151450Z-001/CMU-MOSEI/
├── Audio_chunk/
│   ├── Train_modified/    # 2087 wav
│   ├── Val_modified/      # 1176 wav
│   └── Test_modified/     # 1982 wav
├── Labels/
│   ├── Data_Train_modified.csv    # 16274 行
│   ├── Data_Val_modified.csv      # 1861 行
│   └── Data_Test_modified.csv     # 4653 行
├── Test_original/
└── Val_original/
```

### 4.2 特征提取计划（P2 前完成）

待提取维度同 MOSI：Text 1024d + Audio 1024d + Vision 1024d。

视觉特征提取需要原始视频文件来源（当前 `new_code/data/CMU-MOSEI` 中未找到现成视频帧）。待用户确认方案。

---

## 五、不采用的特征

| 特征 | 原因 |
|------|------|
| `mosi_simulated_features.pkl` | 文件名含 "simulated"，真实性存疑 |
| Tri_modal_ER MOSEI train/val/test.features | 标签为 0-1 二值，非标准回归；特征维度跨 split 不一致 |
| V9 RoBERTa-large MOSEI pkl | 路线已排除，且文件在外部路径 |
