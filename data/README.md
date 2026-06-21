# 数据集说明文档

**日期**: 2026-06-21
**版本**: v1.0.0-final

---

## 目录结构

```
data/
├── README.md                                    ← 本文件
├── textft_multimodal_dataset.py                 ← 多模态数据加载器（核心代码）
├── features_strong_sequence_mosi_v3_T40/        ← P6K 历史参考特征（MOSI v3，T=40帧视觉）
├── features_strong_sequence_mosi_v6_vision_l14_full/ ← MOSI v6 源特征（CLIP-L14 视觉）
└── processed/                                   ← 处理后的特征（模型实际读取）
    ├── mosei_full/                              ← MOSEI 完整特征（含标签）
    ├── mosei_tav_openface2_v1/                  ← MOSEI TAV 特征（train/valid/test）
    ├── mosei_tav_complete_case_v2/              ← MOSEI 严格 complete-case 特征
    └── mosi_tav_v1/                             ← MOSI TAV 特征（train/val/test）
```

---

## 一、数据集概览

### CMU-MOSEI

| 属性 | 值 |
|------|-----|
| 全称 | CMU Multimodal Opinion Sentiment and Emotion Intensity |
| 样本总数 | 22,856 个视频片段 |
| 标注范围 | [-3, +3] 情感强度 |
| 模态 | 文本（原始转写）、音频（COVAREP）、视觉（OpenFace2） |
| 官方分割 | train 14,700 / valid 1,759 / test 4,221 |
| 标签文件 | `processed/mosei_full/label.csv` |

### CMU-MOSI

| 属性 | 值 |
|------|-----|
| 全称 | CMU Multimodal Opinion-level Sentiment Intensity |
| 样本总数 | 2,199 个视频片段 |
| 标注范围 | [-3, +3] 情感强度 |
| 模态 | 文本（原始转写）、音频（data2vec）、视觉（CLIP-L14） |
| 官方分割 | train 1,284 / val 229 / test 686 |
| 标签文件 | 原始 `features_strong_sequence_mosi_v3_T40` 内含 label，已迁移至 `mosi_tav_v1` 使用 |

---

## 二、特征格式详解

### 2.1 标准 .npz 文件格式

模型读取的每个样本是一个 `.npz` 文件，包含以下四个键：

| 键名 | 形状 (MOSEI) | 形状 (MOSI) | 数据类型 | 说明 |
|------|-------------|-------------|----------|------|
| `audio_seq` | (100, 74) | (100, 768) | float32 | 音频特征序列 |
| `audio_mask` | (100,) | (100,) | int64 | 音频有效帧掩码（1=有效，0=填充） |
| `vision_seq` | (50, 713) | (32, 1024) | float32 | 视觉特征序列 |
| `vision_mask` | (50,) | (32,) | int64 | 视觉有效帧掩码（1=有效，0=填充） |

### 2.2 特征来源对比

| 属性 | MOSEI | MOSI |
|------|-------|------|
| 音频特征源 | COVAREP（传统声学特征） | data2vec（Transformer 深度特征） |
| 音频维度 | 74 | 768 |
| 音频帧数 | 100 | 100 |
| 视觉特征源 | OpenFace2（传统面部CV特征） | CLIP-L14（Transformer 深度特征） |
| 视觉维度 | 713 | 1024 |
| 视觉帧数 | 50 | 32 |
| 文本处理 | 原始文本 → RoBERTa-large tokenizer | 原始文本 → RoBERTa-large tokenizer |

### 2.3 文件命名规则

每个 `.npz` 文件的文件名为 `{video_id}_{clip_id}.npz`，例如：
- MOSEI: `-3g5yACwYnA_119_919.npz`（video_id = `-3g5yACwYnA`，clip_id = `119_919`）
- MOSI: `03bSnISJMiM_11.npz`（video_id = `03bSnISJMiM`，clip_id = `11`）

---

## 三、数据被模型读取的完整流程

### 3.1 入口：训练脚本

训练脚本 `scripts/train_textft_lora_mainline.py` 从 YAML 配置文件读取数据路径：

```yaml
# MOSI 示例 (configs/experiments/p6aj_mosi_87_recovery/P1_p6k_init_tav_s42.yaml)
data:
  csv_path: data/mosi/label.csv              # 标签 CSV 路径（已迁移至本地证据）
  dataset: mosi                                # 数据集名称
  feature_root: data/processed/mosi_tav_v1     # 特征文件目录
  formal_mode: true                            # 正式模式：缺少特征即报错

# MOSEI 示例 (configs/canonical/mosei/control_awaf_slstm_s42.yaml)
data:
  csv_path: data/processed/mosei_full/label.csv
  dataset: mosei
  feature_root: data/processed/mosei_tav_openface2_v1
  formal_mode: true
```

### 3.2 数据集类：`TextFTMultimodalDataset`

核心代码：`data/textft_multimodal_dataset.py`

```python
class TextFTMultimodalDataset(Dataset):
    def __init__(self, csv_path, feature_root, split, tokenizer_name, formal_mode):
        # 1. 读取标签 CSV
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))

        # 2. 按 split 筛选样本
        #    MOSEI: mode 列为 'train'/'valid'/'test'
        #    MOSI:  mode 列为 'train'/'val'/'test'
        self.data = [r for r in rows if r['mode'] == split]

        # 3. 加载 RoBERTa tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-large')

        # 4. 定位特征目录（尝试多个可能的目录名）
        for dir_name in [split, 'valid', 'val']:
            candidate = os.path.join(feature_root, dir_name)
            if os.path.isdir(candidate):
                self.feat_dir = candidate
                break

    def __getitem__(self, idx):
        r = self.data[idx]
        sample_id = f"{r['video_id']}_{r['clip_id']}"

        # === 文本处理 ===
        text = r['text']                              # 原始转写文本
        tokens = self.tokenizer(
            text,
            padding='max_length',                     # 填充到固定长度
            truncation=True,                          # 超长截断
            max_length=128,                           # 最大 128 tokens
            return_tensors='pt'
        )
        # 输出: input_ids [1, 128], attention_mask [1, 128]

        # === 标签 ===
        label = float(r['label'])                     # 情感强度 [-3, +3]

        # === 音频 + 视觉特征 ===
        feat_path = os.path.join(self.feat_dir, f'{sample_id}.npz')
        if os.path.exists(feat_path):
            feat = np.load(feat_path)
            audio_seq  = feat['audio_seq'][:max_audio_len]    # 截断到最大长度
            audio_mask = feat['audio_mask'][:max_audio_len]
            vision_seq = feat['vision_seq'][:max_vision_len]
            vision_mask= feat['vision_mask'][:max_vision_len]
        elif self.formal_mode:
            raise FileNotFoundError(...)               # 正式模式：特征缺失即报错
        else:
            # 调试模式：特征缺失时填充全零（严禁在正式实验中使用）
            audio_seq = torch.zeros(max_audio_len, 768)
            ...

        return {
            'input_ids':      tokens['input_ids'].squeeze(0),   # [128]
            'attention_mask': tokens['attention_mask'].squeeze(0), # [128]
            'audio':          audio_seq,      # [max_audio_len, audio_dim]
            'audio_mask':     audio_mask,     # [max_audio_len]
            'vision':         vision_seq,     # [max_vision_len, vision_dim]
            'vision_mask':    vision_mask,    # [max_vision_len]
            'label':          label,          # [1]
            'id':             sample_id,      # string
        }
```

### 3.3 批处理函数：`collate_textft`

由于 MOSI 的音频序列长度不固定（11–100 帧），需要 `collate_textft` 进行批次内填充：

```python
def collate_textft(batch):
    # 以批次内第一个样本的长度为基准
    max_al = batch[0]['audio'].size(0)
    max_vl = batch[0]['vision'].size(0)

    # 创建全零批次张量
    audio  = torch.zeros(B, max_al, audio_dim)
    vision = torch.zeros(B, max_vl, vision_dim)
    ...

    # 逐样本填充（超出部分截断，不足部分保持为 0）
    for i, item in enumerate(batch):
        al = min(item['audio'].size(0), max_al)
        audio[i, :al] = item['audio'][:al]
        ...
```

### 3.4 模型接收数据后的处理流程

```
DataLoader 输出 batch
      │
      ▼
model.forward(batch)
      │
      ├── Text:  input_ids → RoBERTa-large + LoRA → TextMLP → h_t       [B, 256]
      │
      ├── Audio: audio_seq → Linear(audio_dim→256) → sLSTM → Pool → h_a [B, 256]
      │           （通过 audio_mask 控制有效帧）
      │
      ├── Vision: vision_seq → Linear(vision_dim→256) → sLSTM → Pool → h_v [B, 256]
      │           （通过 vision_mask 控制有效帧）
      │
      └── Fusion: h_t, h_a, h_v → 融合模块 → y_hat                      [B, 1]
                 （MOSEI: AWAF softmax 加权 / MOSI: Text-Anchored Reliable Fusion）
```

---

## 四、各数据目录详细说明

### 4.1 `processed/mosei_full/`

| 属性 | 值 |
|------|-----|
| 用途 | MOSEI 完整特征 + 标签 |
| 样本数 | train 14,700 / valid 1,759 / test 4,221 |
| 特征版本 | COVAREP 74d 音频 + 原始 768d 视觉（旧版） |
| 标签文件 | `label.csv`（含 video_id, clip_id, text, label, mode） |
| 使用场景 | label.csv 被所有 MOSEI 实验引用；特征被旧版 configs 使用 |

### 4.2 `processed/mosei_tav_openface2_v1/`

| 属性 | 值 |
|------|-----|
| 用途 | MOSEI TAV 实验的主特征目录 |
| 样本数 | train 14,700 / valid 1,759 / test 4,221 |
| 特征版本 | COVAREP 74d 音频 + OpenFace2 713d 视觉 |
| Cohort | `mosei_official_tav_intersection_v1`（全部 20,680 样本） |
| 使用场景 | P6AA 主模型、P6AB 消融实验 |
| 注意 | 包含 2,109 个非严格 complete-case 样本（1,326 视觉全零，783 音频 Inf） |

### 4.3 `processed/mosei_tav_complete_case_v2/`

| 属性 | 值 |
|------|-----|
| 用途 | MOSEI 严格 complete-case 筛选后的特征 |
| 样本数 | train 13,239 / valid 1,561 / test 3,771 |
| 特征版本 | COVAREP 74d 音频 + OpenFace2 713d 视觉 |
| 筛选条件 | audio_mask>0, vision_mask>0, 无不合法值, 非全零 |
| 使用场景 | 已构建但尚未重新训练（P6AD-G0 审计产物） |
| 注意 | 这是论文级严格 cohort，未来重训应使用此版本 |

### 4.4 `processed/mosi_tav_v1/`

| 属性 | 值 |
|------|-----|
| 用途 | MOSI TAV 实验的主特征目录 |
| 样本数 | train 1,284 / val 229 / test 686 |
| 特征版本 | data2vec 768d 音频 + CLIP-L14 1024d 视觉 |
| 源数据 | `features_strong_sequence_mosi_v6_vision_l14_full/` |
| 构建脚本 | `scripts/build_mosi_tav_dataset_v1.py`（已归档） |
| 使用场景 | P6AJ 最终主模型、P6AK 消融实验 |
| 注意 | 音频长度不固定（11–100 帧），需 collate_textft 处理 |

### 4.5 `features_strong_sequence_mosi_v3_T40/`

| 属性 | 值 |
|------|-----|
| 用途 | P6K 历史参考特征（文本+音频+视觉一体 npz） |
| 样本数 | train 1,284 / val 229 / test 686 |
| 特征版本 | data2vec 768d 音频 + 768d 视觉（40 帧） |
| 使用场景 | P6K T+A conservative 参考模型（ACC2=88.72%） |
| 注意 | 视觉为 768d（非 CLIP-L14 1024d），与 mosi_tav_v1 特征版本不同 |

### 4.6 `features_strong_sequence_mosi_v6_vision_l14_full/`

| 属性 | 值 |
|------|-----|
| 用途 | `mosi_tav_v1` 的源数据 |
| 样本数 | train 1,284 / val 229 / test 686 |
| 特征版本 | data2vec 768d 音频 + CLIP-L14 1024d 视觉（32 帧） |
| npz 内容 | 包含 text_seq, audio_seq, vision_seq, label, sample_id |
| 使用场景 | 已转换为 `mosi_tav_v1` 标准格式。原始文件作为备份保留 |

---

## 五、数据预处理说明

### 5.1 MOSI TAV 特征构建流程

```
原始数据: features_strong_sequence_mosi_v6_vision_l14_full/
    │  (每个 npz 包含: text_seq, audio_seq, vision_seq, label, sample_id)
    │
    ▼ build_mosi_tav_dataset_v1.py
    │
处理后: processed/mosi_tav_v1/
    │  (每个 npz 仅包含: audio_seq, audio_mask, vision_seq, vision_mask)
    │  (标签从 data/mosi/label.csv 读取，文本由 RoBERTa tokenizer 实时处理)
```

### 5.2 MOSEI TAV 特征构建流程

```
原始数据: 外部 OpenFace2 特征 + COVAREP 特征
    │
    ▼ build_mosei_tav_openface2_v1.py
    │
处理后: processed/mosei_tav_openface2_v1/
    │  (每个 npz 包含: audio_seq, audio_mask, vision_seq, vision_mask)
```

### 5.3 MOSEI Strict Cohort 筛选流程

```
processed/mosei_tav_openface2_v1/  (20,680 样本)
    │
    ▼ audit_and_build_mosei_strict_complete_case_v2.py
    │  筛选条件:
    │  - audio_mask.sum() > 0
    │  - vision_mask.sum() > 0
    │  - 音频全部有限（无 NaN/Inf）
    │  - 视觉全部有限
    │  - 音频非全零
    │  - 视觉非全零
    │  排除: 1,326 视觉全零 + 783 音频 Inf = 2,109 样本
    │
    ▼
processed/mosei_tav_complete_case_v2/  (18,571 样本)
```

### 5.4 特征质量控制

在 `textft_multimodal_dataset.py` 的 `__getitem__` 中不进行 NaN/Inf 清洗。特征质量在构建阶段已保证：

- MOSEI: COVAREP 音频可能含 -Inf（log(0)），在数据集 `__getitem__` 中通过 `_clean_features()` 替换为 0.0
- MOSI: data2vec + CLIP-L14 特征均为有限值，无需额外清洗

---

## 六、标签 CSV 格式

### MOSEI (`processed/mosei_full/label.csv`)

| 列名 | 类型 | 示例 | 说明 |
|------|------|------|------|
| video_id | string | `-3g5yACwYnA` | 视频 ID |
| clip_id | string | `119_919` | 片段 ID |
| text | string | `"That was really funny"` | 原始转写文本 |
| label | float | `1.0` | 情感强度 [-3, +3] |
| mode | string | `train` | 数据集分割（train/valid/test） |
| annotation | string | `"happy"` | 情感标签（可选） |

### MOSI (`data/mosi/label.csv` — 本地保留)

| 列名 | 类型 | 示例 | 说明 |
|------|------|------|------|
| video_id | string | `03bSnISJMiM` | 视频 ID |
| clip_id | string | `11` | 片段 ID |
| text | string | `"A lot of sad parts"` | 原始转写文本 |
| label | float | `-0.5` | 情感强度 [-3, +3] |
| mode | string | `train` | 数据集分割（train/val/test） |
| annotation | string | `Neutral` | 情感标签（可选） |

---

## 七、重要注意事项

1. **两个数据集的特征版本不同**：MOSI 使用现代 Transformer 特征（data2vec + CLIP-L14），MOSEI 使用传统手工特征（COVAREP + OpenFace2）。两者不可直接数值比较。

2. **MOSEI 严格 cohort 尚未用于训练**：P6AA/P6AB 使用 20,680 样本版本。严格 18,571 完整条件版本（`mosei_tav_complete_case_v2`）已构建但未重新训练。

3. **MOSI 音频长度不固定**：需要 `collate_textft` 函数进行批次内填充，通过 `audio_mask` 标识有效帧。

4. **`formal_mode=True` 是必须的**：正式实验中必须设置 `formal_mode: true`，确保缺失的特征文件会抛出错误而非静默填充全零。

5. **数据文件和模型权重不在 Git 仓库中**：本目录下的所有特征数据（.npz）、标签文件、模型权重（.pth）均未上传到 GitHub。仅代码和配置文件在仓库中。
