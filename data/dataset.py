"""
data/dataset.py — 统一数据集类

当前支持:
  - TMDC MOSI 特征 (clip-level 单向量, T=1)
  - 兼容未来 [T, D] 序列特征

特征标注:
  Vision = CLIP-ViT-B/32 + fixed random projection to 1024d
  (目录名 manet_UTT 为历史命名，不代表 MANet)

引用决策 D014, D015。
"""
import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, Optional


class TMDCMOSIDataset(Dataset):
    """
    TMDC MOSI 数据集。

    特征来源:
      tmdc_adapter/features/mosi_pkls/CMUMOSI_features_raw_2way.pkl
      tmdc_adapter/features/mosi/deberta-large-4-UTT/
      tmdc_adapter/features/mosi/wav2vec-large-c-UTT/
      tmdc_adapter/features/mosi/manet_UTT/

    数据格式:
      每个 sample: {'text': [1, 1024], 'audio': [1, 1024], 'vision': [1, 1024],
                    'mask': [1], 'label': [1], 'id': str}

    注意:
      - 当前为 T=1 clip-level 单向量，兼容 future [T, D] 序列
      - Vision 实际为 CLIP-ViT-B/32 + projection，非 MANet
      - 接口设计支持 T>1 序列 (通过 expand 或未来直接加载序列特征)
    """

    def __init__(
        self,
        split: str = 'train',
        feature_root: str = None,
        pkl_path: str = None,
        modalities: tuple = ('text', 'audio', 'vision'),
    ):
        """
        Args:
            split: 'train' | 'val' | 'test'
            feature_root: TMDC 特征根目录
            pkl_path: TMDC 7-tuple pkl 路径
            modalities: 加载哪些模态
        """
        self.split = split

        # 默认路径 (相对于 new_code)
        if feature_root is None:
            feature_root = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'tmdc_adapter', 'features', 'mosi'
            )
        if pkl_path is None:
            pkl_path = os.path.join(feature_root, '..', 'mosi_pkls',
                                    'CMUMOSI_features_raw_2way.pkl')

        self.feature_root = feature_root
        self.modalities = modalities

        # --- 加载 split 和标签 ---
        with open(pkl_path, 'rb') as f:
            obj = pickle.load(f, encoding='latin1')
        videoIDs, videoLabels, videoSpeakers, videoSentences, trainVids, valVids, testVids = obj

        # 映射 split → video list
        if split == 'train':
            self.vids = trainVids
        elif split in ('val', 'valid'):
            self.vids = valVids
        elif split == 'test':
            self.vids = testVids
        else:
            raise ValueError(f"Unknown split: {split}")

        # 构建 (video_id, clip_idx) 列表
        self.samples = []
        for vid in self.vids:
            for idx in range(len(videoIDs[vid])):
                self.samples.append({
                    'uid': videoIDs[vid][idx],
                    'label': float(videoLabels[vid][idx]),
                    'vid': vid,
                })

        # 特征目录映射
        self.feature_dirs = {
            'text': os.path.join(feature_root, 'deberta-large-4-UTT'),
            'audio': os.path.join(feature_root, 'wav2vec-large-c-UTT'),
            'vision': os.path.join(feature_root, 'manet_UTT'),  # 历史命名
        }

    def __len__(self) -> int:
        return len(self.samples)

    def _load_feature(self, uid: str, modality: str) -> np.ndarray:
        """加载单个模态的 npy 特征 [D] → 返回 [1, D]"""
        fpath = os.path.join(self.feature_dirs[modality], uid + '.npy')
        feat = np.load(fpath).astype(np.float32)
        # 单向量 → [1, D] 以兼容序列接口
        if feat.ndim == 1:
            feat = feat[np.newaxis, :]  # [1, D]
        return feat

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        uid = sample['uid']

        result = {
            'id': uid,
            'label': torch.tensor([sample['label']], dtype=torch.float32),
        }

        if 'text' in self.modalities:
            text_feat = self._load_feature(uid, 'text')
            result['text'] = torch.from_numpy(text_feat).float()
            result['text_mask'] = torch.ones(text_feat.shape[0], dtype=torch.long)

        if 'audio' in self.modalities:
            audio_feat = self._load_feature(uid, 'audio')
            result['audio'] = torch.from_numpy(audio_feat).float()
            result['audio_mask'] = torch.ones(audio_feat.shape[0], dtype=torch.long)

        if 'vision' in self.modalities:
            vision_feat = self._load_feature(uid, 'vision')
            result['vision'] = torch.from_numpy(vision_feat).float()
            result['vision_mask'] = torch.ones(vision_feat.shape[0], dtype=torch.long)

        return result


def collate_fn(batch):
    """
    批处理 collate: 处理变长或不一致的 batch。

    当前 TMDC MOSI 所有样本 T=1 统一，简单 stack 即可。
    未来序列特征 T 不一致时，需要在此 padding。
    """
    keys = batch[0].keys()
    collated = {}
    for k in keys:
        vals = [b[k] for b in batch]
        if isinstance(vals[0], torch.Tensor):
            # 堆叠 tensors
            collated[k] = torch.stack(vals, dim=0)
        elif isinstance(vals[0], str):
            collated[k] = vals
        else:
            collated[k] = vals
    return collated


# ============================================================
# 数据集快速验证
# ============================================================
if __name__ == '__main__':
    print("=== TMDC MOSI 数据集验证 ===\n")

    for split in ['train', 'val', 'test']:
        ds = TMDCMOSIDataset(split=split)
        print(f"  [{split}] N={len(ds)}")

        if len(ds) > 0:
            sample = ds[0]
            print(f"    id:     {sample['id']}")
            print(f"    label:  {sample['label'].item():.3f}")
            print(f"    text:   {sample['text'].shape}")
            print(f"    audio:  {sample['audio'].shape}")
            print(f"    vision: {sample['vision'].shape}")

            # 验证形状
            T = sample['text'].shape[0]
            assert T == 1, f"Expected T=1, got T={T}"
            assert sample['text'].shape[1] == 1024
            assert sample['audio'].shape[1] == 1024
            assert sample['vision'].shape[1] == 1024

    # 验证标签范围
    ds_train = TMDCMOSIDataset('train')
    labels = [s['label'].item() for i in range(len(ds_train))
              for s in [ds_train[i]] if i < len(ds_train)][:100]
    all_labels = []
    for i in range(len(ds_train)):
        all_labels.append(ds_train[i]['label'].item())
    print(f"\n  Label range: [{min(all_labels):.3f}, {max(all_labels):.3f}]")
    pos = sum(1 for l in all_labels if l >= 0)
    print(f"  Positive (>=0): {pos}/{len(all_labels)} = {pos/len(all_labels)*100:.1f}%")

    # 验证 train/val/test 不重叠
    train_ids = set(ds_train.samples[i]['uid'] for i in range(len(ds_train)))
    val_ids = set(TMDCMOSIDataset('val').samples[i]['uid'] for i in range(len(TMDCMOSIDataset('val'))))
    test_ids = set(TMDCMOSIDataset('test').samples[i]['uid'] for i in range(len(TMDCMOSIDataset('test'))))
    assert len(train_ids & val_ids) == 0, "Train-Val overlap!"
    assert len(train_ids & test_ids) == 0, "Train-Test overlap!"
    assert len(val_ids & test_ids) == 0, "Val-Test overlap!"
    print(f"  Split 无重叠 ✅")

    print("\n=== 数据集验证完成 ===")
