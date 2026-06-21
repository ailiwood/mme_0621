"""
P6D: TextFT Multimodal Dataset — combines raw text with frozen audio/vision features.
"""
import csv, os, numpy as np, torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class TextFTMultimodalDataset(Dataset):
    def __init__(self, csv_path='data/mosi/label.csv', feature_root='data/features_strong_sequence_mosi_v3_T40',
                 split='train', tokenizer_name='roberta-large', max_text_len=128, max_audio_len=100, max_vision_len=50,
                 formal_mode=True):
        self.feature_root = feature_root
        self.max_text_len = max_text_len
        self.max_audio_len = max_audio_len
        self.max_vision_len = max_vision_len
        self.formal_mode = formal_mode

        # Read labels and raw text
        with open(csv_path, 'r', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        # Label CSV uses 'train'/'valid'/'test' for MOSEI, 'train'/'val'/'test' for MOSI
        csv_split_map = {'train': 'train', 'val': 'valid', 'test': 'test'}
        self.data = [r for r in rows if r.get('mode', 'train') == csv_split_map.get(split, split)]

        # Tokenizer for raw text
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)

        # Feature directory: try 'valid' first (MOSEI), then 'val' (MOSI), then split name
        for dir_name in [csv_split_map.get(split, split), split, split]:
            candidate = os.path.join(feature_root, dir_name)
            if os.path.isdir(candidate) and os.listdir(candidate):
                self.feat_dir = candidate
                break
        else:
            self.feat_dir = os.path.join(feature_root, split)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        r = self.data[idx]
        sample_id = f"{r['video_id']}_{r['clip_id']}"

        # Raw text → tokens
        text = r['text']
        tokens = self.tokenizer(text, padding='max_length', truncation=True,
                                max_length=self.max_text_len, return_tensors='pt')

        # Label
        label = float(r['label'])

        # Load frozen audio/vision features — formal_mode MUST have real features
        feat_path = os.path.join(self.feat_dir, f'{sample_id}.npz')
        if os.path.exists(feat_path):
            feat = np.load(feat_path, allow_pickle=True)
            audio_seq = torch.from_numpy(feat['audio_seq']).float()[:self.max_audio_len]
            audio_mask = torch.from_numpy(feat['audio_mask']).long()[:self.max_audio_len]
            vision_seq = torch.from_numpy(feat['vision_seq']).float()[:self.max_vision_len]
            vision_mask = torch.from_numpy(feat['vision_mask']).long()[:self.max_vision_len]
        elif self.formal_mode:
            raise FileNotFoundError(
                f"TextFTMultimodalDataset formal_mode: missing feature file for "
                f"sample_id={sample_id}, path={feat_path}. "
                f"Cannot use zero fallback in formal experiments."
            )
        else:
            # Debug-only fallback: zero features (NOT for formal experiments)
            audio_seq = torch.zeros(self.max_audio_len, 768)
            audio_mask = torch.zeros(self.max_audio_len, dtype=torch.long)
            vision_seq = torch.zeros(self.max_vision_len, 768)
            vision_mask = torch.zeros(self.max_vision_len, dtype=torch.long)

        return {
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'audio': audio_seq, 'audio_mask': audio_mask,
            'vision': vision_seq, 'vision_mask': vision_mask,
            'label': torch.tensor([label], dtype=torch.float32),
            'id': sample_id,
        }


def _clean_features(tensor, feature_name=''):
    """Replace -inf/+inf with 0.0 in feature tensors (COVAREP may have -inf from log(0))."""
    if torch.isinf(tensor).any():
        # Log first occurrence for debugging
        n_inf = torch.isinf(tensor).sum().item()
        if n_inf > 0:
            # Replace -inf and +inf with 0.0
            tensor = torch.where(torch.isinf(tensor), torch.zeros_like(tensor), tensor)
    if tensor.isnan().any():
        n_nan = tensor.isnan().sum().item()
        if n_nan > 0:
            tensor = torch.where(tensor.isnan(tensor), torch.zeros_like(tensor), tensor)
    return tensor


def collate_textft(batch):
    """Collate for TextFT multimodal data."""
    B = len(batch)
    max_al = batch[0]['audio'].size(0); max_vl = batch[0]['vision'].size(0)
    audio_dim = batch[0]['audio'].size(-1); vision_dim = batch[0]['vision'].size(-1)
    max_tl = batch[0]['input_ids'].size(0)

    input_ids = torch.zeros(B, max_tl, dtype=torch.long)
    attention_mask = torch.zeros(B, max_tl, dtype=torch.long)
    audio = torch.zeros(B, max_al, audio_dim)
    audio_mask = torch.zeros(B, max_al, dtype=torch.long)
    vision = torch.zeros(B, max_vl, vision_dim)
    vision_mask = torch.zeros(B, max_vl, dtype=torch.long)
    labels = torch.zeros(B, 1)
    ids = []

    for i, item in enumerate(batch):
        tl = min(item['input_ids'].size(0), max_tl)
        input_ids[i, :tl] = item['input_ids'][:tl]
        attention_mask[i, :tl] = item['attention_mask'][:tl]
        al = min(item['audio'].size(0), max_al)
        # Clean -inf/+inf from COVAREP audio features before inserting
        audio_i = _clean_features(item['audio'][:al])
        audio[i, :al] = audio_i
        audio_mask[i, :al] = item['audio_mask'][:al]
        vl = min(item['vision'].size(0), max_vl)
        # Clean vision features too (belt-and-suspenders)
        vision_i = _clean_features(item['vision'][:vl])
        vision[i, :vl] = vision_i
        vision_mask[i, :vl] = item['vision_mask'][:vl]
        labels[i] = item['label']
        ids.append(item['id'])

    return {'input_ids': input_ids, 'attention_mask': attention_mask,
            'audio': audio, 'audio_mask': audio_mask,
            'vision': vision, 'vision_mask': vision_mask,
            'label': labels, 'id': ids}
