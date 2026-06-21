#!/usr/bin/env python
"""Quick eval: load checkpoint + config, run test, save predictions."""
import sys, os, csv, torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.textft_multimodal_dataset import TextFTMultimodalDataset, collate_textft
from scripts.train_textft_lora_mainline import build_config, load_config_yaml
from models.textft_lora_xlstm_awaf_residual import TextFTLoRAXLSTMAWAFResidual
from utils.metrics import compute_all_metrics

config_path = sys.argv[1]
ckpt_path = sys.argv[2]
out_dir = sys.argv[3] if len(sys.argv) > 3 else os.path.dirname(ckpt_path)
DEVICE = 'cuda'

yc = load_config_yaml(config_path)
config = build_config(yc, DEVICE)
print(f'Mode: {config.mode}')

model = TextFTLoRAXLSTMAWAFResidual(config).to(DEVICE)
model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
model.eval()

test_ds = TextFTMultimodalDataset(split='test', formal_mode=True)
tl = DataLoader(test_ds, 4, shuffle=False, collate_fn=collate_textft)

tp, tl_, rb, aw, gv, delta = [], [], [], [], [], []
with torch.no_grad():
    for batch in tqdm(tl, desc='Eval'):
        out = model(batch)
        tp.append(out['reg'].cpu())
        tl_.append(batch['label'].cpu())
        if 'reg_text_base' in out: rb.append(out['reg_text_base'].cpu())
        if 'awaf_weights' in out: aw.append(out['awaf_weights'].cpu())
        if 'gate' in out: gv.append(out['gate'].cpu())
        elif 'gate_reg' in out: gv.append(out['gate_reg'].cpu())
        if 'effective_delta_reg' in out: delta.append(out['effective_delta_reg'].cpu())
        elif 'delta' in out: delta.append(out['delta'].cpu())

rp = torch.cat(tp); tg = torch.cat(tl_)
rs = torch.where(rp >= 0, 1.0, -1.0)
m = compute_all_metrics(rp, rs, tg)
print(f'ACC2: {m["ACC2_Non0"]:.2f}%  MAE: {m["MAE"]:.4f}  Corr: {m["Corr"]:.4f}  F1: {m["F1_Non0"]:.2f}%')

if rb:
    rb_t = torch.cat(rb)
    m_base = compute_all_metrics(rb_t, rs, tg)
    print(f'text_base ACC2: {m_base["ACC2_Non0"]:.2f}%  gain: {m["ACC2_Non0"]-m_base["ACC2_Non0"]:+.2f}%')

# Save
sample_ids = [f'sample_{i}' for i in range(len(tg))]
with open(os.path.join(out_dir, 'predictions_test.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    def assign_group(label_val):
        if label_val <= -1.5: return 'strong_neg'
        elif label_val <= -0.3: return 'weak_neg'
        elif label_val < 0.3: return 'near_zero'
        elif label_val < 1.5: return 'weak_pos'
        else: return 'strong_pos'
    cols = ['sample_id','label','text_base_pred','final_pred','delta','gate']
    if aw: cols += ['w_t','w_a','w_v']
    cols += ['text_base_correct','final_correct','group']
    w.writerow(cols)
    rp_n = rp.numpy().flatten(); tg_n = tg.numpy().flatten()
    rb_n = rb_t.numpy().flatten() if rb else rp_n
    gv_n = torch.cat(gv).numpy().flatten() if gv else np.zeros(len(tg))
    d_n = torch.cat(delta).numpy().flatten() if delta else np.zeros(len(tg))
    aw_n = torch.cat(aw).numpy() if aw else np.zeros((len(tg), 3))
    for i in range(len(tg)):
        row = [sample_ids[i], tg_n[i], rb_n[i], rp_n[i], d_n[i], gv_n[i]]
        if aw: row += [aw_n[i,0], aw_n[i,1], aw_n[i,2]]
        row += [int((rb_n[i]>=0)==(tg_n[i]>=0)), int((rp_n[i]>=0)==(tg_n[i]>=0)),
                assign_group(float(tg_n[i]))]
        w.writerow(row)
print(f'Saved: {os.path.join(out_dir, "predictions_test.csv")}')
