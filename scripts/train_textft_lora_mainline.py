#!/usr/bin/env python
"""
scripts/train_textft_lora_mainline.py — P6H-R TextFT LoRA AWAF 训练主脚本

支持:
  - YAML 配置文件
  - per-epoch metrics CSV
  - per-sample 预测 / AWAF 权重 / delta / gate 导出
  - 训练曲线、混淆矩阵、权重分布图
  - val-only 模式 (sweep 阶段不碰 test)
  - AWAF entropy regularization

用法:
  python scripts/train_textft_lora_mainline.py --config configs/experiments/p6h_repair/mosi_r1_norm_tau_dsr_gate.yaml
"""
import sys, os, json, time, csv, argparse, yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from transformers import AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.textft_multimodal_dataset import TextFTMultimodalDataset, collate_textft
from models.textft_lora_xlstm_awaf_residual import TextFTLoRAConfig, TextFTLoRAXLSTMAWAFResidual
from utils.metrics import compute_all_metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, required=True, help='YAML config path')
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--max_epochs', type=int, default=0, help='Override epochs (0=use config)')
    p.add_argument('--limit_batches', type=int, default=0, help='Limit train/val batches (0=use all)')
    p.add_argument('--smoke', action='store_true', help='Smoke test: 1 epoch, 5 batches')
    p.add_argument('--init_checkpoint', type=str, default='', help='Path to checkpoint for weight initialization (P6K compatible)')
    return p.parse_args()


def load_config_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def build_config(yc, device='cuda'):
    """从 YAML dict 构建 TextFTLoRAConfig (P6I)。"""
    m = yc.get('model', yc)
    t = yc.get('training', {})
    return TextFTLoRAConfig(
        # --- P6I mode ---
        mode=m.get('mode', 'text_av_residual'),
        freeze_text_base=m.get('freeze_text_base', False),
        # --- Text ---
        text_model_name=m.get('text_model_name', 'roberta-large'),
        hidden_dim=m.get('hidden_dim', 256),
        audio_input_dim=m.get('audio_dim', 768),
        vision_input_dim=m.get('vision_dim', 768),
        slstm_num_layers=m.get('slstm_num_layers', 1),
        slstm_dropout=m.get('slstm_dropout', 0.2),
        text_mlp_hidden=m.get('text_mlp_hidden', 512),
        text_dropout=m.get('text_dropout', 0.1),
        lora_r=m.get('lora_r', 16),
        lora_alpha=m.get('lora_alpha', 32),
        lora_dropout=m.get('lora_dropout', 0.05),
        lora_targets=tuple(m.get('lora_targets', ['query', 'value'])),
        # --- AWAF ---
        awaf_fusion_mode=m.get('awaf_fusion_mode', 'awaf'),
        tau_init=m.get('tau_init', 3.0),
        awaf_dropout=m.get('awaf_dropout', 0.1),
        use_modality_dropout=m.get('use_modality_dropout', True),
        modality_dropout_prob=m.get('modality_dropout_prob', 0.1),
        use_modal_layernorm=m.get('use_modal_layernorm', True),
        awaf_uniform_mix=m.get('awaf_uniform_mix', 0.0),
        lambda_awaf_entropy=m.get('lambda_awaf_entropy', 0.01),
        # --- Old Gate/Delta (P6I: default off) ---
        use_uncertainty_gate=m.get('use_uncertainty_gate', False),
        gate_init_bias=m.get('gate_init_bias', 2.0),
        use_delta_experts=m.get('use_delta_experts', False),
        max_delta=m.get('max_delta', 1.0),
        delta_scale_init=m.get('delta_scale_init', 0.2),
        # --- Text-Confidence Residual (P6I) ---
        use_text_conf_residual=m.get('use_text_conf_residual', False),
        tcr_gate_hidden_dim=m.get('tcr_gate_hidden_dim', 128),
        tcr_delta_hidden_dim=m.get('tcr_delta_hidden_dim', 128),
        tcr_dropout=m.get('tcr_dropout', 0.1),
        tcr_max_delta=m.get('tcr_max_delta', 1.0),
        tcr_gate_floor=m.get('tcr_gate_floor', 0.2),
        tcr_detach_text_for_residual=m.get('tcr_detach_text_for_residual', True),
        delta_loss_weight=m.get('delta_loss_weight', 1.0),
        # --- P6AB: ablation-critical fields ---
        fusion_type=m.get('fusion_type', 'awaf'),
        awaf_context=m.get('awaf_context', True),
        awaf_interaction=m.get('awaf_interaction', True),
        temporal_encoder=m.get('temporal_encoder', 'slstm'),
        slstm_bidirectional=m.get('slstm_bidirectional', False),
        # --- Device ---
        device=device,
    )


# ================================================================
# Group label
# ================================================================
def assign_group(label_val: float) -> str:
    """按 label 值分组: strong_neg / weak_neg / near_zero / weak_pos / strong_pos"""
    if label_val <= -1.5:
        return 'strong_neg'
    elif label_val <= -0.3:
        return 'weak_neg'
    elif label_val < 0.3:
        return 'near_zero'
    elif label_val < 1.5:
        return 'weak_pos'
    else:
        return 'strong_pos'


# ================================================================
# Save helpers
# ================================================================
def save_predictions_csv(filepath, sample_ids, labels, rtb, final, delta, gate, weights):
    """保存 per-sample 预测 CSV。"""
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['sample_id', 'label', 'text_base_pred', 'final_pred', 'delta',
                     'gate', 'w_t', 'w_a', 'w_v', 'text_base_correct', 'final_correct', 'group'])
        for i in range(len(sample_ids)):
            tb_correct = int((rtb[i] >= 0) == (labels[i] >= 0))
            fn_correct = int((final[i] >= 0) == (labels[i] >= 0))
            group = assign_group(float(labels[i]))
            w.writerow([
                sample_ids[i], float(labels[i]),
                float(rtb[i]), float(final[i]), float(delta[i]), float(gate[i]),
                float(weights[i][0]), float(weights[i][1]), float(weights[i][2]),
                tb_correct, fn_correct, group,
            ])


def save_group_error_csv(filepath, sample_ids, labels, rtb, final, weights):
    """保存分组错误分析 CSV。"""
    groups = {'strong_neg': [], 'weak_neg': [], 'near_zero': [], 'weak_pos': [], 'strong_pos': []}
    for i in range(len(sample_ids)):
        g = assign_group(float(labels[i]))
        groups[g].append({
            'sample_id': sample_ids[i],
            'label': float(labels[i]),
            'text_base_pred': float(rtb[i]),
            'final_pred': float(final[i]),
            'abs_error_tb': abs(float(rtb[i]) - float(labels[i])),
            'abs_error_final': abs(float(final[i]) - float(labels[i])),
            'tb_correct': int((rtb[i] >= 0) == (labels[i] >= 0)),
            'fn_correct': int((final[i] >= 0) == (labels[i] >= 0)),
            'w_t': float(weights[i][0]), 'w_a': float(weights[i][1]), 'w_v': float(weights[i][2]),
        })
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['group', 'n_samples', 'tb_acc', 'fn_acc', 'mae_tb', 'mae_fn',
                     'mean_w_t', 'mean_w_a', 'mean_w_v'])
        for gname, items in groups.items():
            n = len(items)
            if n == 0:
                w.writerow([gname, 0, '', '', '', '', '', '', ''])
                continue
            tb_acc = sum(it['tb_correct'] for it in items) / n * 100
            fn_acc = sum(it['fn_correct'] for it in items) / n * 100
            mae_tb = sum(it['abs_error_tb'] for it in items) / n
            mae_fn = sum(it['abs_error_final'] for it in items) / n
            mwt = sum(it['w_t'] for it in items) / n
            mwa = sum(it['w_a'] for it in items) / n
            mwv = sum(it['w_v'] for it in items) / n
            w.writerow([gname, n, f'{tb_acc:.2f}', f'{fn_acc:.2f}',
                         f'{mae_tb:.4f}', f'{mae_fn:.4f}',
                         f'{mwt:.4f}', f'{mwa:.4f}', f'{mwv:.4f}'])


def save_plots(out_dir, epochs_list, metrics_dict, rtb, final, labels, weights, gates, deltas):
    """生成训练曲线 + 分布图 (需要 matplotlib)。"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('[WARN] matplotlib 不可用，跳过绘图')
        return

    # 1. 训练曲线
    if epochs_list and metrics_dict:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        ax = axes[0, 0]
        ax.plot(epochs_list, metrics_dict.get('val_ACC2', []), 'b-o', label='val_ACC2', markersize=4)
        ax.axhline(y=max(metrics_dict.get('val_ACC2', [0])), color='b', linestyle='--', alpha=0.3)
        ax.set_xlabel('Epoch'); ax.set_ylabel('ACC2 (%)'); ax.legend(); ax.grid(True, alpha=0.3)
        ax.set_title('Validation ACC2')

        ax = axes[0, 1]
        ax.plot(epochs_list, metrics_dict.get('train_loss', []), 'r-s', label='train_loss', markersize=4)
        ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.legend(); ax.grid(True, alpha=0.3)
        ax.set_title('Training Loss')

        ax = axes[1, 0]
        ax.plot(epochs_list, metrics_dict.get('val_MAE', []), 'g-^', label='val_MAE', markersize=4)
        ax.set_xlabel('Epoch'); ax.set_ylabel('MAE'); ax.legend(); ax.grid(True, alpha=0.3)
        ax.set_title('Validation MAE')

        ax = axes[1, 1]
        ax.plot(epochs_list, metrics_dict.get('val_Corr', []), 'm-d', label='val_Corr', markersize=4)
        ax.set_xlabel('Epoch'); ax.set_ylabel('Corr'); ax.legend(); ax.grid(True, alpha=0.3)
        ax.set_title('Validation Corr')

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'mosi_training_curves.png'), dpi=150, bbox_inches='tight')
        plt.close()

    # 2. AWAF 权重分布
    if weights is not None and len(weights) > 0:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for idx, (ax, mod, col) in enumerate(zip(axes, ['Text', 'Audio', 'Vision'], ['blue', 'orange', 'green'])):
            ax.hist(weights[:, idx], bins=50, alpha=0.7, color=col, edgecolor='black')
            ax.axvline(x=weights[:, idx].mean(), color='red', linestyle='--', linewidth=2)
            ax.set_xlabel(f'w_{mod.lower()}'); ax.set_ylabel('Count')
            ax.set_title(f'AWAF {mod} Weight (mean={weights[:, idx].mean():.4f})')
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'mosi_awaf_weight_distribution.png'), dpi=150, bbox_inches='tight')
        plt.close()

    # 3. Gate 分布
    if gates is not None and len(gates) > 0:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(gates, bins=50, alpha=0.7, color='purple', edgecolor='black')
        ax.axvline(x=gates.mean(), color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Gate Value'); ax.set_ylabel('Count')
        ax.set_title(f'Gate Distribution (mean={gates.mean():.4f}, std={gates.std():.4f})')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'mosi_gate_distribution.png'), dpi=150, bbox_inches='tight')
        plt.close()

    # 4. Delta 分布
    if deltas is not None and len(deltas) > 0:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(deltas, bins=50, alpha=0.7, color='brown', edgecolor='black')
        ax.axvline(x=deltas.mean(), color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Effective Delta'); ax.set_ylabel('Count')
        ax.set_title(f'Effective Delta Distribution (mean={deltas.mean():.4f}, std={deltas.std():.4f})')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'mosi_delta_distribution.png'), dpi=150, bbox_inches='tight')
        plt.close()

    # 5. Text vs Final scatter
    if rtb is not None and final is not None and labels is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, preds, title in zip(axes, [rtb, final], ['Text Base', 'Final']):
            ax.scatter(labels, preds, alpha=0.3, s=10)
            ax.plot([-3, 3], [-3, 3], 'r--', linewidth=1)
            ax.set_xlabel('Label'); ax.set_ylabel('Prediction')
            ax.set_title(f'{title} vs Label')
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'mosi_text_vs_final_scatter.png'), dpi=150, bbox_inches='tight')
        plt.close()

    # 6. Confusion matrix (ACC2)
    if labels is not None and final is not None:
        from sklearn.metrics import confusion_matrix
        y_true_bin = (labels >= 0).astype(int)
        y_pred_bin = (final >= 0).astype(int)
        cm = confusion_matrix(y_true_bin, y_pred_bin)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap='Blues')
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=16,
                        color='white' if cm[i, j] > cm.max() / 2 else 'black')
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(['Neg', 'Pos']); ax.set_yticklabels(['Neg', 'Pos'])
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
        ax.set_title(f'Confusion Matrix (ACC2)')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'mosi_confusion_matrix.png'), dpi=150, bbox_inches='tight')
        plt.close()

    print(f'[PLOTS] 6 张图已保存到 {out_dir}')


# ================================================================
# Main training loop
# ================================================================
def main():
    args = parse_args()
    yc = load_config_yaml(args.config)
    t = yc.get('training', {})
    out_cfg = yc.get('output', {})

    SEED = t.get('seed', 42)
    EPOCHS = args.max_epochs or t.get('epochs', 30)
    LIMIT_BATCHES = args.limit_batches or (5 if args.smoke else 0)
    if args.smoke:
        EPOCHS = 1
        print(f'[SMOKE] 1 epoch, {LIMIT_BATCHES} batches')
    BATCH = t.get('batch_size', 4)
    ACCUM = t.get('grad_accum_steps', 4)
    LR = t.get('lr', 3e-5)
    LR_LORA = t.get('lr_text_lora', 3e-6)
    WEIGHT_DECAY = t.get('weight_decay', 0.03)
    PATIENCE = t.get('early_stopping_patience', 6)
    TEST_EVAL = t.get('test_eval', False)           # 是否在 best epoch 后 eval test
    TEST_FINAL_ONCE = t.get('test_final_once', True) # 是否训练结束后 test once
    VAL_ONLY = t.get('val_only', False)              # sweep 模式: 只看 val
    LAMBDA_ENTROPY = yc.get('model', {}).get('lambda_awaf_entropy', 0.0)

    DEVICE = args.device
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ================================================================
    # Data
    # ================================================================
    data_cfg = yc.get('data', {})
    DATASET_NAME = data_cfg.get('dataset', 'mosi')
    FORMAL_MODE = data_cfg.get('formal_mode', True)
    CSV_PATH = data_cfg.get('csv_path', f'data/{DATASET_NAME}/label.csv')
    FEATURE_ROOT = data_cfg.get('feature_root', f'data/features_strong_sequence_{DATASET_NAME}_v3_T40')

    print(f'[DATA] Loading {DATASET_NAME.upper()} (formal_mode={FORMAL_MODE})...')
    try:
        train_ds = TextFTMultimodalDataset(csv_path=CSV_PATH, feature_root=FEATURE_ROOT,
                                           split='train', formal_mode=FORMAL_MODE)
        val_ds = TextFTMultimodalDataset(csv_path=CSV_PATH, feature_root=FEATURE_ROOT,
                                         split='val', formal_mode=FORMAL_MODE)
        test_ds = TextFTMultimodalDataset(csv_path=CSV_PATH, feature_root=FEATURE_ROOT,
                                          split='test', formal_mode=FORMAL_MODE)
    except Exception as e:
        print(f'[ERROR] Failed to load {DATASET_NAME.upper()} dataset: {e}')
        raise
    print(f'  Train={len(train_ds)}  Val={len(val_ds)}  Test={len(test_ds)}')

    tl = DataLoader(train_ds, BATCH, shuffle=True, collate_fn=collate_textft)
    vl = DataLoader(val_ds, BATCH, shuffle=False, collate_fn=collate_textft)
    tl_test = DataLoader(test_ds, BATCH, shuffle=False, collate_fn=collate_textft)

    # ================================================================
    # Model
    # ================================================================
    config = build_config(yc, DEVICE)
    print(f'[MODEL] Building TextFTLoRAXLSTMAWAFResidual (mode={config.mode})...')
    print(f'  needs_text={config.needs_text}  needs_audio={config.needs_audio_branch}  needs_vision={config.needs_vision_branch}')
    print(f'  use_text_conf_residual={config.use_text_conf_residual}  freeze_text_base={config.freeze_text_base}')
    model = TextFTLoRAXLSTMAWAFResidual(config)
    model = model.to(DEVICE)
    print(f'  Params: {model.count_trainable()}')

    # P6AJ: Load initialization checkpoint (e.g., P6K weights)
    if args.init_checkpoint:
        print(f'[INIT] Loading checkpoint: {args.init_checkpoint}')
        init_sd = torch.load(args.init_checkpoint, map_location=DEVICE)
        model_sd = model.state_dict()
        loaded = 0
        skipped_shape = 0
        for k, v in init_sd.items():
            if k in model_sd and model_sd[k].shape == v.shape:
                model_sd[k].copy_(v)
                loaded += 1
            elif k in model_sd:
                skipped_shape += 1
        model.load_state_dict(model_sd)
        print(f'  Loaded {loaded} params, skipped {skipped_shape} (shape mismatch)')

    tokenizer = AutoTokenizer.from_pretrained(config.text_model_name)

    # Optimizer: 分组 lr (text_only mode has no residual params)
    if config.needs_text:
        lora_params = [p for n, p in model.roberta.named_parameters() if 'lora' in n.lower() and p.requires_grad]
        other_params = [p for p in model.collect_trainable_params() if p not in set(lora_params)]
    else:
        lora_params = []
        other_params = list(model.collect_trainable_params())
    param_groups = [
        {'params': lora_params, 'lr': LR_LORA},
        {'params': other_params, 'lr': LR},
    ]
    # Filter empty groups
    param_groups = [g for g in param_groups if g['params']]
    opt = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler(device='cuda')

    # ================================================================
    # Output dir
    # ================================================================
    ts = time.strftime('%Y%m%d_%H%M%S')
    out_root = out_cfg.get('root', 'outputs/P6H_repair')
    exp_name = out_cfg.get('exp_name', os.path.splitext(os.path.basename(args.config))[0])
    out_dir = os.path.join(out_root, f'{exp_name}_s{SEED}_{ts}')
    os.makedirs(out_dir, exist_ok=True)
    print(f'[OUTPUT] {out_dir}')

    # ================================================================
    # Train
    # ================================================================
    metrics_epoch = {'epoch': [], 'train_loss': [], 'val_ACC2': [], 'val_MAE': [], 'val_Corr': [],
                      'val_F1': [], 'awaf_entropy': [], 'gate_mean': [], 'delta_abs_mean': []}
    best_val_acc = 0.0
    best_epoch = 0
    no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        # --- Train ---
        model.train()
        total_loss = 0.0
        opt.zero_grad()
        pbar = tqdm(tl, desc=f'E{epoch:2d}', leave=False)
        for i, batch in enumerate(pbar):
            if LIMIT_BATCHES and i >= LIMIT_BATCHES:
                break
            out = model(batch)
            lbl = batch['label'].to(DEVICE)

            # Loss: L1 regression
            lr_loss = F.l1_loss(out['reg'], lbl)

            # Delta loss (mode-dependent)
            ld = torch.tensor(0.0, device=DEVICE)
            DELTA_LOSS_WEIGHT = t.get('delta_loss_weight', 1.0)

            if config.use_text_conf_residual and 'delta_loss' in out and out['delta_loss'] is not None:
                # P6I: text_confidence_residual provides its own delta_loss
                ld = out['delta_loss']
                loss = lr_loss + DELTA_LOSS_WEIGHT * ld
            elif 'effective_delta_reg' in out and 'reg_text_base' in out:
                # Old delta alignment
                td = lbl - out['reg_text_base'].detach()
                ld = F.smooth_l1_loss(out['effective_delta_reg'], td)
                loss = lr_loss + 0.2 * ld
            else:
                loss = lr_loss

            # AWAF entropy regularization
            if LAMBDA_ENTROPY > 0.0 and 'awaf_weights' in out:
                w = out['awaf_weights']
                awaf_entropy = model.awaf.compute_entropy(w).mean()
                loss = loss - LAMBDA_ENTROPY * awaf_entropy

            loss = loss / ACCUM
            scaler.scale(loss).backward()

            if (i + 1) % ACCUM == 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.collect_trainable_params(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
            total_loss += loss.item() * ACCUM
            pbar.set_postfix({'loss': f'{loss.item()*ACCUM:.4f}'})

        avg_loss = total_loss / min(len(tl), LIMIT_BATCHES or len(tl))

        # --- Val ---
        model.eval()
        vp_list, vl_list = [], []
        with torch.no_grad():
            for vi, batch in enumerate(vl):
                if LIMIT_BATCHES and vi >= LIMIT_BATCHES:
                    break
                out = model(batch)
                vp_list.append(out['reg'].cpu())
                vl_list.append(batch['label'].cpu())
        rp = torch.cat(vp_list)
        tg = torch.cat(vl_list)
        rs = torch.where(rp >= 0, 1.0, -1.0)
        m = compute_all_metrics(rp, rs, tg)

        # --- Diagnostics ---
        awaf_ent = 0.0
        gate_m = 0.0
        delta_abs_m = 0.0
        text_conf_m = 0.0
        with torch.no_grad():
            batch0 = next(iter(vl))
            out0 = model({k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch0.items()})
            if 'awaf_weights' in out0 and model.config.needs_awaf:
                awaf_ent = model.awaf.compute_entropy(out0['awaf_weights']).mean().item()
            if 'gate' in out0:
                gate_m = out0['gate'].mean().item()
            elif 'gate_reg' in out0:
                gate_m = out0['gate_reg'].mean().item()
            if 'effective_delta_reg' in out0:
                delta_abs_m = out0['effective_delta_reg'].abs().mean().item()
            elif 'delta' in out0:
                delta_abs_m = out0['delta'].abs().mean().item()
            if 'text_confidence' in out0:
                text_conf_m = out0['text_confidence'].mean().item()

        # --- Log ---
        extra = ''
        if config.use_text_conf_residual:
            extra = f'  conf={text_conf_m:.4f}  gate={gate_m:.4f}  |δ|={delta_abs_m:.4f}'
        elif config.needs_awaf:
            extra = f'  ent={awaf_ent:.4f}  gate={gate_m:.4f}  |δ|={delta_abs_m:.4f}'
        print(f'E{epoch:2d}: loss={avg_loss:.4f}  val_ACC2={m["ACC2_Non0"]:.2f}%  '
              f'val_MAE={m["MAE"]:.4f}  val_Corr={m["Corr"]:.4f}{extra}')

        # --- Record ---
        metrics_epoch['epoch'].append(epoch)
        metrics_epoch['train_loss'].append(avg_loss)
        metrics_epoch['val_ACC2'].append(m['ACC2_Non0'])
        metrics_epoch['val_MAE'].append(m['MAE'])
        metrics_epoch['val_Corr'].append(m['Corr'])
        metrics_epoch['val_F1'].append(m['F1_Non0'])
        metrics_epoch['awaf_entropy'].append(awaf_ent)
        metrics_epoch['gate_mean'].append(gate_m)
        metrics_epoch['delta_abs_mean'].append(delta_abs_m)

        # --- Early stopping ---
        if m['ACC2_Non0'] > best_val_acc:
            best_val_acc = m['ACC2_Non0']
            best_epoch = epoch
            no_improve = 0
            # Save best
            torch.save(model.state_dict(), os.path.join(out_dir, 'best_model.pth'))
        else:
            no_improve += 1

        # Save per-epoch CSV
        with open(os.path.join(out_dir, 'metrics_epoch.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            keys = list(metrics_epoch.keys())
            w.writerow(keys)
            for i in range(len(metrics_epoch['epoch'])):
                w.writerow([metrics_epoch[k][i] for k in keys])

        if PATIENCE > 0 and no_improve >= PATIENCE:
            print(f'[EARLY STOP] No improvement for {PATIENCE} epochs. Best: epoch={best_epoch} ACC2={best_val_acc:.2f}%')
            break
        elif PATIENCE == 0:
            pass  # no early stopping — run all epochs

    print(f'\n=== Training done: best_val_ACC2={best_val_acc:.2f}% at epoch {best_epoch} ===')

    # ================================================================
    # Test evaluation (only if not val_only)
    # ================================================================
    if VAL_ONLY:
        print('[SWEEP MODE] val_only=True, skipping test eval.')
        # Still save sweep summary
        sweep_summary = {
            'config': args.config,
            'seed': SEED,
            'epochs_run': epoch,
            'best_epoch': best_epoch,
            'best_val_ACC2': best_val_acc,
        }
        json.dump(sweep_summary, open(os.path.join(out_dir, 'sweep_summary.json'), 'w'))
        print(f'[DONE] Sweep summary saved to {out_dir}')
        return

    # Load best model for test
    if os.path.exists(os.path.join(out_dir, 'best_model.pth')):
        model.load_state_dict(torch.load(os.path.join(out_dir, 'best_model.pth'), map_location=DEVICE))
        print(f'[TEST] Loaded best model (epoch {best_epoch})')

    model.eval()
    tp_list, tl_list = [], []
    aw_list, gv_list, rb_list, delta_list = [], [], [], []
    tconf_list, tunc_list = [], []  # P6I

    has_rb = config.needs_text
    has_aw = config.needs_awaf
    has_gate = 'gate' in (config.mode and ['text_confidence_residual'] or []) or config.use_uncertainty_gate

    with torch.no_grad():
        for batch in tqdm(tl_test, desc='Test'):
            out = model(batch)
            tp_list.append(out['reg'].cpu())
            tl_list.append(batch['label'].cpu())
            if has_rb and 'reg_text_base' in out:
                rb_list.append(out['reg_text_base'].cpu())
            if has_aw and 'awaf_weights' in out:
                aw_list.append(out['awaf_weights'].cpu())
            if 'gate' in out:
                gv_list.append(out['gate'].cpu())
            elif 'gate_reg' in out:
                gv_list.append(out['gate_reg'].cpu())
            if 'effective_delta_reg' in out:
                delta_list.append(out['effective_delta_reg'].cpu())
            elif 'delta' in out:
                delta_list.append(out['delta'].cpu())
            if 'text_confidence' in out:
                tconf_list.append(out['text_confidence'].cpu())
            if 'text_uncertainty' in out:
                tunc_list.append(out['text_uncertainty'].cpu())

    rp = torch.cat(tp_list)
    tg = torch.cat(tl_list)
    rs = torch.where(rp >= 0, 1.0, -1.0)
    m_final = compute_all_metrics(rp, rs, tg)

    rb_np = None
    m_base = None
    if rb_list:
        rb = torch.cat(rb_list)
        rb_np = rb.numpy().flatten()
        m_base = compute_all_metrics(rb, rs, tg)

    aw_np = np.zeros((len(tg), 3)) if aw_list else None
    if aw_list:
        aw = torch.cat(aw_list)
        aw_np = aw.numpy()

    gv_np = np.zeros(len(tg)) if gv_list else None
    if gv_list:
        gv = torch.cat(gv_list)
        gv_np = gv.numpy().flatten()

    ed_np = np.zeros(len(tg)) if delta_list else None
    if delta_list:
        ed = torch.cat(delta_list)
        ed_np = ed.numpy().flatten()

    tconf_np = np.zeros(len(tg)) if tconf_list else None
    if tconf_list:
        tconf = torch.cat(tconf_list)
        tconf_np = tconf.numpy().flatten()

    # ================================================================
    # Print & save results
    # ================================================================
    residual_gain = 0.0
    if m_base:
        residual_gain = m_final['ACC2_Non0'] - m_base['ACC2_Non0']

    print(f'\n{"="*60}')
    print(f'P6I MOSI s{SEED} mode={config.mode} — Best epoch: {best_epoch}')
    if m_base:
        print(f'  Text-base ACC2: {m_base["ACC2_Non0"]:.2f}%')
    print(f'  Final    ACC2: {m_final["ACC2_Non0"]:.2f}%')
    if m_base:
        print(f'  Residual GAIN: {residual_gain:+.2f}%')
    print(f'  MAE: {m_final["MAE"]:.4f}  Corr: {m_final["Corr"]:.4f}  ACC7: {m_final["ACC7"]:.2f}%')
    print(f'  F1_Non0: {m_final["F1_Non0"]:.2f}%')
    if aw_np is not None and config.needs_awaf:
        print(f'  AWAF: w_t={aw_np[:,0].mean():.4f} w_a={aw_np[:,1].mean():.4f} w_v={aw_np[:,2].mean():.4f}')
    if gv_np is not None:
        print(f'  Gate mean: {gv_np.mean():.4f}')
    if ed_np is not None:
        print(f'  |δ| mean: {np.abs(ed_np).mean():.4f}')
    if tconf_np is not None:
        print(f'  Text confidence mean: {tconf_np.mean():.4f}')
    print(f'{"="*60}')

    # Save results JSON
    result = {
        'config': args.config, 'seed': SEED, 'mode': config.mode,
        'epochs': epoch, 'best_epoch': best_epoch, 'best_val_ACC2': best_val_acc,
        'final_ACC2': m_final['ACC2_Non0'], 'F1_Non0': m_final['F1_Non0'],
        'MAE': m_final['MAE'], 'Corr': m_final['Corr'], 'ACC7': m_final['ACC7'],
        'trainable_M': model.count_trainable()['trainable_M'],
    }
    if m_base:
        result['text_base_ACC2'] = m_base['ACC2_Non0']
        result['residual_gain'] = residual_gain
    if aw_np is not None:
        result.update({
            'awaf_w_t': float(aw_np[:,0].mean()), 'awaf_w_a': float(aw_np[:,1].mean()),
            'awaf_w_v': float(aw_np[:,2].mean()),
        })
    if gv_np is not None:
        result['gate_mean'] = float(gv_np.mean())
        result['gate_std'] = float(gv_np.std())
    if ed_np is not None:
        result['delta_abs_mean'] = float(np.abs(ed_np).mean())
    if tconf_np is not None:
        result['text_confidence_mean'] = float(tconf_np.mean())
    json.dump(result, open(os.path.join(out_dir, 'result.json'), 'w'), indent=2)

    # Save per-sample CSVs (mode-dependent)
    sample_ids = [f'sample_{i}' for i in range(len(tg))]
    rp_np = rp.numpy().flatten()
    tg_np = tg.numpy().flatten()

    if rb_np is not None:
        # P6AG: save predictions even without AWAF weights (text_anchored mode)
        _aw_np = aw_np if aw_np is not None else np.zeros((len(tg_np), 3))
        save_predictions_csv(
            os.path.join(out_dir, 'predictions_test.csv'),
            sample_ids, tg_np, rb_np, rp_np,
            ed_np if ed_np is not None else np.zeros_like(rp_np),
            gv_np if gv_np is not None else np.zeros_like(rp_np),
            _aw_np,
        )
        save_group_error_csv(
            os.path.join(out_dir, 'group_error_analysis.csv'),
            sample_ids, tg_np, rb_np, rp_np, _aw_np,
        )

    if aw_np is not None:
        with open(os.path.join(out_dir, 'awaf_weights_test.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['sample_id', 'w_t', 'w_a', 'w_v'])
            for i in range(len(sample_ids)):
                w.writerow([sample_ids[i], float(aw_np[i,0]), float(aw_np[i,1]), float(aw_np[i,2])])

    # Save text_confidence CSV (P6I)
    if tconf_np is not None:
        with open(os.path.join(out_dir, 'text_confidence_test.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['sample_id', 'label', 'final_pred', 'text_base_pred', 'text_confidence',
                         'text_uncertainty', 'gate', 'delta'])
            for i in range(len(sample_ids)):
                w.writerow([sample_ids[i], float(tg_np[i]), float(rp_np[i]),
                             float(rb_np[i]) if rb_np is not None else '',
                             float(tconf_np[i]), float(tunc_np[i]) if tunc_np is not None else '',
                             float(gv_np[i]) if gv_np is not None else '',
                             float(ed_np[i]) if ed_np is not None else ''])

    # Save plots (only when data available)
    if rb_np is not None:
        save_plots(out_dir, metrics_epoch['epoch'], metrics_epoch,
                   rb_np, rp_np, tg_np,
                   aw_np if aw_np is not None else np.zeros((len(tg_np), 3)),
                   gv_np if gv_np is not None else np.zeros(len(tg_np)),
                   ed_np if ed_np is not None else np.zeros(len(tg_np)))

    print(f'\n[DONE] All outputs saved to {out_dir}')


if __name__ == '__main__':
    main()
