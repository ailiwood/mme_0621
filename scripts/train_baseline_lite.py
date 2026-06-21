#!/usr/bin/env python
"""P6O: Baseline-Lite formal training with full metrics and checkpointing."""
import sys, os, json, time, csv, argparse, yaml
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.textft_multimodal_dataset import TextFTMultimodalDataset, collate_textft
from utils.metrics import compute_all_metrics


# Collapse detection thresholds
COLLAPSE_GUARD = {
    'pred_std_min': 1e-6,       # prediction std below this → collapse
    'positive_ratio_min': 0.01,  # <1% positive → near all-neg collapse
    'positive_ratio_max': 0.99,  # >99% positive → near all-pos collapse
    'f1_non0_min': 0.01,         # F1_Non0 < 0.01 for 2 epochs → collapse
    'nan_loss_trigger': True,    # Any NaN loss → immediate stop
    'consecutive_epochs': 2,     # How many consecutive bad epochs before stop
    'identical_metric_trigger': True,  # If ACC2 identical for 3 epochs → dead model
}

MODEL_MAP = {
    'tfn_lite': 'models.baselines.tfn_lite.TFNLite',
    'lmf_lite': 'models.baselines.lmf_lite.LMFLite',
    'mult_lite': 'models.baselines.mult_lite.MulTLite',
    'self_mm_lite': 'models.baselines.self_mm_lite.SelfMMLite',
    'selfmm_lite': 'models.baselines.self_mm_lite.SelfMMLite',
    'misa_lite': 'models.baselines.misa_lite.MISALite',
    'mmim_lite': 'models.baselines.mmim_lite.MMIMLite',
    'mlcl_lite': 'models.baselines.mlcl_lite.MLCLLite',
    'dlf_lite': 'models.baselines.dlf_lite.DLFLite',
}


def _check_collapse(reg_preds, epoch, model_name, prev_check=None):
    """Check regression predictions for collapse signals.

    Returns:
        (is_collapsed: bool, reasons: list, info: dict, debug_data: dict)
    """
    rp = reg_preds  # numpy array
    reasons = []
    info = {'pred_std': float(np.std(rp)), 'positive_ratio': float((rp >= 0).mean()),
            'nan_count': int(np.isnan(rp).sum()), 'total': len(rp)}

    if info['nan_count'] > 0:
        reasons.append(f'NAN_COUNT={info["nan_count"]}/{info["total"]}')
    if info['pred_std'] < COLLAPSE_GUARD['pred_std_min']:
        reasons.append(f'PRED_STD={info["pred_std"]:.2e}<{COLLAPSE_GUARD["pred_std_min"]}')
    if info['positive_ratio'] <= COLLAPSE_GUARD['positive_ratio_min']:
        reasons.append(f'POS_RATIO={info["positive_ratio"]:.4f}≤{COLLAPSE_GUARD["positive_ratio_min"]}')
    if info['positive_ratio'] >= COLLAPSE_GUARD['positive_ratio_max']:
        reasons.append(f'POS_RATIO={info["positive_ratio"]:.4f}≥{COLLAPSE_GUARD["positive_ratio_max"]}')

    # Check identical ACC2 over epochs
    if prev_check and abs(info.get('acc2', 0) - prev_check.get('acc2', 0)) < 1e-6:
        reasons.append('IDENTICAL_ACC2')

    return len(reasons) > 0, reasons, info, {}


def import_model(name):
    mod_path, cls_name = MODEL_MAP[name].rsplit('.', 1)
    import importlib
    return getattr(importlib.import_module(mod_path), cls_name)


def inject_text_feature(batch, roberta, device, cached_features=None, split='train'):
    """Add RoBERTa CLS feature to batch (cached or online)."""
    if cached_features and split in cached_features:
        feats, id_to_idx = cached_features[split]
        sample_ids = batch.get('id', [])
        indices = []
        for sid in sample_ids:
            if sid in id_to_idx:
                indices.append(id_to_idx[sid])
            else:
                indices.append(0)  # fallback (shouldn't happen if cache is complete)
        batch['roberta_cls'] = torch.from_numpy(feats[indices]).float().to(device)
    elif roberta is not None:
        with torch.no_grad():
            ids = batch['input_ids'].to(device)
            am = batch['attention_mask'].to(device)
            out = roberta(input_ids=ids, attention_mask=am)
            batch['text_feature'] = out.last_hidden_state[:, 0, :]
    return batch


def evaluate(model, loader, device, roberta=None, cached_features=None, split='val'):
    """Return predictions, labels, and compute_all_metrics dict."""
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = inject_text_feature(batch, roberta, device, cached_features, split)
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            out = model(batch)
            preds.append(out['reg'].cpu())
            labels.append(batch['label'].cpu())
    rp = torch.cat(preds)
    tg = torch.cat(labels).squeeze(-1)
    rs = torch.where(rp >= 0, 1.0, -1.0)
    m = compute_all_metrics(rp, rs, tg)
    return rp.numpy(), tg.numpy(), m


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--max_epochs', type=int, default=None)
    p.add_argument('--limit_batches', type=int, default=0)
    p.add_argument('--debug_shapes', action='store_true')
    p.add_argument('--quick_test', action='store_true')
    args = p.parse_args()

    with open(args.config) as f:
        yc = yaml.safe_load(f)
    m = yc.get('model', {})
    t = yc.get('training', {})
    d = yc.get('data', {})
    o = yc.get('output', {})

    DEVICE = args.device
    SEED = t.get('seed', 42)
    EPOCHS = args.max_epochs or t.get('epochs', 50)
    MIN_EPOCHS = t.get('min_epochs', 30)
    PATIENCE = t.get('early_stopping_patience', 10)
    BATCH = t.get('batch_size', 16)
    LR = t.get('lr', 1e-4)
    WD = t.get('weight_decay', 0.01)
    MODE = m.get('modality_mode', 'text_audio_vision')
    MODEL_NAME = m.get('model_name', 'tfn_lite')
    TEST_ONCE = t.get('test_final_once', True)
    torch.manual_seed(SEED); np.random.seed(SEED)

    # Data
    ds_name = d.get('dataset', 'mosi')
    csv_path = d.get('csv_path', f'data/{ds_name}/label.csv')
    feat_root = d.get('feature_root', f'data/features_strong_sequence_{ds_name}_v3_T40')
    train_subset = t.get('train_subset', 0)  # 0 = use all, N = use first N samples
    print(f'[DATA] {ds_name.upper()} from {csv_path}')
    train_ds = TextFTMultimodalDataset(csv_path=csv_path, feature_root=feat_root, split='train')
    if train_subset > 0:
        from torch.utils.data import Subset
        indices = list(range(min(train_subset, len(train_ds))))
        train_ds = Subset(train_ds, indices)
        print(f'  Train subset: {len(train_ds)} samples (overfit debug mode)')
    val_ds = TextFTMultimodalDataset(csv_path=csv_path, feature_root=feat_root, split='val')
    test_ds = TextFTMultimodalDataset(csv_path=csv_path, feature_root=feat_root, split='test')
    print(f'  Train={len(train_ds)} Val={len(val_ds)} Test={len(test_ds)}')
    tl = DataLoader(train_ds, BATCH, shuffle=True, collate_fn=collate_textft)
    vl = DataLoader(val_ds, BATCH, shuffle=False, collate_fn=collate_textft)
    tlt = DataLoader(test_ds, BATCH, shuffle=False, collate_fn=collate_textft)

    # Model
    use_pretrained_text = m.get('use_pretrained_text', False)
    cache_dir = m.get('roberta_cache_dir', 'data/processed/mosei_full/roberta_cache')
    roberta = None
    cached_features = {}
    if use_pretrained_text:
        import json
        # Load cached features if available
        for split in ['train', 'valid', 'test']:
            feat_path = os.path.join(cache_dir, f'{split}_roberta_cls.npy')
            ids_path = os.path.join(cache_dir, f'{split}_ids.json')
            if os.path.exists(feat_path) and os.path.exists(ids_path):
                feats = np.load(feat_path)
                with open(ids_path) as f:
                    ids = json.load(f)
                id_to_idx = {sid: i for i, sid in enumerate(ids)}
                cached_features[split] = (feats, id_to_idx)
                print(f'[CACHE] Loaded {split}: {feats.shape}')
        if cached_features:
            print(f'[CACHE] Using cached RoBERTa features from {cache_dir}')
        else:
            # Fallback: online RoBERTa
            from transformers import AutoModel
            print('[MODEL] Loading RoBERTa-large for text features (frozen)...')
            roberta = AutoModel.from_pretrained('roberta-large').to(DEVICE)
            for p in roberta.parameters():
                p.requires_grad = False
            roberta.eval()

    model_cls = import_model(MODEL_NAME)
    model = model_cls({**m, 'modality_mode': MODE}).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'[MODEL] {MODEL_NAME} mode={MODE} params={n_params/1e6:.2f}M pretrained_text={use_pretrained_text}')

    if args.debug_shapes:
        batch = next(iter(tl))
        batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        out = model(batch)
        print(f'[SHAPES] reg={out["reg"].shape}')
        if args.quick_test:
            return

    # Output dir
    ts = time.strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(o.get('root', 'outputs/P6O'), f'{MODEL_NAME}_s{SEED}_{ts}')
    os.makedirs(out_dir, exist_ok=True)
    yaml.dump(yc, open(os.path.join(out_dir, 'config.yaml'), 'w'))
    with open(os.path.join(out_dir, 'command.txt'), 'w') as f:
        f.write(' '.join(sys.argv))

    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    # Training state
    metrics_epoch = {
        'epoch': [], 'train_loss': [],
        'ACC2_Non0': [], 'F1_Non0': [], 'ACC2_Has0': [], 'F1_Has0': [],
        'MAE': [], 'Corr': [], 'ACC7': [],
    }
    best_val = {'ACC2_Non0': 0.0, 'MAE': 999, 'Corr': -999}
    best_epoch = 0
    best_state = None
    no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        total_loss = 0.0
        for i, batch in enumerate(tqdm(tl, desc=f'E{epoch}', leave=False)):
            if args.limit_batches and i >= args.limit_batches:
                break
            batch = inject_text_feature(batch, roberta, DEVICE, cached_features, 'train')
            batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            out = model(batch)
            lbl = batch['label'].squeeze(-1)
            loss = out.get('loss_terms', {}).get('loss_total', nn.L1Loss()(out['reg'], lbl))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()

        avg_loss = total_loss / min(len(tl), args.limit_batches or len(tl))

        # Val
        val_preds, val_labels, val_m = evaluate(model, vl, DEVICE, roberta, cached_features, 'valid')

        # === Anti-collapse guard ===
        guard_info = {'acc2': val_m['ACC2_Non0']}
        guard_data = {}
        try:
            is_collapsed, reasons, guard_info, guard_data = _check_collapse(
                val_preds, epoch, MODEL_NAME,
                prev_check=getattr(main, '_prev_guard', None))
            main._prev_guard = guard_info
        except Exception:
            pass  # Guard errors are non-fatal

        # Collapse triggers
        collapse_stop = False
        if np.isnan(avg_loss):
            collapse_stop = True
            guard_info['collapse_reason'] = 'train_loss_NaN'
        if val_m['MAE'] is not None and np.isnan(val_m['MAE']):
            collapse_stop = True
            guard_info['collapse_reason'] = guard_info.get('collapse_reason', '') + 'val_MAE_NaN'
        if val_m['Corr'] is not None and np.isnan(val_m['Corr']):
            collapse_stop = True
            guard_info['collapse_reason'] = guard_info.get('collapse_reason', '') + 'val_Corr_NaN'

        if is_collapsed:
            consec = guard_info.get('consecutive', 0) + 1
            guard_info['consecutive'] = consec
            if consec >= COLLAPSE_GUARD['consecutive_epochs']:
                collapse_stop = True
                guard_info['collapse_reason'] = guard_info.get('collapse_reason', '') + ';'.join(reasons)
        else:
            guard_info['consecutive'] = 0

        if collapse_stop:
            print(f'\n[COLLAPSE GUARD] EPOCH {epoch}: {guard_info.get("collapse_reason", "unknown")}')
            print(f'  Pred std={guard_info.get("pred_std", -1):.6f} pos_ratio={guard_info.get("positive_ratio", -1):.4f}')
            print(f'  NaN count={guard_info.get("nan_count", -1)} reasons={reasons}')
            # Save debug batch
            try:
                dbg_batch = next(iter(tl))
                dbg_batch = inject_text_feature(dbg_batch, roberta, DEVICE, cached_features, 'train')
                torch.save({k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in dbg_batch.items()},
                           os.path.join(out_dir, 'collapse_debug_batch.pt'))
                print(f'  Debug batch saved to collapse_debug_batch.pt')
            except Exception:
                pass
            # Write error log
            with open(os.path.join(out_dir, 'error.log'), 'w') as ef:
                ef.write(f'COLLAPSE_GUARD triggered at epoch {epoch}\n')
                ef.write(f'Reason: {guard_info.get("collapse_reason", "unknown")}\n')
                ef.write(f'Pred std={guard_info.get("pred_std", -1):.6f}\n')
                ef.write(f'Pos ratio={guard_info.get("positive_ratio", -1):.4f}\n')
                ef.write(f'NaN count={guard_info.get("nan_count", -1)}\n')
                ef.write(f'Reasons: {reasons}\n')
                metric_keys = ['ACC2_Non0', 'F1_Non0', 'MAE', 'Corr']
                ef.write(f'Metrics: {json.dumps({k: val_m.get(k, 0) for k in metric_keys})}\n')
            raise RuntimeError(f'COLLAPSE_GUARD: Training aborted at epoch {epoch}. '
                               f'Reason: {guard_info.get("collapse_reason", "unknown")}')
        # === End anti-collapse guard ===

        # Record
        metrics_epoch['epoch'].append(epoch)
        metrics_epoch['train_loss'].append(avg_loss)
        for k in ['ACC2_Non0', 'F1_Non0', 'ACC2_Has0', 'F1_Has0', 'MAE', 'Corr', 'ACC7']:
            metrics_epoch[k].append(val_m[k])

        # Best checkpoint: ACC2_Non0 primary, MAE tiebreaker
        is_better = False
        if val_m['ACC2_Non0'] > best_val['ACC2_Non0']:
            is_better = True
        elif val_m['ACC2_Non0'] == best_val['ACC2_Non0'] and val_m['MAE'] < best_val['MAE']:
            is_better = True

        if is_better:
            best_val = {k: val_m[k] for k in best_val}
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        print(f'E{epoch:2d}: loss={avg_loss:.4f}  '
              f'ACC2_Non0={val_m["ACC2_Non0"]:.2f}%  MAE={val_m["MAE"]:.4f}  '
              f'Corr={val_m["Corr"]:.4f}  best={best_epoch}({best_val["ACC2_Non0"]:.2f}%)')

        # Save per-epoch CSV
        with open(os.path.join(out_dir, 'metrics_epoch.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            keys = list(metrics_epoch.keys())
            w.writerow(keys)
            for i in range(len(metrics_epoch['epoch'])):
                w.writerow([metrics_epoch[k][i] for k in keys])

        # Early stopping
        if PATIENCE > 0 and no_improve >= PATIENCE and epoch >= MIN_EPOCHS:
            print(f'[EARLY STOP] epoch={epoch} >= {MIN_EPOCHS}, patience={PATIENCE}')
            break

    # Save best and last
    if best_state:
        torch.save(best_state, os.path.join(out_dir, 'best_model.pth'))
    torch.save(model.state_dict(), os.path.join(out_dir, 'last_model.pth'))

    # Test final once with best model
    if TEST_ONCE and best_state:
        model.load_state_dict(best_state)
    test_preds, test_labels, test_m = evaluate(model, tlt, DEVICE, roberta, cached_features, 'test')

    # Save predictions
    with open(os.path.join(out_dir, 'predictions_test.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['sample_id', 'label', 'prediction'])
        for i in range(len(test_labels)):
            w.writerow([f'sample_{i}', float(test_labels[i]), float(test_preds[i])])

    # Save metrics_best.json
    metrics_best = {
        'best_epoch': best_epoch,
        'monitor_metric': 'ACC2_Non0',
        'best_val': {k: best_val[k] for k in best_val},
        'test_at_best': {k: test_m[k] for k in ['ACC2_Non0', 'F1_Non0', 'ACC2_Has0', 'F1_Has0', 'MAE', 'Corr', 'ACC7']},
    }
    json.dump(metrics_best, open(os.path.join(out_dir, 'metrics_best.json'), 'w'), indent=2)

    # Save result.json
    result = {
        'phase': 'P6O', 'model': MODEL_NAME, 'dataset': ds_name, 'modalities': MODE,
        'seed': SEED, 'epochs_run': epoch, 'best_epoch': best_epoch,
        'params_M': n_params / 1e6,
        **{k: test_m[k] for k in ['ACC2_Non0', 'F1_Non0', 'ACC2_Has0', 'F1_Has0', 'MAE', 'Corr', 'ACC7']},
        'best_val_ACC2_Non0': best_val['ACC2_Non0'],
    }
    json.dump(result, open(os.path.join(out_dir, 'result.json'), 'w'), indent=2)

    # Save loss curve plot
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(metrics_epoch['epoch'], metrics_epoch['train_loss'], 'r-', label='Train Loss')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.legend(); ax.grid(True, alpha=0.3)
        ax.set_title(f'{MODEL_NAME} s{SEED} Training Loss')

        ax2 = ax.twinx()
        ax2.plot(metrics_epoch['epoch'], metrics_epoch['ACC2_Non0'], 'b-', label='Val ACC2_Non0')
        ax2.set_ylabel('ACC2_Non0 (%)'); ax2.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'loss_curve.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f'[PLOT] loss_curve.png saved')
    except ImportError:
        print('[WARN] matplotlib not available, skipping plot')

    print(f'\n=== P6O {MODEL_NAME} s{SEED} ===')
    print(f'  Best epoch: {best_epoch}  val_ACC2_Non0={best_val["ACC2_Non0"]:.2f}%')
    print(f'  Test ACC2_Non0={test_m["ACC2_Non0"]:.2f}%  F1={test_m["F1_Non0"]:.2f}%')
    print(f'  MAE={test_m["MAE"]:.4f}  Corr={test_m["Corr"]:.4f}  ACC7={test_m["ACC7"]:.2f}%')
    print(f'  Saved: {out_dir}')


if __name__ == '__main__':
    main()
