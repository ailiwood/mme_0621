#!/usr/bin/env python
"""
Unified Ablation Experiment Runner — all fusion/encoder variants, saves best pth + results.

Usage:
  python scripts/run_ablation_experiments.py --dataset mosei --device cuda
  python scripts/run_ablation_experiments.py --dataset mosei --device cuda --smoke

Ablation matrix (MOSEI):
  control_awaf_slstm, fusion_mean, fusion_gated, awaf_no_interaction, encoder_lstm

All variants use same training budget, seed=42, official split.
Each saves: best_model.pth, predictions_test.csv, result.json, awaf_weights_test.csv.
"""
import sys, os, argparse, subprocess, time, json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ABLATION_MATRIX = {
    'mosei': [
        ('control_awaf_slstm', 'configs/canonical/mosei/control_awaf_slstm_s42.yaml'),
        ('fusion_mean',        'configs/canonical/mosei/fusion_mean_s42.yaml'),
        ('fusion_gated',       'configs/canonical/mosei/fusion_gated_s42.yaml'),
        ('awaf_no_interaction','configs/canonical/mosei/awaf_no_interaction_s42.yaml'),
        ('encoder_lstm',       'configs/canonical/mosei/encoder_lstm_s42.yaml'),
    ],
}

SCRIPT = 'scripts/train_textft_lora_mainline.py'

def run(cmd, timeout=86400):
    print(f'  {" ".join(cmd)}')
    return subprocess.run(cmd, timeout=timeout)

def main():
    p = argparse.ArgumentParser(description='Unified Ablation Experiment Runner')
    p.add_argument('--dataset', required=True, choices=['mosei'])
    p.add_argument('--device', default='cuda')
    p.add_argument('--smoke', action='store_true', help='Smoke: 1 epoch, 5 batches')
    p.add_argument('--start-from', type=int, default=0, help='Start from index N')
    args = p.parse_args()

    matrix = ABLATION_MATRIX.get(args.dataset, [])
    results = {}

    for i, (name, config) in enumerate(matrix[args.start_from:]):
        print(f'\n{"="*60}')
        print(f'ABLATION [{i+1+args.start_from}/{len(matrix)}]: {name}')
        print(f'{"="*60}')

        cmd = ['python', SCRIPT, '--config', config, '--device', args.device]
        if args.smoke:
            cmd.append('--smoke')

        start = time.time()
        result = run(cmd)
        elapsed = time.time() - start
        status = 'PASS' if result.returncode == 0 else 'FAIL'
        print(f'  {status} ({elapsed:.0f}s)')
        results[name] = status

        if status == 'FAIL':
            print(f'  WARNING: {name} failed, continuing with next...')

    print(f'\n{"="*60}')
    print('ABLATION SUMMARY')
    for k, v in results.items():
        print(f'  {k}: {v}')
    passed = sum(1 for v in results.values() if v == 'PASS')
    print(f'  Total: {passed}/{len(results)} passed')

if __name__ == '__main__':
    main()
