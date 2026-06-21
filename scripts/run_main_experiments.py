#!/usr/bin/env python
"""
Unified Main Experiment Runner — MOSI + MOSEI, train + eval, save all artifacts.

Usage:
  python scripts/run_main_experiments.py --dataset mosi --device cuda
  python scripts/run_main_experiments.py --dataset mosei --device cuda
  python scripts/run_main_experiments.py --dataset all --device cuda

Output per experiment:
  outputs/main_experiments/<dataset>/<model_name>_s<seed>/
    config.yaml, command.txt, train.log, metrics_epoch.csv,
    metrics_best.json, result.json, best_model.pth, last_model.pth,
    predictions_test.csv, awaf_weights_test.csv, loss_curve.png
"""
import sys, os, argparse, subprocess, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EXPERIMENTS = {
    'mosi': [
        {'name': 'canonical_awaf_slstm', 'config': 'configs/references/mosi_conservative/text_audio_conservative_s42.yaml',
         'script': 'scripts/train_textft_lora_mainline.py'},
    ],
    'mosei': [
        {'name': 'canonical_awaf_slstm', 'config': 'configs/canonical/mosei/control_awaf_slstm_s42.yaml',
         'script': 'scripts/train_textft_lora_mainline.py'},
        # Baselines (optional; comment out if not needed)
        {'name': 'baseline_misa_lite', 'config': 'configs/baselines/mosei/misa_lite_s42.yaml',
         'script': 'scripts/train_baseline_lite.py'},
        {'name': 'baseline_mult_lite', 'config': 'configs/baselines/mosei/mult_lite_s42.yaml',
         'script': 'scripts/train_baseline_lite.py'},
    ],
}

def run(cmd, timeout=86400):
    print(f'  CMD: {" ".join(cmd)}')
    return subprocess.run(cmd, timeout=timeout)

def main():
    p = argparse.ArgumentParser(description='Unified Main Experiment Runner')
    p.add_argument('--dataset', required=True, choices=['mosi', 'mosei', 'all'])
    p.add_argument('--device', default='cuda')
    p.add_argument('--smoke', action='store_true', help='Smoke test: 1 epoch, 5 batches')
    args = p.parse_args()

    datasets = ['mosi', 'mosei'] if args.dataset == 'all' else [args.dataset]

    results = {}
    for ds in datasets:
        print(f'\n{"="*60}')
        print(f'DATASET: {ds.upper()}')
        print(f'{"="*60}')
        for exp in EXPERIMENTS.get(ds, []):
            print(f'\n--- {exp["name"]} ---')
            cmd = ['python', exp['script'], '--config', exp['config'], '--device', args.device]
            if args.smoke:
                cmd.append('--smoke')
            start = time.time()
            result = run(cmd)
            elapsed = time.time() - start
            status = 'PASS' if result.returncode == 0 else 'FAIL'
            print(f'  {status} ({elapsed:.0f}s)')
            results[f'{ds}/{exp["name"]}'] = status

    print(f'\n{"="*60}')
    print('SUMMARY')
    for k, v in results.items():
        print(f'  {k}: {v}')
    passed = sum(1 for v in results.values() if v == 'PASS')
    print(f'  Total: {passed}/{len(results)} passed')

if __name__ == '__main__':
    main()
