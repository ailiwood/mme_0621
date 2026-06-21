"""
tests/test_metrics_mosei_non0.py — Metrics unit tests covering MOSEI edge cases.

Covers:
  1. Normal pos/neg labels
  2. Labels with 0
  3. All-positive predictions
  4. All-negative predictions
  5. NaN predictions should fail-fast
  6. Empty Non0 should return 0, not NaN
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from utils.metrics import (
    compute_mae, compute_corr, compute_acc2_non0, compute_f1_non0,
    compute_acc2_has0, compute_f1_has0, compute_acc7, compute_all_metrics,
)


def test_normal_pos_neg():
    """Test 1: Normal positive and negative labels."""
    targets = np.array([2.0, -1.0, 1.0, -2.0, 0.5, -0.5])
    preds = np.array([1.5, -0.8, 0.8, -1.5, 0.3, -0.3])
    logits = np.where(preds >= 0, 1.0, -1.0)

    m = compute_all_metrics(preds, logits, targets)
    assert m['MAE'] > 0, f'MAE should be >0, got {m["MAE"]}'
    assert m['Corr'] > 0, f'Corr should be >0, got {m["Corr"]}'
    assert m['ACC2_Non0'] > 0, f'ACC2_Non0 should be >0'
    assert m['F1_Non0'] > 0, f'F1_Non0 should be >0'
    print(f'TEST1 PASS: normal pos/neg — MAE={m["MAE"]:.4f} Corr={m["Corr"]:.4f} ACC2_Non0={m["ACC2_Non0"]:.1f}%')


def test_labels_with_zero():
    """Test 2: Labels containing 0 (should be excluded from Non0 metrics)."""
    targets = np.array([2.0, 0.0, -1.0, 0.0, 1.0, 0.0])
    preds = np.array([1.5, 0.0, -0.8, 0.0, 0.8, 0.0])
    logits = np.array([1.0, 0.0, -1.0, 0.0, 1.0, 0.0])

    acc2_n0 = compute_acc2_non0(logits, targets)
    f1_n0 = compute_f1_non0(logits, targets)
    acc2_h0 = compute_acc2_has0(logits, targets)

    assert acc2_n0 == 100.0, f'ACC2_Non0 should be 100%, got {acc2_n0}'
    assert f1_n0 == 100.0, f'F1_Non0 should be 100%, got {f1_n0}'
    assert acc2_h0 == 100.0, f'ACC2_Has0 should be 100%, got {acc2_h0}'
    print(f'TEST2 PASS: labels with zero — ACC2_Non0={acc2_n0:.1f}% F1_Non0={f1_n0:.1f}% ACC2_Has0={acc2_h0:.1f}%')


def test_all_positive_pred():
    """Test 3: Model predicts all positive, labels are mixed."""
    targets = np.array([2.0, -1.0, 1.0, -2.0, 0.5, -0.5])
    preds = np.ones(6)
    logits = np.ones(6)

    m = compute_all_metrics(preds, logits, targets)
    # ACC2_Non0 should equal fraction of positive in Non0
    nz = targets != 0
    pos_nz = (targets[nz] >= 0).mean()
    assert abs(m['ACC2_Non0'] - pos_nz * 100) < 0.01
    assert m['F1_Non0'] > 0, f'F1 should be >0 with all-pos, got {m["F1_Non0"]}'
    assert not np.isnan(m['MAE']), 'MAE should not be NaN'
    print(f'TEST3 PASS: all-positive pred — ACC2_Non0={m["ACC2_Non0"]:.1f}% F1_Non0={m["F1_Non0"]:.1f}% MAE={m["MAE"]:.4f}')


def test_all_negative_pred():
    """Test 4: Model predicts all negative, labels are mixed."""
    targets = np.array([2.0, -1.0, 1.0, -2.0, 0.5, -0.5])
    preds = -np.ones(6)
    logits = -np.ones(6)

    m = compute_all_metrics(preds, logits, targets)
    assert not np.isnan(m['MAE']), 'MAE should not be NaN with all-neg'
    assert m['F1_Non0'] == 0.0, f'F1 should be 0 with all-neg (no pos predictions), got {m["F1_Non0"]}'
    print(f'TEST4 PASS: all-negative pred — ACC2_Non0={m["ACC2_Non0"]:.1f}% F1_Non0={m["F1_Non0"]:.1f}% MAE={m["MAE"]:.4f}')


def test_nan_prediction_fail_fast():
    """Test 5: NaN predictions should raise ValueError."""
    targets = np.array([2.0, -1.0, 1.0])
    preds = np.array([np.nan, np.nan, np.nan])
    logits = np.array([1.0, -1.0, 1.0])

    try:
        compute_all_metrics(preds, logits, targets)
        assert False, 'Should have raised ValueError for all-NaN predictions'
    except ValueError as e:
        assert 'NaN' in str(e), f'Error should mention NaN: {e}'
        print(f'TEST5 PASS: NaN prediction fail-fast — {e}')


def test_empty_non0():
    """Test 6: All labels are 0 (empty Non0) — should return 0, not NaN."""
    targets = np.zeros(10)
    preds = np.random.randn(10)
    logits = np.where(preds >= 0, 1.0, -1.0)

    acc2_n0 = compute_acc2_non0(logits, targets)
    f1_n0 = compute_f1_non0(logits, targets)

    assert acc2_n0 == 0.0, f'ACC2_Non0 with all-zero labels should be 0, got {acc2_n0}'
    assert f1_n0 == 0.0, f'F1_Non0 with all-zero labels should be 0, got {f1_n0}'
    print(f'TEST6 PASS: empty Non0 — ACC2_Non0={acc2_n0} F1_Non0={f1_n0}')


def test_mosei_majority():
    """Test 7: Verify that MOSEI test-set majority prediction matches expectations.

    If model predicts all-negative on MOSEI test:
      - ACC2_Non0 = neg_non0/non0 = 1253/3294 ≈ 38.0389%
      - ACC2_Has0 = neg/total = 1253/4221 ≈ 29.6849%
    """
    # Simulate MOSEI test distribution
    np.random.seed(0)
    N = 4221
    # Rough MOSEI distribution: ~48% pos, ~30% neg, ~22% zero
    labels = np.random.choice([-1.0, 1.0, 0.0], N, p=[0.30, 0.48, 0.22])
    labels = labels + np.random.normal(0, 0.5, N)
    labels = np.clip(labels, -3, 3)

    # All-negative predictions
    preds = -np.ones(N)
    logits = -np.ones(N)

    m = compute_all_metrics(preds, logits, labels)

    nz = labels != 0
    neg_nz = (labels[nz] < 0).mean() * 100
    neg_total = (labels < 0).mean() * 100

    print(f'  MOSEI sim: neg/non0={neg_nz:.2f}%  ACC2_Non0={m["ACC2_Non0"]:.2f}%')
    print(f'  MOSEI sim: neg/total={neg_total:.2f}%  ACC2_Has0={m["ACC2_Has0"]:.2f}%')
    assert m['F1_Non0'] == 0.0, 'All-neg should give F1=0'
    assert abs(m['ACC2_Non0'] - neg_nz) < 0.1
    print(f'TEST7 PASS: MOSEI majority simulation')


def test_acc7_computation():
    """Test 8: ACC7 7-class accuracy."""
    targets = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    preds = np.array([-2.6, -2.4, -1.3, 0.2, 0.8, 2.1, 2.7])
    acc7 = compute_acc7(preds, targets)
    assert acc7 == 100.0, f'ACC7 ideal should be 100%, got {acc7}'
    print(f'TEST8 PASS: ACC7 ideal = {acc7}%')

    # Partial case
    preds2 = np.array([-2.6, 0.0, -1.3, 0.2, 1.5, 1.5, 2.7])
    acc7_2 = compute_acc7(preds2, targets)
    assert 0 < acc7_2 < 100
    print(f'TEST8b PASS: ACC7 partial = {acc7_2}%')


if __name__ == '__main__':
    print('=== Metrics Unit Tests for MOSEI ===\n')
    tests = [
        test_normal_pos_neg,
        test_labels_with_zero,
        test_all_positive_pred,
        test_all_negative_pred,
        test_nan_prediction_fail_fast,
        test_empty_non0,
        test_mosei_majority,
        test_acc7_computation,
    ]
    passed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f'FAIL: {test.__name__} — {e}')
            import traceback
            traceback.print_exc()
    print(f'\n=== {passed}/{len(tests)} tests passed ===')
