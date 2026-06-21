"""
utils/metrics.py — 全项目唯一指标实现

统一输出：
  - ACC2_Non0 / F1_Non0    (主列，剔除 label == 0)
  - ACC2_Has0 / F1_Has0    (含 label == 0)
  - MAE / Corr             (回归)
  - ACC7                   (七分类，[-3,+3] 四舍五入)

输入格式：numpy array 或 torch.Tensor，自动转换。
"""
import numpy as np
import torch


def _to_numpy(x):
    """统一转换为 numpy 1D array。"""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x).flatten().astype(np.float64)
    return x


def compute_mae(preds, targets):
    """Mean Absolute Error。

    Raises ValueError if ALL predictions are NaN (fail-fast for training collapse).
    """
    preds = _to_numpy(preds)
    targets = _to_numpy(targets)
    valid = ~(np.isnan(preds) | np.isnan(targets))
    if valid.sum() == 0:
        raise ValueError(
            f'MAE: ALL {len(preds)} predictions are NaN! '
            f'(targets NaN={np.isnan(targets).sum()}, preds NaN={np.isnan(preds).sum()}). '
            f'This indicates model collapse — check data for -inf/NaN inputs.'
        )
    return float(np.mean(np.abs(preds[valid] - targets[valid])))


def compute_corr(preds, targets):
    """Pearson Correlation Coefficient。"""
    preds = _to_numpy(preds)
    targets = _to_numpy(targets)
    valid = ~(np.isnan(preds) | np.isnan(targets))
    preds = preds[valid]
    targets = targets[valid]
    if len(preds) < 2 or np.std(preds) == 0 or np.std(targets) == 0:
        return 0.0
    corr = float(np.corrcoef(preds, targets)[0, 1])
    return 0.0 if np.isnan(corr) else corr


def compute_acc2_non0(logits_or_probs, targets, threshold=0.0):
    """
    ACC2_Non0：剔除 label==0 后的二分类准确率。

    Args:
        logits_or_probs: 分类 logit (以 0 为阈值) 或概率 (以 0.5 为阈值)。
        targets: 连续回归标签 [-3, 3]。
        threshold: 对于 logit，≥threshold → positive；对于 prob，可用 0.5。

    Returns:
        accuracy (%): 0-100
    """
    logits_or_probs = _to_numpy(logits_or_probs)
    targets = _to_numpy(targets)

    # 如果输入是 0-1 概率值，阈值用 0.5；否则用传入的 threshold
    # 自动检测：如果全部值在 [0,1] 且 threshold 未显式修改，用 0.5
    if threshold == 0.0 and np.all((logits_or_probs >= 0) & (logits_or_probs <= 1)):
        threshold = 0.5

    pred_bin = (logits_or_probs >= threshold).astype(int)
    tgt_bin = (targets >= 0).astype(int)

    nz = targets != 0
    if nz.sum() == 0:
        return 0.0

    return float((pred_bin[nz] == tgt_bin[nz]).mean()) * 100.0


def compute_f1_non0(logits_or_probs, targets, threshold=0.0):
    """
    F1_Non0：剔除 label==0 后的 F1 score。

    Returns:
        F1 (%): 0-100
    """
    logits_or_probs = _to_numpy(logits_or_probs)
    targets = _to_numpy(targets)

    if threshold == 0.0 and np.all((logits_or_probs >= 0) & (logits_or_probs <= 1)):
        threshold = 0.5

    pred_bin = (logits_or_probs >= threshold).astype(int)
    tgt_bin = (targets >= 0).astype(int)

    nz = targets != 0
    if nz.sum() == 0:
        return 0.0

    pb = pred_bin[nz]
    tb = tgt_bin[nz]

    tp = int(((pb == 1) & (tb == 1)).sum())
    fp = int(((pb == 1) & (tb == 0)).sum())
    fn = int(((pb == 0) & (tb == 1)).sum())

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return f1 * 100.0


def compute_acc2_has0(logits_or_probs, targets, threshold=0.0):
    """
    ACC2_Has0：含 label==0 的二分类准确率。
    """
    logits_or_probs = _to_numpy(logits_or_probs)
    targets = _to_numpy(targets)

    if threshold == 0.0 and np.all((logits_or_probs >= 0) & (logits_or_probs <= 1)):
        threshold = 0.5

    pred_bin = (logits_or_probs >= threshold).astype(int)
    tgt_bin = (targets >= 0).astype(int)

    return float((pred_bin == tgt_bin).mean()) * 100.0


def compute_f1_has0(logits_or_probs, targets, threshold=0.0):
    """
    F1_Has0：含 label==0 的 F1 score。
    """
    logits_or_probs = _to_numpy(logits_or_probs)
    targets = _to_numpy(targets)

    if threshold == 0.0 and np.all((logits_or_probs >= 0) & (logits_or_probs <= 1)):
        threshold = 0.5

    pred_bin = (logits_or_probs >= threshold).astype(int)
    tgt_bin = (targets >= 0).astype(int)

    tp = int(((pred_bin == 1) & (tgt_bin == 1)).sum())
    fp = int(((pred_bin == 1) & (tgt_bin == 0)).sum())
    fn = int(((pred_bin == 0) & (tgt_bin == 1)).sum())

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return f1 * 100.0


def compute_acc7(preds, targets):
    """
    ACC7：将回归值 [-3, 3] 四舍五入并 clip 到 [-3, 3] 得到 7 类。
    计算预测与真实 7 类的准确率。
    """
    preds = _to_numpy(preds)
    targets = _to_numpy(targets)

    # clip 到 [-3, 3]，四舍五入到整数类
    pred_cls = np.clip(np.round(preds), -3, 3).astype(int)
    tgt_cls = np.clip(np.round(targets), -3, 3).astype(int)

    valid = ~(np.isnan(preds) | np.isnan(targets))
    if valid.sum() == 0:
        return 0.0

    return float((pred_cls[valid] == tgt_cls[valid]).mean()) * 100.0


def compute_all_metrics(reg_pred, cls_logit, targets):
    """
    一次性计算全部 7 个指标。

    Args:
        reg_pred:  [N] 回归预测值
        cls_logit: [N] 二分类 logit（或概率）
        targets:   [N] 真实回归标签

    Returns:
        dict with keys:
            MAE, Corr, ACC2_Non0, F1_Non0, ACC2_Has0, F1_Has0, ACC7

    Raises:
        ValueError: if reg_pred has NaN count > 10% (fail-fast on collapse)
    """
    reg_pred = _to_numpy(reg_pred)
    cls_logit = _to_numpy(cls_logit)
    targets = _to_numpy(targets)

    # Fail-fast: if >10% of predictions are NaN, raise immediately
    nan_ratio = np.isnan(reg_pred).mean()
    if nan_ratio > 0.1:
        raise ValueError(
            f'Collapse detected: {nan_ratio*100:.1f}% of reg_pred ({np.isnan(reg_pred).sum()}/{len(reg_pred)}) '
            f'are NaN. Check data pipeline for -inf/NaN inputs or model divergence.'
        )

    return {
        'MAE': compute_mae(reg_pred, targets),
        'Corr': compute_corr(reg_pred, targets),
        'ACC2_Non0': compute_acc2_non0(cls_logit, targets),
        'F1_Non0': compute_f1_non0(cls_logit, targets),
        'ACC2_Has0': compute_acc2_has0(cls_logit, targets),
        'F1_Has0': compute_f1_has0(cls_logit, targets),
        'ACC7': compute_acc7(reg_pred, targets),
    }


# ============================================================
# 单元验证 (直接运行本文件)
# ============================================================
if __name__ == '__main__':
    print("=== 指标单元验证 ===\n")

    # 构造测试数据
    np.random.seed(42)
    N = 1000
    targets = np.random.uniform(-3, 3, N)
    # 添加一些 zero label
    targets[:50] = 0.0
    preds = targets + np.random.normal(0, 0.5, N)
    # 二分类 logit：与 target 符号一致 + 噪声
    logits = np.where(targets >= 0, 1.0, -1.0) + np.random.normal(0, 0.3, N)

    metrics = compute_all_metrics(preds, logits, targets)
    for k, v in metrics.items():
        print(f"  {k:12s}: {v:.4f}")

    # 边界测试：全正标签
    t_pos = np.ones(100) * 2.0
    p_pos = t_pos + np.random.normal(0, 0.3, 100)
    l_pos = np.ones(100)
    m_pos = compute_all_metrics(p_pos, l_pos, t_pos)
    print(f"\n  全正标签测试: ACC2_Non0={m_pos['ACC2_Non0']:.2f}%")

    # 边界测试：corr 分母为 0
    t_const = np.ones(10)
    p_const = np.ones(10) * 0.5
    m_const = compute_all_metrics(p_const, p_const, t_const)
    print(f"  常量输入测试: Corr={m_const['Corr']:.4f} (期望 0.0)")

    # 边界测试：全 zero label
    t_zeros = np.zeros(50)
    p_zeros = np.random.normal(0, 0.5, 50)
    m_zeros = compute_all_metrics(p_zeros, p_zeros, t_zeros)
    print(f"  全 zero 标签测试: ACC2_Non0={m_zeros['ACC2_Non0']:.2f} (期望 0.0)")

    # ACC7 测试
    t7 = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    p7 = np.array([-2.6, -2.4, -1.3, 0.2, 0.8, 2.1, 2.7])
    acc7 = compute_acc7(p7, t7)
    print(f"\n  ACC7 理想测试: {acc7:.2f}% (期望 100.0%)")

    print("\n=== 单元验证完成 ===")
