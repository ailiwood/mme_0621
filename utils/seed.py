"""
utils/seed.py — 全项目统一随机种子控制。

控制 random / numpy / torch / cuda 全部随机源，
保证实验可复现。
"""
import random
import numpy as np
import torch
import os


def set_seed(seed: int, deterministic: bool = False):
    """
    设置全局随机种子。

    Args:
        seed: 随机种子整数
        deterministic: 是否启用 cudnn deterministic 模式
                       (开启后可能略微降低性能，但提高可复现性)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    else:
        # 默认：允许 cudnn 自动选择最优算法（benchmark）
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # Python hash 稳定性 (可选)
    os.environ['PYTHONHASHSEED'] = str(seed)
