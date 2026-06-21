# env/install_commands.md — 环境安装命令记录

> 生成时间：2026-06-16 P2  
> 环境：mme_xlstm

---

## 一、创建环境

```bash
# 创建 conda 环境
conda create -p E:/Anaconda3/envs/mme_xlstm python=3.10 -y --override-channels -c defaults

# 激活环境
conda activate E:/Anaconda3/envs/mme_xlstm
```

## 二、安装 PyTorch (nightly, CUDA 12.8)

RTX 5070 Ti (Blackwell sm_120) 需要 CUDA 12.8。多次尝试：

1. PyTorch 2.3.0+cu118 → GPU 不可用 (无 sm_120 kernel)
2. PyTorch 2.5.1+cu124 → GPU 不可用 (无 sm_120 kernel)
3. PyTorch 2.6.0+cu124 → GPU 不可用 (无 sm_120 kernel)
4. **PyTorch 2.12.0.dev20260408+cu128 (nightly) → GPU 可用 ✅**

```bash
pip install torch --index-url https://download.pytorch.org/whl/nightly/cu128 --upgrade
pip install torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

⚠ **临时 nightly 方案**：因 RTX 5070 Ti (Blackwell sm_120) 在 stable PyTorch 预编译二进制中缺少 kernel。P4 正式训练前如 stable 版本已支持，应切换回 stable release。

## 三、安装核心依赖

```bash
pip install transformers==4.34.1 tqdm einops scikit-learn matplotlib openpyxl pyyaml --index-url https://pypi.org/simple/
```

注：tuna tsinghua 镜像 SSL 证书问题，需使用默认 PyPI 或 pip config 切换。

## 四、numpy 兼容性

```bash
# torch 2.x 与 numpy 2.x 不兼容 (A module compiled using NumPy 1.x...)
pip install "numpy<2" --index-url https://pypi.org/simple/
```

## 五、验证

```bash
python --version                           # Python 3.10.20
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -c "import transformers; print(transformers.__version__)"
```

## 六、导出环境

```bash
pip freeze > env/pip_freeze_mme_xlstm.txt
conda env export -p E:/Anaconda3/envs/mme_xlstm > env/environment_mme_xlstm.yml
```

## 七、注意事项

- RTX 5070 Ti 是 Blackwell 架构 (sm_120)，需要 CUDA Toolkit ≥ 12.4
- PyTorch 版本必须匹配 CUDA 版本：torch 2.5.1+cu124
- 旧 mme 环境 (Python 3.9.21, PyTorch 2.3.0+cu118) 保留不变，不新增包
- MLCL baseline (P6) 可能需要 Python 3.8.10 + PyTorch 1.12.1+cu113，建在 mme_mlcl 环境
