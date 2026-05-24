 # 长序列训练 MoE Block 显存优化

本项目是是在不改变 MoE Block 数学定义的前提下，优化长序列训练过程中的峰值显存占用，并尽量降低单步训练耗时。

项目基于赛题提供的 `MoEBlockBaseline`，实现了可通过 `load_state_dict(strict=True)` 对齐权重的 `MoEBlockOptimized`。最终优化版位于 `xx_code/solution.py`，提交入口类名为 `MoEBlockOptimized`。

## 项目结构

```text
.
├── 赛题说明.md              # 赛题背景、环境限制、提交与评分规则
├── xx_report.md             # 方案报告：优化思路、正确性、实验结果
├── images/                  # 报告或说明中使用的图片
├── 附件/
│   ├── baseline.py          # 官方 baseline: MoEBlockBaseline
│   ├── solution.py          # 官方模板/参考提交格式
│   ├── correctness_check.py # 正确性自测脚本
│   ├── benchmark.py         # 显存与速度 benchmark 脚本
│   └── readme.md            # 官方附件运行说明
├── xx_code/
│   ├── solution.py          # 最终优化实现 v6
│   ├── solution_v1.py       # 历史迭代版本
│   ├── ...
│   └── compare_baseline.py  # baseline 与 optimized 对比脚本
└── 提交/                    # 本地提交产物，已被 .gitignore 忽略
```

## 核心方案

最终版本 `xx_code/solution.py` 采用自定义 `torch.autograd.Function` 和分块计算降低训练激活显存：

- `MoERouterExpertsFunction`：将 Router 与 Routed Experts 合并到单个 autograd function，前向仅保存必要输入和权重，反向逐 expert 重算中间激活并立即释放。
- `ChunkedSharedExpertFunction`：按 chunk 计算 Shared Expert，避免在长序列下同时保存完整的 `N x 6144` 中间激活。
- `FusedAddRMSNormFunction`：融合 routed/shared 输出相加与 RMSNorm，并按 chunk 处理 RMSNorm 的 float32 中间变量。
- 高效 token 分发：使用 `argsort` 与 `bincount` 替代 baseline 中的稠密 `one_hot` 分发方式。
- 缓冲区复用：在 expert 前向与反向循环中预分配并复用临时张量，减少 CUDA alloc/free 和碎片化开销。

## 环境要求

赛题评测环境：

- Python 3.12
- PyTorch 2.8.0
- CUDA 12.8
- 单卡 H20-96GB

本地正确性脚本可在 CPU 上运行小规模 smoke test；性能 benchmark 需要 CUDA 环境。

## 快速开始

### 1. 正确性检查

在项目根目录运行：

```powershell
python 附件\correctness_check.py --solution xx_code\solution.py
```

小规模快速检查：

```powershell
python 附件\correctness_check.py --solution xx_code\solution.py --seq-len 128 --hidden-size 256 --intermediate-size 768 --moe-intermediate-size 96 --num-experts 16 --top-k 4
```

正确性脚本会检查：

- 前向输出一致性：`rtol=2e-2, atol=1e-3`
- 输入梯度一致性：`cosine_sim >= 0.995` 且 `relative_l2 <= 1e-2`
- 参数梯度一致性：覆盖 router、routed experts、shared expert 和 post norm

### 2. 显存与速度 benchmark

```powershell
python 附件\benchmark.py --solution xx_code\solution.py
```

快速 smoke run：

```powershell
python 附件\benchmark.py --solution xx_code\solution.py --seq-lens 2048 --warmup 1 --measure 1
```

默认序列长度：

```text
2048, 8192, 32768, 65536, 131072, 262144
```

### 3. 与 baseline 对比

```powershell
python xx_code\compare_baseline.py --seq-lens 2048,8192,32768 --warmup 1 --measure 3
```

该脚本会同时输出正确性检查、baseline/optimized 的峰值显存与耗时对比，以及按赛题权重模拟的评分。

## 实验结果摘要

报告中的本地实验环境为 RTX 5070 Ti Laptop，`bf16`，`batch_size=1`。实测正确性全部通过，主要性能结果如下：

| SeqLen | Baseline Peak Mem | Optimized Peak Mem | Mem 降幅 | Baseline Avg | Optimized Avg | Speed 提升 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | 4788.99 MB | 3746.66 MB | 21.8% | 1285.51 ms | 132.62 ms | 89.7% |
| 8192 | 5767.72 MB | 3919.90 MB | 32.0% | 1381.23 ms | 222.28 ms | 83.9% |
| 32768 | 10605.04 MB | 4610.64 MB | 56.5% | 25687.42 ms | 574.39 ms | 97.8% |

更完整的优化原理、梯度推导与实验分析见 `xx_report.md`。

## 提交说明

赛题提交时需要提供：

- `code.zip`：包含一个 `solution.py`，且其中定义 `MoEBlockOptimized`
- `report.pdf`：方案报告

当前仓库中的最终代码可从 `xx_code/solution.py` 作为提交版 `solution.py` 使用。
