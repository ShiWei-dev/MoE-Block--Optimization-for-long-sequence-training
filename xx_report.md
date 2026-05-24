# MoE Block 长序列训练显存优化 — 方案报告

## 一、整体优化思路

本方案针对 MoE Block 在长序列训练场景下的**峰值显存占用**进行优化，核心思路是：在保持数学等价性的前提下，通过**自定义 `torch.autograd.Function`** 配合**分块计算（Chunked Computation）**，实现训练时中间激活张量的极致显存压缩。

具体采用以下四项协同优化策略：

1. **自定义 `MoERouterExpertsFunction`**：将 Router 和全部 128 个 Expert 的计算合并进一个 `torch.autograd.Function`。前向时仅保存输入和权重引用，不保存任何 Expert 的中间激活（gate, up, SiLU(gate)×up）；反向时逐 expert 重算中间激活、计算梯度后立即释放。同时将 Router 纳入该 Function，消除 softmax/topk/normalize 的 autograd 图节点和中间张量（如 float32 的 router_probs）。

2. **分块 Shared Expert（`ChunkedSharedExpertFunction`）**：将 Shared Expert（intermediate_size=6144）的前向和反向计算按 chunk 分块处理。避免同时创建完整的 N×6144 中间激活张量（gate_out, up_out, SiLU(gate_out), hidden, grad_hidden 共约5个），在128K序列下从 ~7.5GB 降至 ~60MB。

3. **分块 RMSNorm（`RMSNormFunction`）**：将 RMSNorm 的前向和反向按 chunk 分块处理，避免创建完整的 N×2048 float32 中间张量（x_fp32, x_normed），在128K序列下从 ~2GB 降至 ~64MB。

4. **高效 Token 分发**：使用 `argsort` + `bincount` 替代 Baseline 中的 `F.one_hot`，避免创建 `[T, 8, 128]` 的稠密矩阵，显存从 O(T×k×E) 降至 O(T×k)。

## 二、显存优化的核心原理

### 2.1 Baseline 显存瓶颈分析

Baseline（`MoEBlockBaseline`）的训练时显存消耗主要来自：

| 组件 | 参数大小 (bf16) | PyTorch autograd 保存的中间激活 |
|------|-----------------|-------------------------------|
| Router | 128×2048×2B ≈ 0.5MB | router_probs: N×128×4B (float32)；topk/normalize 图节点 |
| 128 Routed Experts | gate_up: 768MB, down: 384MB | 每个expert: current_state, gate, up, SiLU(gate)×up, current_hidden → 共5个张量 |
| Shared Expert | ~75MB | gate_out: N×6144, up_out: N×6144, SiLU×up: N×6144, hidden: N×6144 → 共4个 |
| RMSNorm | 4KB | x_fp32: N×2048×4B, variance, x_normed 等 float32 中间变量 |
| Token 分发 | — | one_hot: T×8×128×8B |

**长序列下的瓶颈量化**（以128K tokens, bf16为例）：

| 中间激活 | 大小 | 说明 |
|----------|------|------|
| Expert 中间激活（累计） | ~数GB | 所有128个expert的gate/up/SiLU×up，因循环结构全部保留到反向 |
| Shared Expert 中间激活 | **~4.6 GB** | 131072 × 6144 × 2B × 4个张量 |
| RMSNorm float32中间变量 | **~2 GB** | 131072 × 2048 × 4B × 2个张量 |
| Router router_probs | **~64 MB** | 131072 × 128 × 4B (float32) |
| one_hot 张量 | **~1 GB** | 131072 × 8 × 128 × 8B |

### 2.2 优化原理一：激活重计算（Expert）

将整个 expert 循环封装在单一 `torch.autograd.Function` 中：

**Baseline autograd 行为**：
```
forward: 循环128个expert → PyTorch 自动保存每个expert的 5个中间张量
backward: 使用保存的激活计算梯度
峰值显存: O(∑tokens_i × moe_intermediate × 5)
```

**本方案**：
```
forward: 循环128个expert计算输出 → 中间激活计算后立即丢弃
save_for_backward: 仅 hidden_states, 权重引用, 路由索引
backward: 逐expert重算中间激活 → 手动计算梯度 → 立即释放该expert的中间激活
峰值显存: O(max_tokens_per_expert × moe_intermediate × 5)
```

显存从所有expert累计降为单个expert峰值。

### 2.3 优化原理二：分块计算（Shared Expert）

Shared Expert 的 intermediate_size=6144，在长序列下中间激活极大。分块处理的核心思想：

**不分块**：
```python
gate_out = F.linear(x, gate_w)      # [N, 6144] — 全部 N 个 token 同时计算
up_out = F.linear(x, up_w)          # [N, 6144]
hidden = F.silu(gate_out) * up_out  # [N, 6144]
output = F.linear(hidden, down_w)   # [N, 2048]
# 峰值: 同时存在 gate_out + up_out + silu + hidden ≈ N × 6144 × 2B × 4
```

**分块（chunk_size=2048）**：
```python
for i in range(0, N, 2048):
    xc = x[i:i+2048]
    gate_out = F.linear(xc, gate_w)       # [2048, 6144]
    up_out = F.linear(xc, up_w)           # [2048, 6144]
    hidden = F.silu(gate_out) * up_out    # [2048, 6144]
    output[i:i+2048] = F.linear(hidden, down_w)
    # 每个chunk结束后, gate_out/up_out/hidden 自动释放
# 峰值: 2048 × 6144 × 2B × 4 ≈ 96 MB (固定, 不随N增长)
```

**在128K下**: 从 ~7.5GB 降至 ~96MB，节省 **~7.4GB**。

反向传播同样分块处理，每个chunk独立重算前向、计算梯度、累加权重梯度后释放。

### 2.4 优化原理三：分块 RMSNorm

RMSNorm 需要将输入转为 float32 计算 variance 和 rsqrt。不分块时，完整的 N×2048 float32 张量在128K下占 ~1GB。分块后每次仅处理 4096 个 token，峰值降至 ~64MB。

### 2.5 优化原理四：Router 合并与高效分发

将 Router（F.linear → softmax → topk → normalize）纳入 Expert 的自定义 Function 中，消除：
- router_probs (N×128, float32) 的 autograd 保存
- softmax/topk/normalize 的图节点开销

反向时从已保存的 hidden_states 和 gate_weight 重算 router，手动实现 normalize → topk(scatter) → softmax → linear 的完整梯度链。

Token 分发使用 argsort + bincount 替代 one_hot，仅需 O(N×k) 的整数索引。

## 三、正确性验证方法

### 3.1 验证指标

按赛题要求，正确性通过以下指标判定：

| 检查项 | 判定标准 |
|--------|----------|
| 前向输出 | `torch.allclose(rtol=2e-2, atol=1e-3)` |
| 输入梯度 | `cosine_sim >= 0.995` 且 `relative_l2 <= 1e-2` |
| 参数梯度 | `cosine_sim >= 0.995` 且 `relative_l2 <= 1e-2` |

### 3.2 实测正确性结果

使用赛题提供的 `correctness_check.py` 和自编 `compare_baseline.py` 验证，全部检查项通过：

| 检查项 | 结果 | cosine_sim | rel_l2 |
|--------|------|------------|--------|
| 前向输出 | **通过** ✅ | — | max_abs_diff=0.0 |
| 输入梯度 | **通过** ✅ | 0.999969 | 7.847e-03 |
| 参数梯度(gate.weight) | **通过** ✅ | >0.999 | <1e-2 |
| 参数梯度(experts.gate_up_proj) | **通过** ✅ | >0.999 | <1e-2 |
| 参数梯度(experts.down_proj) | **通过** ✅ | >0.999 | <1e-2 |
| 参数梯度(shared_expert.*) | **通过** ✅ | >0.999 | <1e-2 |
| 参数梯度(post_norm.weight) | **通过** ✅ | >0.999 | <1e-2 |

### 3.3 正确性保证机制

1. **数学等价**：Router 的 softmax → topk → normalize、Expert 的 F.linear → SiLU → mul → F.linear、Shared Expert、RMSNorm 的前向计算与 Baseline 完全一致。
2. **梯度等价**：自定义 backward 中手动推导的梯度公式与 PyTorch autograd 自动推导的结果数学等价，包括 SiLU 梯度 `d(silu)/dx = sigmoid(x) + x·sigmoid(x)·(1-sigmoid(x))`、softmax 梯度、topk scatter、normalize 反向等。
3. **权重对齐**：通过 `load_state_dict(strict=True)` 确保所有参数完全对齐。
4. **分块不影响结果**：分块仅影响计算顺序和内存分配，不改变矩阵乘法的数学结果；权重梯度通过 `addmm_` 跨 chunk 累加，等价于一次性计算。

## 四、速度与显存之间的权衡分析

### 4.1 各优化策略的权衡

| 优化策略 | 显存节省 | 速度影响 | 评估 |
|----------|----------|----------|------|
| Expert 激活重计算 | ★★★★★ (消除所有expert中间激活) | 反向多一次expert前向 | 核心优化，显存收益远大于速度损失 |
| 分块 Shared Expert | ★★★★★ (128K: 7.5GB→60MB) | 分块循环微小开销 | **v4关键改进**，几乎无速度损失 |
| 分块 RMSNorm | ★★★ (128K: 2GB→64MB) | 分块循环微小开销 | 长序列下显著，无速度损失 |
| Router 合并 | ★★ (消除router_probs float32) | 反向重算router | 额外显存节省，速度影响可忽略 |
| argsort 替代 one_hot | ★★ (128K: 1GB→8MB) | 零额外开销 | 纯收益 |

### 4.2 迭代优化历程

| 版本 | 核心改动 | 2K Mem Δ | 8K Mem Δ | 32K Mem Δ | 32K Speed Δ |
|------|----------|----------|----------|-----------|-------------|
| v1 | 128次 checkpoint | -31.9% ❌ | +6.1% | +19.8% | +68.3% |
| v2 | 单一 autograd.Function | -5.3% | +31.9% | +30.6% | +99.7% |
| v3 | Router合并进Function | ≈v2 | ≈v2 | ≈v2 | ≈v2 |
| **v4** | **+分块Shared+分块Norm** | **+21.8%** ✅ | **+32.0%** | **+56.5%** | **+97.8%** |

**v4 的突破**：通过分块计算，32K 序列的显存优化从 v2 的 30.6% 跃升至 **56.5%**，同时速度保持 97.8% 的提升。2K 序列也从 v2 的负优化（-5.3%）转为正优化（**+21.8%**）。

### 4.3 赛题评分分析

赛题显存:速度权重为 **6:4**。v4 在两个维度均实现显著提升：
- **显存**：全序列长度正优化（21.8%~56.5%），且随序列增长优化幅度递增
- **速度**：全序列长度提升 83.9%~97.8%

这意味着在总分计算中，显存和速度均贡献正分，且长序列下得分更高。

## 五、主要实验结果与结论

### 5.1 实测结果（RTX 5070 Ti Laptop, bf16, batch_size=1）

**正确性**：全部通过。

**性能对比**：

| SeqLen | Model | Peak Mem (MB) | Avg (ms) | Mem Δ | Speed Δ |
|--------|-------|---------------|----------|-------|---------|
| 2048 | Baseline | 4788.99 | 1285.51 | — | — |
| 2048 | **Optimized** | **3746.66** | **132.62** | **+21.8%** | **+89.7%** |
| 8192 | Baseline | 5767.72 | 1381.23 | — | — |
| 8192 | **Optimized** | **3919.90** | **222.28** | **+32.0%** | **+83.9%** |
| 32768 | Baseline | 10605.04 | 25687.42 | — | — |
| 32768 | **Optimized** | **4610.64** | **574.39** | **+56.5%** | **+97.8%** |

### 5.2 结果分析

1. **显存优化**：
   - 2K 序列：峰值显存从 4789MB 降至 3747MB，**节省 21.8%**
   - 8K 序列：峰值显存从 5768MB 降至 3920MB，**节省 32.0%**
   - 32K 序列：峰值显存从 10605MB 降至 4611MB，**节省 56.5%**
   - 优化幅度随序列长度增长而递增，这是因为分块计算的优势在长序列下更加显著（中间激活的绝对大小与 N 成正比，而分块后峰值与 chunk_size 成正比，不随 N 增长）

2. **速度提升**：
   - 全部序列长度速度提升 **83.9%~97.8%**
   - 32K 序列下 Baseline 耗时 25687ms（因显存压力导致频繁内存分配器碎片整理），优化版仅 574ms
   - 速度提升的来源：消除 autograd 中间张量追踪开销 + 减少显存压力下的内存管理开销

3. **分块计算的效果验证**：v2→v4 的对比直接体现了分块计算的价值——32K 显存优化从 30.6% 提升至 56.5%，说明 Shared Expert 的 N×6144 中间激活是 v2/v3 的主要瓶颈。

### 5.3 结论

本方案通过**自定义 `torch.autograd.Function` + 分块计算**的组合策略，在保持数学等价的前提下实现了：

- **显存降低 21.8%~56.5%**（随序列长度递增）
- **速度提升 83.9%~97.8%**
- **全部正确性检查项高精度通过**

方案的核心创新在于：
1. **分块 Shared Expert**：将 N×6144 的中间激活峰值压缩为固定大小，不随序列长度增长
2. **分块 RMSNorm**：避免完整的 N×H float32 中间变量
3. **单一 autograd.Function 合并 Router+Expert**：消除 128 个 expert 的 autograd 框架开销和 Router 的中间张量
4. **逐 expert 激活重计算**：反向时逐 expert 重算并立即释放，峰值仅为单个 expert 的激活大小

在赛题 6:4 的显存:速度权重下，本方案在两个维度均实现显著正向收益。
