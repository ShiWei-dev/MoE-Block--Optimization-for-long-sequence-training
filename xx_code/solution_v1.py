"""
MoE Block 显存优化实现
=====================

核心优化策略：
1. Expert 计算的 Gradient Checkpointing —— 不保存128个expert的中间激活,反向时重算
2. Shared Expert 的 Gradient Checkpointing —— 不保存 intermediate_size=6144 的中间激活
3. 高效 Token 分发 —— 使用 argsort 替代 one_hot,避免创建 [T, k, num_experts] 的稠密张量
4. 自定义 RMSNorm Autograd Function —— 减少 float32 中间变量的保存
5. 原地操作 —— 在安全的位置使用原地操作减少临时张量

保持与 MoEBlockBaseline 数学等价,权重结构完全一致。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 自定义 RMSNorm Autograd Function
# ============================================================================
class RMSNormFunction(torch.autograd.Function):
    """
    自定义 RMSNorm 的前向/反向传播。
    前向时仅保存 input, weight, rstd（rsqrt 值），避免保存 float32 的 hidden_states 副本。
    """

    @staticmethod
    def forward(ctx, hidden_states, weight, eps):
        # hidden_states: [N, H], weight: [H]
        input_dtype = hidden_states.dtype
        x_fp32 = hidden_states.to(torch.float32)
        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        rstd = torch.rsqrt(variance + eps)
        x_normed = x_fp32 * rstd  # [N, H] in float32
        output = (weight * x_normed.to(input_dtype))

        # 保存用于反向的张量: 原始输入、权重、rstd
        ctx.save_for_backward(hidden_states, weight, rstd.to(input_dtype))
        ctx.input_dtype = input_dtype

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output: [N, H]
        hidden_states, weight, rstd = ctx.saved_tensors
        input_dtype = ctx.input_dtype
        H = hidden_states.shape[-1]

        # 在 float32 下计算梯度
        grad_output_fp32 = grad_output.to(torch.float32)
        x_fp32 = hidden_states.to(torch.float32)
        rstd_fp32 = rstd.to(torch.float32)
        weight_fp32 = weight.to(torch.float32)

        # x_normed = x * rstd
        x_normed = x_fp32 * rstd_fp32

        # grad_weight
        grad_weight = (grad_output_fp32 * x_normed).sum(dim=0)

        # grad_input: d(loss)/d(x)
        # y = weight * x * rstd
        # dy/dx = weight * rstd - weight * x * rstd^3 * (2x / (2*H))
        #       = weight * rstd * (1 - x_normed^2 / H)  ... simplified
        # 使用标准 RMSNorm 反向公式
        grad_normed = grad_output_fp32 * weight_fp32  # [N, H]
        # d(x*rstd)/dx = rstd - x * rstd^3 * x / H = rstd * (1 - x_normed^2 / H)
        # 但更准确的实现:
        # grad_input = rstd * (grad_normed - x_normed * (grad_normed * x_normed).mean(-1, keepdim=True))
        dot = (grad_normed * x_normed).mean(dim=-1, keepdim=True)
        grad_input = rstd_fp32 * (grad_normed - x_normed * dot)
        grad_input = grad_input.to(input_dtype)

        return grad_input, grad_weight, None


def rmsnorm_forward(hidden_states, weight, eps=1e-6):
    """使用自定义 autograd function 的 RMSNorm 前向"""
    return RMSNormFunction.apply(hidden_states, weight, eps)


# ============================================================================
# Expert 单个前向计算 (用于 checkpointing)
# ============================================================================
def expert_forward_fn(current_state, gate_up_weight, down_weight):
    """
    单个 expert 的前向计算。
    current_state: [tokens, hidden_size]
    gate_up_weight: [2*intermediate, hidden_size]
    down_weight: [hidden_size, intermediate]
    """
    gate_up = F.linear(current_state, gate_up_weight)
    gate, up = gate_up.chunk(2, dim=-1)
    hidden = F.silu(gate) * up
    output = F.linear(hidden, down_weight)
    return output


# ============================================================================
# Shared Expert 前向计算 (用于 checkpointing)
# ============================================================================
def shared_expert_forward_fn(x, gate_weight, up_weight, down_weight):
    """
    Shared expert 的前向计算。
    """
    gate_out = F.linear(x, gate_weight)
    up_out = F.linear(x, up_weight)
    hidden = F.silu(gate_out) * up_out
    output = F.linear(hidden, down_weight)
    return output


# ============================================================================
# MoEBlockOptimized
# ============================================================================
class MoEBlockOptimized(nn.Module):
    """
    显存优化版本的 MoE Block。

    与 MoEBlockBaseline 数学定义完全等价。

    权重结构与 baseline 一致：
    - self.experts.gate_up_proj: [num_experts, 2*moe_intermediate_size, hidden_size]
    - self.experts.down_proj: [num_experts, hidden_size, moe_intermediate_size]
    - self.shared_expert.gate_proj.weight: [intermediate_size, hidden_size]
    - self.shared_expert.up_proj.weight: [intermediate_size, hidden_size]
    - self.shared_expert.down_proj.weight: [hidden_size, intermediate_size]
    - self.gate.weight: [num_experts, hidden_size]
    - self.post_norm.weight: [hidden_size]
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.moe_intermediate_size = config.moe_intermediate_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.variance_epsilon = 1e-6

        # Router (gate)
        self.gate = nn.Module()
        self.gate.weight = nn.Parameter(
            torch.zeros(self.num_experts, self.hidden_size)
        )

        # Routed Experts
        self.experts = nn.Module()
        self.experts.gate_up_proj = nn.Parameter(
            torch.empty(
                self.num_experts,
                2 * self.moe_intermediate_size,
                self.hidden_size,
            )
        )
        self.experts.down_proj = nn.Parameter(
            torch.empty(
                self.num_experts,
                self.hidden_size,
                self.moe_intermediate_size,
            )
        )

        # Shared Expert
        self.shared_expert = nn.Module()
        self.shared_expert.gate_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=False
        )
        self.shared_expert.up_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=False
        )
        self.shared_expert.down_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=False
        )

        # Post Norm
        self.post_norm = nn.Module()
        self.post_norm.weight = nn.Parameter(torch.ones(self.hidden_size))

    def _router_forward(self, hidden_states):
        """
        Router 前向计算。
        hidden_states: [N, H]
        返回: routing_weights [N, k], selected_experts [N, k]
        """
        router_logits = F.linear(hidden_states, self.gate.weight)
        router_probs = F.softmax(router_logits, dtype=torch.float, dim=-1)
        top_k_weights, top_k_indices = torch.topk(
            router_probs, self.top_k, dim=-1
        )
        if self.norm_topk_prob:
            top_k_weights = top_k_weights / top_k_weights.sum(
                dim=-1, keepdim=True
            )
        top_k_weights = top_k_weights.to(hidden_states.dtype)
        return top_k_weights, top_k_indices

    def _experts_forward(self, hidden_states, top_k_indices, top_k_weights):
        """
        使用 gradient checkpointing 的 expert 计算。
        使用 argsort 进行高效 token 分发，替代 one_hot。

        hidden_states: [N, H]
        top_k_indices: [N, k]
        top_k_weights: [N, k]
        """
        N = hidden_states.shape[0]
        final_hidden_states = torch.zeros_like(hidden_states)

        # ---- 高效 token 分发: argsort 替代 one_hot ----
        # top_k_indices: [N, k] -> flat: [N*k]
        flat_expert_ids = top_k_indices.reshape(-1)
        flat_token_ids = torch.arange(N, device=hidden_states.device).unsqueeze(1).expand(-1, self.top_k).reshape(-1)
        flat_pos_ids = torch.arange(self.top_k, device=hidden_states.device).unsqueeze(0).expand(N, -1).reshape(-1)

        # 按 expert_id 排序
        sorted_order = flat_expert_ids.argsort(stable=True)
        sorted_token_ids = flat_token_ids[sorted_order]
        sorted_pos_ids = flat_pos_ids[sorted_order]

        # 统计每个 expert 分到的 token 数
        expert_counts = torch.bincount(
            flat_expert_ids[sorted_order].long(),
            minlength=self.num_experts,
        )
        # expert_offsets: [0, count_0, count_0+count_1, ...]
        expert_offsets = torch.zeros(
            self.num_experts + 1,
            dtype=torch.long,
            device=hidden_states.device,
        )
        torch.cumsum(expert_counts[:self.num_experts], dim=0, out=expert_offsets[1:])

        # 遍历活跃的 expert
        for expert_idx in range(self.num_experts):
            start = expert_offsets[expert_idx].item()
            end = expert_offsets[expert_idx + 1].item()
            if start == end:
                continue

            token_idx = sorted_token_ids[start:end]
            pos_idx = sorted_pos_ids[start:end]
            current_state = hidden_states[token_idx]

            # Gradient Checkpointing: 前向时不保存中间激活
            current_hidden_states = torch.utils.checkpoint.checkpoint(
                expert_forward_fn,
                current_state,
                self.experts.gate_up_proj[expert_idx],
                self.experts.down_proj[expert_idx],
                use_reentrant=False,
            )

            # 加权
            weights = top_k_weights[token_idx, pos_idx].unsqueeze(-1)
            current_hidden_states = current_hidden_states * weights

            # 累加到输出
            final_hidden_states.index_add_(
                0, token_idx, current_hidden_states.to(final_hidden_states.dtype)
            )

        return final_hidden_states

    def _shared_expert_fn(self, x):
        """Shared expert 前向，用于 checkpoint 包装"""
        return shared_expert_forward_fn(
            x,
            self.shared_expert.gate_proj.weight,
            self.shared_expert.up_proj.weight,
            self.shared_expert.down_proj.weight,
        )

    def forward(self, hidden_states):
        """
        MoE Block 前向计算。
        hidden_states: [B, T, H]
        返回: [B, T, H]
        """
        bsz, seq_len, hidden_size = hidden_states.shape
        hidden_flat = hidden_states.reshape(-1, hidden_size)

        # 1. Router
        routing_weights, selected_experts = self._router_forward(hidden_flat)

        # 2. Routed Experts (with gradient checkpointing)
        routed_output = self._experts_forward(
            hidden_flat, selected_experts, routing_weights
        )

        # 3. Shared Expert (with gradient checkpointing)
        shared_output = torch.utils.checkpoint.checkpoint(
            self._shared_expert_fn,
            hidden_flat,
            use_reentrant=False,
        )

        # 4. Combine
        combined = routed_output + shared_output

        # 5. RMSNorm (自定义 autograd function)
        output = rmsnorm_forward(combined, self.post_norm.weight, self.variance_epsilon)

        return output.reshape(bsz, seq_len, hidden_size)
