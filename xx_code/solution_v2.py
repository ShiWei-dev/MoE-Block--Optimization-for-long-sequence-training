"""
MoE Block 显存优化实现 v2
========================

核心优化策略：
1. 单一自定义 Autograd Function 包裹全部 Expert 计算 —— 避免128次 checkpoint 调用开销
2. 单一自定义 Autograd Function 包裹 Shared Expert 计算 —— 不保存 T×6144 中间激活
3. 高效 Token 分发 —— 使用 argsort 替代 one_hot
4. 自定义 RMSNorm Autograd Function —— 减少 float32 中间变量保存

保持与 MoEBlockBaseline 数学等价，权重结构完全一致。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 自定义 RMSNorm Autograd Function
# ============================================================================
class RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, weight, eps):
        input_dtype = hidden_states.dtype
        x_fp32 = hidden_states.to(torch.float32)
        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        rstd = torch.rsqrt(variance + eps)
        x_normed = x_fp32 * rstd
        output = (weight * x_normed.to(input_dtype))
        ctx.save_for_backward(hidden_states, weight, rstd.to(input_dtype))
        ctx.input_dtype = input_dtype
        return output

    @staticmethod
    def backward(ctx, grad_output):
        hidden_states, weight, rstd = ctx.saved_tensors
        input_dtype = ctx.input_dtype

        grad_output_fp32 = grad_output.to(torch.float32)
        x_fp32 = hidden_states.to(torch.float32)
        rstd_fp32 = rstd.to(torch.float32)
        weight_fp32 = weight.to(torch.float32)

        x_normed = x_fp32 * rstd_fp32
        grad_weight = (grad_output_fp32 * x_normed).sum(dim=0)

        grad_normed = grad_output_fp32 * weight_fp32
        dot = (grad_normed * x_normed).mean(dim=-1, keepdim=True)
        grad_input = rstd_fp32 * (grad_normed - x_normed * dot)
        grad_input = grad_input.to(input_dtype)

        return grad_input, grad_weight, None


# ============================================================================
# 全部 Expert 的自定义 Autograd Function（单次调用，避免128次 checkpoint 开销）
# ============================================================================
class MoEExpertsFunction(torch.autograd.Function):
    """
    将整个 expert 循环包在一个 autograd.Function 中。
    前向：计算所有 expert 输出，但不保存中间激活（gate, up, silu*up）。
    反向：重新计算每个 expert 的前向以获取中间激活，然后计算梯度。

    这比128次 torch.utils.checkpoint.checkpoint 开销小得多。
    """

    @staticmethod
    def forward(
        ctx,
        hidden_states,      # [N, H]
        gate_up_proj,        # [E, 2*I, H]
        down_proj,           # [E, H, I]
        top_k_indices,       # [N, k]
        top_k_weights,       # [N, k]
        num_experts,
        top_k,
    ):
        N, H = hidden_states.shape

        final_hidden_states = torch.zeros(N, H, dtype=hidden_states.dtype, device=hidden_states.device)

        # ---- 高效 token 分发: argsort ----
        flat_expert_ids = top_k_indices.reshape(-1)
        flat_token_ids = torch.arange(N, device=hidden_states.device).unsqueeze(1).expand(-1, top_k).reshape(-1)
        flat_pos_ids = torch.arange(top_k, device=hidden_states.device).unsqueeze(0).expand(N, -1).reshape(-1)

        sorted_order = flat_expert_ids.argsort(stable=True)
        sorted_token_ids = flat_token_ids[sorted_order]
        sorted_pos_ids = flat_pos_ids[sorted_order]

        expert_counts = torch.bincount(flat_expert_ids.long(), minlength=num_experts)
        expert_offsets = torch.zeros(num_experts + 1, dtype=torch.long, device=hidden_states.device)
        torch.cumsum(expert_counts[:num_experts], dim=0, out=expert_offsets[1:])

        for expert_idx in range(num_experts):
            start = expert_offsets[expert_idx].item()
            end = expert_offsets[expert_idx + 1].item()
            if start == end:
                continue

            token_idx = sorted_token_ids[start:end]
            pos_idx = sorted_pos_ids[start:end]
            current_state = hidden_states[token_idx]

            # 前向计算（不保存中间激活）
            gate_up = F.linear(current_state, gate_up_proj[expert_idx])
            gate, up = gate_up.chunk(2, dim=-1)
            current_hidden = F.silu(gate) * up
            current_output = F.linear(current_hidden, down_proj[expert_idx])

            weights = top_k_weights[token_idx, pos_idx].unsqueeze(-1)
            current_output = current_output * weights

            final_hidden_states.index_add_(0, token_idx, current_output.to(hidden_states.dtype))

        # 只保存计算图需要的：输入、权重、路由信息
        ctx.save_for_backward(hidden_states, gate_up_proj, down_proj, top_k_indices, top_k_weights)
        ctx.sorted_token_ids = sorted_token_ids
        ctx.sorted_pos_ids = sorted_pos_ids
        ctx.expert_offsets = expert_offsets
        ctx.num_experts = num_experts
        ctx.top_k = top_k

        return final_hidden_states

    @staticmethod
    def backward(ctx, grad_output):
        hidden_states, gate_up_proj, down_proj, top_k_indices, top_k_weights = ctx.saved_tensors
        sorted_token_ids = ctx.sorted_token_ids
        sorted_pos_ids = ctx.sorted_pos_ids
        expert_offsets = ctx.expert_offsets
        num_experts = ctx.num_experts

        N, H = hidden_states.shape

        grad_hidden_states = torch.zeros_like(hidden_states)
        grad_gate_up_proj = torch.zeros_like(gate_up_proj)
        grad_down_proj = torch.zeros_like(down_proj)
        grad_top_k_weights = torch.zeros_like(top_k_weights)

        for expert_idx in range(num_experts):
            start = expert_offsets[expert_idx].item()
            end = expert_offsets[expert_idx + 1].item()
            if start == end:
                continue

            token_idx = sorted_token_ids[start:end]
            pos_idx = sorted_pos_ids[start:end]
            current_state = hidden_states[token_idx]
            weights = top_k_weights[token_idx, pos_idx].unsqueeze(-1)

            # ---- 重新前向计算（获取中间激活） ----
            gate_up = F.linear(current_state, gate_up_proj[expert_idx])
            gate, up = gate_up.chunk(2, dim=-1)
            gate_act = F.silu(gate)  # sigmoid(gate) * gate
            current_hidden = gate_act * up
            current_output = F.linear(current_hidden, down_proj[expert_idx])

            # ---- 反向梯度计算 ----
            grad_out_local = grad_output[token_idx]  # [tokens, H]

            # grad through weight multiplication: output = linear_out * weights
            grad_linear_out = grad_out_local * weights
            grad_w = (grad_out_local * current_output).sum(dim=-1)
            grad_top_k_weights[token_idx, pos_idx] += grad_w

            # grad through down_proj: current_output = current_hidden @ down_proj.T
            # grad_current_hidden = grad_linear_out @ down_proj
            # grad_down_proj[expert_idx] += grad_linear_out.T @ current_hidden
            grad_current_hidden = grad_linear_out.mm(down_proj[expert_idx])  # [tokens, I]
            grad_down_proj[expert_idx].addmm_(grad_linear_out.t(), current_hidden)

            # grad through gate_act * up
            grad_gate_act = grad_current_hidden * up
            grad_up = grad_current_hidden * gate_act

            # grad through SiLU: silu(x) = x * sigmoid(x)
            # d(silu)/dx = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
            #            = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
            #            = silu(x) / x + sigmoid(x) * (1 - sigmoid(x)) * x  ... but simpler:
            #            = sigmoid(x) * (1 + gate - gate_act)  if gate_act = silu(gate)
            sigmoid_gate = torch.sigmoid(gate)
            grad_gate = grad_gate_act * (sigmoid_gate + gate * sigmoid_gate * (1.0 - sigmoid_gate))

            # grad through chunk: gate_up = [gate, up]
            grad_gate_up = torch.cat([grad_gate, grad_up], dim=-1)

            # grad through linear: gate_up = current_state @ gate_up_proj.T
            grad_current_state = grad_gate_up.mm(gate_up_proj[expert_idx])
            grad_gate_up_proj[expert_idx].addmm_(grad_gate_up.t(), current_state)

            # accumulate grad for hidden_states
            grad_hidden_states.index_add_(0, token_idx, grad_current_state)

        return grad_hidden_states, grad_gate_up_proj, grad_down_proj, None, grad_top_k_weights, None, None


# ============================================================================
# Shared Expert 的自定义 Autograd Function
# ============================================================================
class SharedExpertFunction(torch.autograd.Function):
    """
    Shared Expert 的自定义前向/反向。
    前向不保存 gate_out, up_out, silu(gate_out)*up_out 等中间激活。
    反向时重新计算。
    """

    @staticmethod
    def forward(ctx, x, gate_weight, up_weight, down_weight):
        gate_out = F.linear(x, gate_weight)
        up_out = F.linear(x, up_weight)
        hidden = F.silu(gate_out) * up_out
        output = F.linear(hidden, down_weight)

        # 只保存输入和权重，不保存中间激活
        ctx.save_for_backward(x, gate_weight, up_weight, down_weight)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, gate_weight, up_weight, down_weight = ctx.saved_tensors

        # 重新前向计算
        gate_out = F.linear(x, gate_weight)
        up_out = F.linear(x, up_weight)
        gate_act = F.silu(gate_out)
        hidden = gate_act * up_out

        # grad through down_proj
        grad_hidden = grad_output.mm(down_weight)
        grad_down_weight = grad_output.t().mm(hidden)

        # grad through gate_act * up_out
        grad_gate_act = grad_hidden * up_out
        grad_up_out = grad_hidden * gate_act

        # grad through SiLU
        sigmoid_gate = torch.sigmoid(gate_out)
        grad_gate_out = grad_gate_act * (sigmoid_gate + gate_out * sigmoid_gate * (1.0 - sigmoid_gate))

        # grad through linear layers
        grad_x_from_gate = grad_gate_out.mm(gate_weight)
        grad_gate_weight = grad_gate_out.t().mm(x)

        grad_x_from_up = grad_up_out.mm(up_weight)
        grad_up_weight = grad_up_out.t().mm(x)

        grad_x = grad_x_from_gate + grad_x_from_up

        return grad_x, grad_gate_weight, grad_up_weight, grad_down_weight


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

    def forward(self, hidden_states):
        bsz, seq_len, hidden_size = hidden_states.shape
        hidden_flat = hidden_states.reshape(-1, hidden_size)

        # 1. Router
        router_logits = F.linear(hidden_flat, self.gate.weight)
        router_probs = F.softmax(router_logits, dtype=torch.float, dim=-1)
        top_k_weights, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        if self.norm_topk_prob:
            top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        top_k_weights = top_k_weights.to(hidden_flat.dtype)

        # 2. Routed Experts（单一 autograd Function，无128次 checkpoint 开销）
        routed_output = MoEExpertsFunction.apply(
            hidden_flat,
            self.experts.gate_up_proj,
            self.experts.down_proj,
            top_k_indices,
            top_k_weights,
            self.num_experts,
            self.top_k,
        )

        # 3. Shared Expert（自定义 autograd Function，不保存中间激活）
        shared_output = SharedExpertFunction.apply(
            hidden_flat,
            self.shared_expert.gate_proj.weight,
            self.shared_expert.up_proj.weight,
            self.shared_expert.down_proj.weight,
        )

        # 4. Combine + RMSNorm
        combined = routed_output + shared_output
        output = RMSNormFunction.apply(combined, self.post_norm.weight, self.variance_epsilon)

        return output.reshape(bsz, seq_len, hidden_size)
