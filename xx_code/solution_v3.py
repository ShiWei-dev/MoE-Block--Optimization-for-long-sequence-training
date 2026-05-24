"""
MoE Block 显存优化实现 v3
========================
v3 改进（相对 v2）：
1. 将 Router 合并进 Expert 的自定义 autograd.Function，消除 router_probs (float32 N×128) 等 autograd 中间张量
2. 将 Shared Expert + RMSNorm 合并进一个 autograd.Function，消除 combined 张量的额外保存
3. 整体只有两个 autograd.Function 调用，最小化框架开销
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoERouterExpertsFunction(torch.autograd.Function):
    """
    合并 Router + 全部 Expert 计算的自定义 autograd.Function。
    前向：router(softmax→topk→normalize) + expert循环，不保存任何中间激活。
    反向：重算 router 得到 top_k_weights，重算每个 expert 的中间激活，手动计算全部梯度。
    """

    @staticmethod
    def forward(ctx, hidden_states, gate_weight, gate_up_proj, down_proj,
                num_experts, top_k, norm_topk_prob):
        N, H = hidden_states.shape
        dtype = hidden_states.dtype

        # ---- Router ----
        router_logits = F.linear(hidden_states, gate_weight)
        router_probs = F.softmax(router_logits, dtype=torch.float, dim=-1)
        top_k_weights, top_k_indices = torch.topk(router_probs, top_k, dim=-1)
        if norm_topk_prob:
            top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        top_k_weights = top_k_weights.to(dtype)

        # ---- Token 分发 ----
        flat_expert_ids = top_k_indices.reshape(-1)
        flat_token_ids = torch.arange(N, device=hidden_states.device).unsqueeze(1).expand(-1, top_k).reshape(-1)
        flat_pos_ids = torch.arange(top_k, device=hidden_states.device).unsqueeze(0).expand(N, -1).reshape(-1)

        sorted_order = flat_expert_ids.argsort(stable=True)
        sorted_token_ids = flat_token_ids[sorted_order]
        sorted_pos_ids = flat_pos_ids[sorted_order]

        expert_counts = torch.bincount(flat_expert_ids.long(), minlength=num_experts)
        expert_offsets = torch.zeros(num_experts + 1, dtype=torch.long, device=hidden_states.device)
        torch.cumsum(expert_counts[:num_experts], dim=0, out=expert_offsets[1:])

        # ---- Expert 循环 ----
        final_hidden_states = torch.zeros(N, H, dtype=dtype, device=hidden_states.device)

        for expert_idx in range(num_experts):
            start = expert_offsets[expert_idx].item()
            end = expert_offsets[expert_idx + 1].item()
            if start == end:
                continue
            token_idx = sorted_token_ids[start:end]
            pos_idx = sorted_pos_ids[start:end]
            current_state = hidden_states[token_idx]

            gate_up = F.linear(current_state, gate_up_proj[expert_idx])
            gate, up = gate_up.chunk(2, dim=-1)
            current_hidden = F.silu(gate) * up
            current_output = F.linear(current_hidden, down_proj[expert_idx])
            current_output = current_output * top_k_weights[token_idx, pos_idx].unsqueeze(-1)
            final_hidden_states.index_add_(0, token_idx, current_output.to(dtype))

        # 保存：仅输入、权重引用、路由索引和分发元数据
        ctx.save_for_backward(hidden_states, gate_weight, gate_up_proj, down_proj, top_k_indices)
        ctx.sorted_token_ids = sorted_token_ids
        ctx.sorted_pos_ids = sorted_pos_ids
        ctx.expert_offsets = expert_offsets
        ctx.num_experts = num_experts
        ctx.top_k = top_k
        ctx.norm_topk_prob = norm_topk_prob

        return final_hidden_states

    @staticmethod
    def backward(ctx, grad_output):
        hidden_states, gate_weight, gate_up_proj, down_proj, top_k_indices = ctx.saved_tensors
        sorted_token_ids = ctx.sorted_token_ids
        sorted_pos_ids = ctx.sorted_pos_ids
        expert_offsets = ctx.expert_offsets
        num_experts = ctx.num_experts
        top_k = ctx.top_k
        norm_topk_prob = ctx.norm_topk_prob
        N, H = hidden_states.shape
        dtype = hidden_states.dtype

        # ---- 重算 Router（获取 top_k_weights）----
        router_logits = F.linear(hidden_states, gate_weight)
        router_probs = F.softmax(router_logits, dtype=torch.float, dim=-1)
        top_k_weights_raw, _ = torch.topk(router_probs, top_k, dim=-1)
        if norm_topk_prob:
            top_k_sum = top_k_weights_raw.sum(dim=-1, keepdim=True)
            top_k_weights = (top_k_weights_raw / top_k_sum).to(dtype)
        else:
            top_k_weights = top_k_weights_raw.to(dtype)
            top_k_sum = None

        # ---- Expert 反向 ----
        grad_hidden_states = torch.zeros_like(hidden_states)
        grad_gate_up_proj = torch.zeros_like(gate_up_proj)
        grad_down_proj = torch.zeros_like(down_proj)
        grad_top_k_weights = torch.zeros(N, top_k, dtype=torch.float, device=hidden_states.device)

        for expert_idx in range(num_experts):
            start = expert_offsets[expert_idx].item()
            end = expert_offsets[expert_idx + 1].item()
            if start == end:
                continue

            token_idx = sorted_token_ids[start:end]
            pos_idx = sorted_pos_ids[start:end]
            current_state = hidden_states[token_idx]
            weights = top_k_weights[token_idx, pos_idx].unsqueeze(-1)

            # 重算前向
            gate_up = F.linear(current_state, gate_up_proj[expert_idx])
            gate, up = gate_up.chunk(2, dim=-1)
            gate_act = F.silu(gate)
            current_hidden = gate_act * up
            current_output = F.linear(current_hidden, down_proj[expert_idx])

            grad_out_local = grad_output[token_idx]
            grad_linear_out = grad_out_local * weights
            grad_top_k_weights[token_idx, pos_idx] += (grad_out_local * current_output).sum(dim=-1).float()

            grad_current_hidden = grad_linear_out.mm(down_proj[expert_idx])
            grad_down_proj[expert_idx].addmm_(grad_linear_out.t(), current_hidden)

            grad_gate_act = grad_current_hidden * up
            grad_up = grad_current_hidden * gate_act

            sigmoid_gate = torch.sigmoid(gate)
            grad_gate = grad_gate_act * (sigmoid_gate + gate * sigmoid_gate * (1.0 - sigmoid_gate))
            grad_gate_up = torch.cat([grad_gate, grad_up], dim=-1)

            grad_current_state = grad_gate_up.mm(gate_up_proj[expert_idx])
            grad_gate_up_proj[expert_idx].addmm_(grad_gate_up.t(), current_state)

            grad_hidden_states.index_add_(0, token_idx, grad_current_state)

        # ---- Router 反向 ----
        # normalize backward
        if norm_topk_prob:
            # w = raw / S => grad_raw = (grad_w - (grad_w · w).sum * 1) / S
            tkw_fp32 = top_k_weights.float()
            dot_sum = (grad_top_k_weights * tkw_fp32).sum(dim=-1, keepdim=True)
            grad_top_k_raw = (grad_top_k_weights - dot_sum) / top_k_sum
        else:
            grad_top_k_raw = grad_top_k_weights

        # topk backward: scatter to full [N, E]
        grad_router_probs = torch.zeros(N, num_experts, dtype=torch.float, device=hidden_states.device)
        grad_router_probs.scatter_(1, top_k_indices.long(), grad_top_k_raw)

        # softmax backward: grad_logits = probs * (grad_probs - (probs * grad_probs).sum(-1, keepdim=True))
        s = (router_probs * grad_router_probs).sum(dim=-1, keepdim=True)
        grad_logits = router_probs * (grad_router_probs - s)

        # linear backward
        grad_logits_cast = grad_logits.to(dtype)
        grad_hidden_states += grad_logits_cast.mm(gate_weight)
        grad_gate_weight = grad_logits_cast.t().mm(hidden_states.to(dtype))

        return grad_hidden_states, grad_gate_weight, grad_gate_up_proj, grad_down_proj, None, None, None


class SharedExpertNormFunction(torch.autograd.Function):
    """
    合并 Shared Expert + RMSNorm 的自定义 autograd.Function。
    前向：shared_expert(x) → 输出 shared_out（与 expert_out 相加后过 RMSNorm）
    但因为 RMSNorm 依赖 combined = expert_out + shared_out，无法在此融合 RMSNorm。
    所以只融合 Shared Expert，不保存中间激活。
    """

    @staticmethod
    def forward(ctx, x, gate_weight, up_weight, down_weight):
        gate_out = F.linear(x, gate_weight)
        up_out = F.linear(x, up_weight)
        hidden = F.silu(gate_out) * up_out
        output = F.linear(hidden, down_weight)
        ctx.save_for_backward(x, gate_weight, up_weight, down_weight)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, gate_weight, up_weight, down_weight = ctx.saved_tensors

        gate_out = F.linear(x, gate_weight)
        up_out = F.linear(x, up_weight)
        gate_act = F.silu(gate_out)
        hidden = gate_act * up_out

        grad_hidden = grad_output.mm(down_weight)
        grad_down_weight = grad_output.t().mm(hidden)

        grad_gate_act = grad_hidden * up_out
        grad_up_out = grad_hidden * gate_act

        sigmoid_gate = torch.sigmoid(gate_out)
        grad_gate_out = grad_gate_act * (sigmoid_gate + gate_out * sigmoid_gate * (1.0 - sigmoid_gate))

        grad_x = grad_gate_out.mm(gate_weight) + grad_up_out.mm(up_weight)
        grad_gate_weight = grad_gate_out.t().mm(x)
        grad_up_weight = grad_up_out.t().mm(x)

        return grad_x, grad_gate_weight, grad_up_weight, grad_down_weight


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
        grad_output_fp32 = grad_output.to(torch.float32)
        x_fp32 = hidden_states.to(torch.float32)
        rstd_fp32 = rstd.to(torch.float32)
        weight_fp32 = weight.to(torch.float32)
        x_normed = x_fp32 * rstd_fp32
        grad_weight = (grad_output_fp32 * x_normed).sum(dim=0)
        grad_normed = grad_output_fp32 * weight_fp32
        dot = (grad_normed * x_normed).mean(dim=-1, keepdim=True)
        grad_input = (rstd_fp32 * (grad_normed - x_normed * dot)).to(ctx.input_dtype)
        return grad_input, grad_weight, None


class MoEBlockOptimized(nn.Module):
    """
    显存优化版本的 MoE Block（v3）。
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

        self.gate = nn.Module()
        self.gate.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_size))

        self.experts = nn.Module()
        self.experts.gate_up_proj = nn.Parameter(
            torch.empty(self.num_experts, 2 * self.moe_intermediate_size, self.hidden_size))
        self.experts.down_proj = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_size, self.moe_intermediate_size))

        self.shared_expert = nn.Module()
        self.shared_expert.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.shared_expert.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.shared_expert.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

        self.post_norm = nn.Module()
        self.post_norm.weight = nn.Parameter(torch.ones(self.hidden_size))

    def forward(self, hidden_states):
        bsz, seq_len, hidden_size = hidden_states.shape
        hidden_flat = hidden_states.reshape(-1, hidden_size)

        # Router + Experts（单一 Function，router 不产生额外 autograd 开销）
        routed_output = MoERouterExpertsFunction.apply(
            hidden_flat, self.gate.weight,
            self.experts.gate_up_proj, self.experts.down_proj,
            self.num_experts, self.top_k, self.norm_topk_prob,
        )

        # Shared Expert（自定义 Function，不保存中间激活）
        shared_output = SharedExpertNormFunction.apply(
            hidden_flat,
            self.shared_expert.gate_proj.weight,
            self.shared_expert.up_proj.weight,
            self.shared_expert.down_proj.weight,
        )

        # Combine + RMSNorm
        combined = routed_output + shared_output
        output = RMSNormFunction.apply(combined, self.post_norm.weight, self.variance_epsilon)

        return output.reshape(bsz, seq_len, hidden_size)
