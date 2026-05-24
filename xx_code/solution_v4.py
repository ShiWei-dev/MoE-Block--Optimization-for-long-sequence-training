"""
MoE Block 显存优化 v4 — 极致优化版
===================================
关键改进（相对 v2/v3）:
1. 分块 Shared Expert: 将 N×6144 中间激活分块处理，峰值从 N×6144×5 降至 chunk×6144×5
   128K下: 从 ~7.5GB 降至 ~60MB
2. 分块 RMSNorm: 避免创建完整的 N×H float32 中间张量
   128K下: 从 ~2GB 降至 ~64MB
3. Router 合并进 Expert Function (同 v3)
4. Expert 激活重计算 (同 v2/v3)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

SHARED_CHUNK = 2048  # shared expert 分块大小
NORM_CHUNK = 4096    # RMSNorm 分块大小


# ============================================================================
# 分块 RMSNorm
# ============================================================================
class RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        dtype = x.dtype
        N, H = x.shape
        output = torch.empty_like(x)
        rstd = torch.empty(N, 1, dtype=dtype, device=x.device)
        C = min(N, NORM_CHUNK)
        for i in range(0, N, C):
            xc = x[i:i+C].float()
            var = xc.pow(2).mean(-1, keepdim=True)
            r = torch.rsqrt(var + eps)
            rstd[i:i+C] = r.to(dtype)
            output[i:i+C] = (weight * (xc * r).to(dtype))
        ctx.save_for_backward(x, weight, rstd)
        ctx.eps = eps
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, rstd = ctx.saved_tensors
        dtype = x.dtype
        N, H = x.shape
        grad_input = torch.empty_like(x)
        grad_weight = torch.zeros(H, dtype=torch.float32, device=x.device)
        w_fp32 = weight.float()
        C = min(N, NORM_CHUNK)
        for i in range(0, N, C):
            go = grad_output[i:i+C].float()
            xc = x[i:i+C].float()
            rc = rstd[i:i+C].float()
            xn = xc * rc
            grad_weight.add_((go * xn).sum(dim=0))
            gn = go * w_fp32
            dot = (gn * xn).mean(dim=-1, keepdim=True)
            grad_input[i:i+C] = (rc * (gn - xn * dot)).to(dtype)
        return grad_input, grad_weight, None


# ============================================================================
# Router + Expert (合并, 激活重计算)
# ============================================================================
class MoERouterExpertsFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, gate_weight, gate_up_proj, down_proj,
                num_experts, top_k, norm_topk_prob):
        N, H = hidden_states.shape
        dtype = hidden_states.dtype

        router_logits = F.linear(hidden_states, gate_weight)
        router_probs = F.softmax(router_logits, dtype=torch.float, dim=-1)
        top_k_weights, top_k_indices = torch.topk(router_probs, top_k, dim=-1)
        if norm_topk_prob:
            top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        top_k_weights = top_k_weights.to(dtype)

        flat_expert_ids = top_k_indices.reshape(-1)
        flat_token_ids = torch.arange(N, device=hidden_states.device).unsqueeze(1).expand(-1, top_k).reshape(-1)
        flat_pos_ids = torch.arange(top_k, device=hidden_states.device).unsqueeze(0).expand(N, -1).reshape(-1)
        sorted_order = flat_expert_ids.argsort(stable=True)
        sorted_token_ids = flat_token_ids[sorted_order]
        sorted_pos_ids = flat_pos_ids[sorted_order]
        expert_counts = torch.bincount(flat_expert_ids.long(), minlength=num_experts)
        expert_offsets = torch.zeros(num_experts + 1, dtype=torch.long, device=hidden_states.device)
        torch.cumsum(expert_counts[:num_experts], dim=0, out=expert_offsets[1:])

        final = torch.zeros(N, H, dtype=dtype, device=hidden_states.device)
        for eidx in range(num_experts):
            s, e = expert_offsets[eidx].item(), expert_offsets[eidx + 1].item()
            if s == e:
                continue
            tidx = sorted_token_ids[s:e]
            pidx = sorted_pos_ids[s:e]
            cs = hidden_states[tidx]
            gu = F.linear(cs, gate_up_proj[eidx])
            g, u = gu.chunk(2, dim=-1)
            ch = F.silu(g) * u
            co = F.linear(ch, down_proj[eidx])
            co = co * top_k_weights[tidx, pidx].unsqueeze(-1)
            final.index_add_(0, tidx, co.to(dtype))

        ctx.save_for_backward(hidden_states, gate_weight, gate_up_proj, down_proj, top_k_indices)
        ctx.sorted_token_ids = sorted_token_ids
        ctx.sorted_pos_ids = sorted_pos_ids
        ctx.expert_offsets = expert_offsets
        ctx.num_experts = num_experts
        ctx.top_k = top_k
        ctx.norm_topk_prob = norm_topk_prob
        return final

    @staticmethod
    def backward(ctx, grad_output):
        hidden_states, gate_weight, gate_up_proj, down_proj, top_k_indices = ctx.saved_tensors
        stids = ctx.sorted_token_ids
        spids = ctx.sorted_pos_ids
        eoffs = ctx.expert_offsets
        NE = ctx.num_experts
        top_k = ctx.top_k
        norm_topk_prob = ctx.norm_topk_prob
        N, H = hidden_states.shape
        dtype = hidden_states.dtype

        # Recompute router
        rl = F.linear(hidden_states, gate_weight)
        rp = F.softmax(rl, dtype=torch.float, dim=-1)
        tkw_raw, _ = torch.topk(rp, top_k, dim=-1)
        if norm_topk_prob:
            tks = tkw_raw.sum(dim=-1, keepdim=True)
            tkw = (tkw_raw / tks).to(dtype)
        else:
            tkw = tkw_raw.to(dtype)
            tks = None

        grad_hs = torch.zeros_like(hidden_states)
        grad_gup = torch.zeros_like(gate_up_proj)
        grad_dp = torch.zeros_like(down_proj)
        grad_tkw = torch.zeros(N, top_k, dtype=torch.float, device=hidden_states.device)

        for eidx in range(NE):
            s, e = eoffs[eidx].item(), eoffs[eidx + 1].item()
            if s == e:
                continue
            tidx, pidx = stids[s:e], spids[s:e]
            cs = hidden_states[tidx]
            w = tkw[tidx, pidx].unsqueeze(-1)

            gu = F.linear(cs, gate_up_proj[eidx])
            g, u = gu.chunk(2, dim=-1)
            ga = F.silu(g)
            ch = ga * u
            co = F.linear(ch, down_proj[eidx])

            gol = grad_output[tidx]
            glo = gol * w
            grad_tkw[tidx, pidx] += (gol * co).sum(dim=-1).float()

            gch = glo.mm(down_proj[eidx])
            grad_dp[eidx].addmm_(glo.t(), ch)

            gga = gch * u
            ggu = gch * ga
            sg = torch.sigmoid(g)
            gg = gga * (sg + g * sg * (1.0 - sg))
            ggu_full = torch.cat([gg, ggu], dim=-1)

            grad_hs.index_add_(0, tidx, ggu_full.mm(gate_up_proj[eidx]))
            grad_gup[eidx].addmm_(ggu_full.t(), cs)

        # Router backward
        if norm_topk_prob:
            tkw_f = tkw.float()
            ds = (grad_tkw * tkw_f).sum(dim=-1, keepdim=True)
            grad_raw = (grad_tkw - ds) / tks
        else:
            grad_raw = grad_tkw
        grad_rp = torch.zeros(N, NE, dtype=torch.float, device=hidden_states.device)
        grad_rp.scatter_(1, top_k_indices.long(), grad_raw)
        ss = (rp * grad_rp).sum(dim=-1, keepdim=True)
        grad_rl = (rp * (grad_rp - ss)).to(dtype)
        grad_hs.addmm_(grad_rl, gate_weight)
        grad_gw = grad_rl.t().mm(hidden_states)

        return grad_hs, grad_gw, grad_gup, grad_dp, None, None, None


# ============================================================================
# 分块 Shared Expert (核心优化: 避免 N×6144 的完整中间张量)
# ============================================================================
class ChunkedSharedExpertFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gate_w, up_w, down_w):
        N, H = x.shape
        output = torch.empty(N, H, dtype=x.dtype, device=x.device)
        C = min(N, SHARED_CHUNK)
        for i in range(0, N, C):
            xc = x[i:i+C]
            go = F.linear(xc, gate_w)
            uo = F.linear(xc, up_w)
            h = F.silu(go) * uo
            output[i:i+C] = F.linear(h, down_w)
        ctx.save_for_backward(x, gate_w, up_w, down_w)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, gate_w, up_w, down_w = ctx.saved_tensors
        N, H = x.shape
        I = gate_w.shape[0]  # intermediate_size
        dtype = x.dtype
        grad_x = torch.empty_like(x)
        grad_gate_w = torch.zeros(I, H, dtype=dtype, device=x.device)
        grad_up_w = torch.zeros(I, H, dtype=dtype, device=x.device)
        grad_down_w = torch.zeros(H, I, dtype=dtype, device=x.device)
        C = min(N, SHARED_CHUNK)
        for i in range(0, N, C):
            xc = x[i:i+C]
            goc = grad_output[i:i+C]
            # Recompute forward
            go = F.linear(xc, gate_w)
            uo = F.linear(xc, up_w)
            ga = F.silu(go)
            h = ga * uo
            # Backward
            gh = goc.mm(down_w)
            grad_down_w.addmm_(goc.t(), h)
            gga = gh * uo
            guo = gh * ga
            sg = torch.sigmoid(go)
            ggo = gga * (sg + go * sg * (1.0 - sg))
            grad_x[i:i+C] = ggo.mm(gate_w) + guo.mm(up_w)
            grad_gate_w.addmm_(ggo.t(), xc)
            grad_up_w.addmm_(guo.t(), xc)
        return grad_x, grad_gate_w, grad_up_w, grad_down_w


# ============================================================================
# MoEBlockOptimized
# ============================================================================
class MoEBlockOptimized(nn.Module):
    """
    显存极致优化版 MoE Block (v4)。与 MoEBlockBaseline 数学等价。

    权重结构与 baseline 一致:
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
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

        self.gate = nn.Module()
        self.gate.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_size))

        self.experts = nn.Module()
        self.experts.gate_up_proj = nn.Parameter(
            torch.empty(self.num_experts, 2 * config.moe_intermediate_size, self.hidden_size))
        self.experts.down_proj = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_size, config.moe_intermediate_size))

        self.shared_expert = nn.Module()
        self.shared_expert.gate_proj = nn.Linear(self.hidden_size, config.intermediate_size, bias=False)
        self.shared_expert.up_proj = nn.Linear(self.hidden_size, config.intermediate_size, bias=False)
        self.shared_expert.down_proj = nn.Linear(config.intermediate_size, self.hidden_size, bias=False)

        self.post_norm = nn.Module()
        self.post_norm.weight = nn.Parameter(torch.ones(self.hidden_size))

    def forward(self, hidden_states):
        bsz, seq_len, hidden_size = hidden_states.shape
        hf = hidden_states.reshape(-1, hidden_size)

        routed = MoERouterExpertsFunction.apply(
            hf, self.gate.weight, self.experts.gate_up_proj, self.experts.down_proj,
            self.num_experts, self.top_k, self.norm_topk_prob)

        shared = ChunkedSharedExpertFunction.apply(
            hf, self.shared_expert.gate_proj.weight,
            self.shared_expert.up_proj.weight, self.shared_expert.down_proj.weight)

        combined = routed + shared
        output = RMSNormFunction.apply(combined, self.post_norm.weight, 1e-6)
        return output.reshape(bsz, seq_len, hidden_size)
