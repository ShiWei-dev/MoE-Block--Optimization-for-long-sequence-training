"""
MoE Block 显存优化 v6 — 极致版
===============================
v6 核心改进:
1. Expert 循环内预分配复用缓冲区, 避免 128 次迭代中反复 alloc/free 导致 CUDA 碎片化
2. 减少 expert backward 中的中间张量数量 (原地操作, 合并计算)
3. 保留 v4/v5: 分块 Shared Expert, 分块 RMSNorm, Router 合并, 激活重计算
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

SHARED_CHUNK = 2048
NORM_CHUNK = 4096


class FusedAddRMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, weight, eps):
        dtype = a.dtype
        N, H = a.shape
        combined = a.add(b)
        output = torch.empty_like(combined)
        rstd = torch.empty(N, 1, dtype=dtype, device=a.device)
        C = min(N, NORM_CHUNK)
        for i in range(0, N, C):
            xc = combined[i:i+C].float()
            var = xc.pow(2).mean(-1, keepdim=True)
            r = torch.rsqrt(var + eps)
            rstd[i:i+C] = r.to(dtype)
            output[i:i+C] = (weight * (xc * r).to(dtype))
        ctx.save_for_backward(combined, weight, rstd)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        combined, weight, rstd = ctx.saved_tensors
        dtype = combined.dtype
        N, H = combined.shape
        grad_combined = torch.empty(N, H, dtype=dtype, device=combined.device)
        grad_weight = torch.zeros(H, dtype=torch.float32, device=combined.device)
        w_fp32 = weight.float()
        C = min(N, NORM_CHUNK)
        for i in range(0, N, C):
            go = grad_output[i:i+C].float()
            xc = combined[i:i+C].float()
            rc = rstd[i:i+C].float()
            xn = xc * rc
            grad_weight.add_((go * xn).sum(dim=0))
            gn = go * w_fp32
            dot = (gn * xn).mean(dim=-1, keepdim=True)
            grad_combined[i:i+C] = (rc * (gn - xn * dot)).to(dtype)
        return grad_combined, grad_combined, grad_weight, None


class MoERouterExpertsFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, gate_weight, gate_up_proj, down_proj,
                num_experts, top_k, norm_topk_prob):
        N, H = hidden_states.shape
        dtype = hidden_states.dtype
        I = gate_up_proj.shape[1] // 2  # moe_intermediate_size

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

        max_tokens = expert_counts.max().item() if N > 0 else 0

        # 预分配 expert 循环复用缓冲区
        buf_gu = torch.empty(max_tokens, 2 * I, dtype=dtype, device=hidden_states.device)
        buf_co = torch.empty(max_tokens, H, dtype=dtype, device=hidden_states.device)

        final = torch.zeros(N, H, dtype=dtype, device=hidden_states.device)
        for eidx in range(num_experts):
            s, e = expert_offsets[eidx].item(), expert_offsets[eidx + 1].item()
            if s == e:
                continue
            n = e - s
            tidx = sorted_token_ids[s:e]
            pidx = sorted_pos_ids[s:e]
            cs = hidden_states[tidx]

            # 复用缓冲区
            gu = buf_gu[:n]
            torch.mm(cs, gate_up_proj[eidx].t(), out=gu)
            g, u = gu[:, :I], gu[:, I:]
            ch = F.silu(g) * u
            co = buf_co[:n]
            torch.mm(ch, down_proj[eidx].t(), out=co)
            co.mul_(top_k_weights[tidx, pidx].unsqueeze(-1))
            final.index_add_(0, tidx, co)

        del buf_gu, buf_co

        ctx.save_for_backward(hidden_states, gate_weight, gate_up_proj, down_proj, top_k_indices)
        ctx.sorted_token_ids = sorted_token_ids
        ctx.sorted_pos_ids = sorted_pos_ids
        ctx.expert_offsets = expert_offsets
        ctx.num_experts = num_experts
        ctx.top_k = top_k
        ctx.norm_topk_prob = norm_topk_prob
        ctx.max_tokens = max_tokens
        return final

    @staticmethod
    def backward(ctx, grad_output):
        hidden_states, gate_weight, gate_up_proj, down_proj, top_k_indices = ctx.saved_tensors
        stids, spids, eoffs = ctx.sorted_token_ids, ctx.sorted_pos_ids, ctx.expert_offsets
        NE, top_k, norm_topk_prob = ctx.num_experts, ctx.top_k, ctx.norm_topk_prob
        max_tokens = ctx.max_tokens
        N, H = hidden_states.shape
        I = gate_up_proj.shape[1] // 2
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
        del rl, tkw_raw

        grad_hs = torch.zeros_like(hidden_states)
        grad_gup = torch.zeros_like(gate_up_proj)
        grad_dp = torch.zeros_like(down_proj)
        grad_tkw = torch.zeros(N, top_k, dtype=torch.float, device=hidden_states.device)

        # 预分配 backward 复用缓冲区
        buf_gu = torch.empty(max_tokens, 2 * I, dtype=dtype, device=hidden_states.device)
        buf_co = torch.empty(max_tokens, H, dtype=dtype, device=hidden_states.device)
        buf_ggu = torch.empty(max_tokens, 2 * I, dtype=dtype, device=hidden_states.device)

        for eidx in range(NE):
            s, e = eoffs[eidx].item(), eoffs[eidx + 1].item()
            if s == e:
                continue
            n = e - s
            tidx, pidx = stids[s:e], spids[s:e]
            cs = hidden_states[tidx]
            w = tkw[tidx, pidx].unsqueeze(-1)

            # 重算前向 (复用缓冲区)
            gu = buf_gu[:n]
            torch.mm(cs, gate_up_proj[eidx].t(), out=gu)
            g, u = gu[:, :I], gu[:, I:]
            ga = torch.sigmoid(g)  # 先算 sigmoid, 后面同时用于 silu 和梯度
            silu_g = g * ga        # silu(g) = g * sigmoid(g)
            ch = silu_g * u
            co = buf_co[:n]
            torch.mm(ch, down_proj[eidx].t(), out=co)

            gol = grad_output[tidx]
            glo = gol * w
            grad_tkw[tidx, pidx] += (gol * co).sum(dim=-1).float()

            gch = glo.mm(down_proj[eidx])
            grad_dp[eidx].addmm_(glo.t(), ch)

            # SiLU 梯度: d(silu)/dx = sigmoid + x*sigmoid*(1-sigmoid) = ga + g*ga*(1-ga) = ga*(1+g*(1-ga))
            # 合并 gate 和 up 梯度, 减少临时张量
            ggu = buf_ggu[:n]
            ggu[:, :I] = (gch * u) * (ga * (1.0 + g * (1.0 - ga)))  # grad_gate
            ggu[:, I:] = gch * silu_g                                  # grad_up

            grad_hs.index_add_(0, tidx, ggu.mm(gate_up_proj[eidx]))
            grad_gup[eidx].addmm_(ggu.t(), cs)

        del buf_gu, buf_co, buf_ggu, tkw

        # Router backward (分块)
        if norm_topk_prob:
            tkw_raw2, _ = torch.topk(rp, top_k, dim=-1)
            tks2 = tkw_raw2.sum(dim=-1, keepdim=True)
            tkw_f = (tkw_raw2 / tks2).float()
            ds = (grad_tkw * tkw_f).sum(dim=-1, keepdim=True)
            grad_raw = (grad_tkw - ds) / tks2
            del tkw_raw2, tks2, tkw_f
        else:
            grad_raw = grad_tkw

        grad_gw = torch.zeros_like(gate_weight)
        RC = min(N, NORM_CHUNK)
        for i in range(0, N, RC):
            rp_c = rp[i:i+RC]
            c_len = rp_c.shape[0]
            grad_rp_c = torch.zeros(c_len, NE, dtype=torch.float, device=hidden_states.device)
            grad_rp_c.scatter_(1, top_k_indices[i:i+RC].long(), grad_raw[i:i+RC])
            ss = (rp_c * grad_rp_c).sum(dim=-1, keepdim=True)
            grad_rl_c = (rp_c * (grad_rp_c - ss)).to(dtype)
            grad_hs[i:i+RC].addmm_(grad_rl_c, gate_weight)
            grad_gw.addmm_(grad_rl_c.t(), hidden_states[i:i+RC])
        del rp

        return grad_hs, grad_gw, grad_gup, grad_dp, None, None, None


class ChunkedSharedExpertFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gate_w, up_w, down_w):
        N, H = x.shape
        I = gate_w.shape[0]
        output = torch.empty(N, H, dtype=x.dtype, device=x.device)
        C = min(N, SHARED_CHUNK)
        for i in range(0, N, C):
            xc = x[i:i+C]
            go = F.linear(xc, gate_w)
            uo = F.linear(xc, up_w)
            output[i:i+C] = F.linear(F.silu(go) * uo, down_w)
        ctx.save_for_backward(x, gate_w, up_w, down_w)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, gate_w, up_w, down_w = ctx.saved_tensors
        N, H = x.shape
        I = gate_w.shape[0]
        dtype = x.dtype
        grad_x = torch.empty_like(x)
        grad_gate_w = torch.zeros(I, H, dtype=dtype, device=x.device)
        grad_up_w = torch.zeros(I, H, dtype=dtype, device=x.device)
        grad_down_w = torch.zeros(H, I, dtype=dtype, device=x.device)
        C = min(N, SHARED_CHUNK)
        for i in range(0, N, C):
            xc = x[i:i+C]
            goc = grad_output[i:i+C]
            go = F.linear(xc, gate_w)
            uo = F.linear(xc, up_w)
            sg = torch.sigmoid(go)
            silu_go = go * sg
            h = silu_go * uo
            gh = goc.mm(down_w)
            grad_down_w.addmm_(goc.t(), h)
            gga = gh * uo
            ggo = gga * (sg * (1.0 + go * (1.0 - sg)))
            guo = gh * silu_go
            grad_x[i:i+C] = ggo.mm(gate_w) + guo.mm(up_w)
            grad_gate_w.addmm_(ggo.t(), xc)
            grad_up_w.addmm_(guo.t(), xc)
        return grad_x, grad_gate_w, grad_up_w, grad_down_w


class MoEBlockOptimized(nn.Module):
    """
    显存极致优化版 MoE Block (v6)。与 MoEBlockBaseline 数学等价。

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

        output = FusedAddRMSNormFunction.apply(routed, shared, self.post_norm.weight, 1e-6)
        return output.reshape(bsz, seq_len, hidden_size)
