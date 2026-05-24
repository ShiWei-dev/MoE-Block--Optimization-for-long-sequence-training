"""
Baseline vs Optimized 对比脚本（含评分模拟）
==========================================
模拟真实评测环境，输出：
1. 正确性检查（前向、输入梯度、参数梯度）
2. 每个序列长度下的性能对比
3. 模拟评分（显存60% + 速度40%）

用法:
    python compare_baseline.py [--seq-lens 8192,131072] [--warmup 3] [--measure 10]
"""

import argparse
import time
import sys
import os
from types import SimpleNamespace

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '附件'))
# pyrefly: ignore [missing-import]
from baseline import MoEBlockBaseline
from solution import MoEBlockOptimized


def build_config():
    return SimpleNamespace(
        hidden_size=2048, intermediate_size=6144, moe_intermediate_size=768,
        num_experts=128, num_experts_per_tok=8, norm_topk_prob=True,
    )


def initialize_weights(model):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name == "gate.weight":
                param.normal_(mean=0.0, std=0.02)
            else:
                param.normal_(mean=0.0, std=0.01)


def run_step(model, x):
    model.zero_grad(set_to_none=True)
    x = x.detach().clone().requires_grad_(True)
    y = model(x)
    loss = y.sum()
    loss.backward()
    return loss, y, x


def benchmark_model(model, x, device, warmup_steps=3, measure_steps=10):
    for _ in range(warmup_steps):
        run_step(model, x)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()

    step_times = []
    for _ in range(measure_steps):
        torch.cuda.synchronize()
        start = time.perf_counter()
        run_step(model, x)
        torch.cuda.synchronize()
        step_times.append(time.perf_counter() - start)

    peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    avg_ms = sum(step_times) / len(step_times) * 1000
    min_ms = min(step_times) * 1000
    max_ms = max(step_times) * 1000
    return peak_mb, avg_ms, min_ms, max_ms


def cosine_sim(a, b):
    a_flat, b_flat = a.float().reshape(-1), b.float().reshape(-1)
    a_n = torch.linalg.vector_norm(a_flat).item()
    b_n = torch.linalg.vector_norm(b_flat).item()
    if a_n == 0 and b_n == 0:
        return 1.0
    if a_n == 0 or b_n == 0:
        return 0.0
    return F.cosine_similarity(a_flat, b_flat, dim=0).item()


def relative_l2(ref, val, eps=1e-12):
    r, v = ref.float().reshape(-1), val.float().reshape(-1)
    return torch.linalg.vector_norm(r - v).item() / (torch.linalg.vector_norm(r).item() + eps)


def check_correctness(baseline_model, optimized_model, device, dtype, seq_len=2048):
    torch.manual_seed(42)
    x = torch.randn(1, seq_len, 2048, device=device, dtype=dtype)
    x_ref = x.detach().clone().requires_grad_(True)
    x_sub = x.detach().clone().requires_grad_(True)

    y_ref = baseline_model(x_ref)
    y_sub = optimized_model(x_sub)

    fwd_ok = torch.allclose(y_ref.float(), y_sub.float(), rtol=2e-2, atol=1e-3)
    fwd_max = (y_ref.float() - y_sub.float()).abs().max().item()

    grad_out = torch.randn_like(y_ref)
    (y_ref * grad_out).sum().backward()
    (y_sub * grad_out).sum().backward()

    ig_cos = cosine_sim(x_ref.grad, x_sub.grad)
    ig_rl2 = relative_l2(x_ref.grad, x_sub.grad)
    ig_ok = ig_cos >= 0.995 and ig_rl2 <= 1e-2

    # 参数梯度
    param_names = [
        "gate.weight", "experts.gate_up_proj", "experts.down_proj",
        "shared_expert.gate_proj.weight", "shared_expert.up_proj.weight",
        "shared_expert.down_proj.weight", "post_norm.weight",
    ]
    param_results = []
    all_param_ok = True
    ref_params = dict(baseline_model.named_parameters())
    sub_params = dict(optimized_model.named_parameters())
    for pname in param_names:
        rp, sp = ref_params.get(pname), sub_params.get(pname)
        if rp is None or sp is None or rp.grad is None or sp.grad is None:
            param_results.append((pname, False, 0.0, 0.0))
            all_param_ok = False
            continue
        c = cosine_sim(rp.grad, sp.grad)
        r = relative_l2(rp.grad, sp.grad)
        ok = c >= 0.995 and r <= 1e-2
        param_results.append((pname, ok, c, r))
        if not ok:
            all_param_ok = False

    return {
        "fwd_ok": fwd_ok, "fwd_max": fwd_max,
        "ig_ok": ig_ok, "ig_cos": ig_cos, "ig_rl2": ig_rl2,
        "param_results": param_results, "all_param_ok": all_param_ok,
        "all_ok": fwd_ok and ig_ok and all_param_ok,
    }


def compute_scores(results_by_seq):
    """
    模拟评分：
    - 显存得分 = mem_reduction_pct (相对 baseline 的显存下降百分比，0~100)
    - 速度得分 = speed_improvement_pct (相对 baseline 的速度提升百分比，负值截断为0)
    - 总分 = 0.6 * 显存得分 + 0.4 * 速度得分
    """
    scores = []
    for seq_len, data in results_by_seq.items():
        base = data.get("Baseline")
        opt = data.get("Optimized")
        correct = data.get("correct", True)

        if not correct:
            scores.append({"seq_len": seq_len, "mem_score": 0, "speed_score": 0, "total": 0, "note": "正确性未通过"})
            continue
        if base is None:
            if opt is not None:
                scores.append({"seq_len": seq_len, "mem_score": 100, "speed_score": 100, "total": 100, "note": "Baseline OOM"})
            else:
                scores.append({"seq_len": seq_len, "mem_score": 0, "speed_score": 0, "total": 0, "note": "Both OOM"})
            continue
        if opt is None:
            scores.append({"seq_len": seq_len, "mem_score": 0, "speed_score": 0, "total": 0, "note": "Optimized OOM"})
            continue

        base_mem, base_time = base
        opt_mem, opt_time = opt

        mem_reduction = max(0, (base_mem - opt_mem) / base_mem * 100)
        speed_improvement = max(0, (base_time - opt_time) / base_time * 100)

        mem_score = mem_reduction
        speed_score = speed_improvement
        total = 0.6 * mem_score + 0.4 * speed_score

        scores.append({
            "seq_len": seq_len, "mem_score": mem_score,
            "speed_score": speed_score, "total": total, "note": "",
        })

    return scores


def parse_seq_lengths(raw):
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser(description="Baseline vs Optimized 对比 + 评分")
    parser.add_argument("--seq-lens", type=str, default="8192,131072")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--measure", type=int, default=10)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--skip-correctness", action="store_true")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "需要 CUDA 环境"
    device = torch.device("cuda")
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]
    seq_lengths = parse_seq_lengths(args.seq_lens)
    config = build_config()

    print("=" * 100)
    print("  MoE Block 显存优化 — Baseline vs Optimized 对比 + 模拟评分")
    print("=" * 100)
    print(f"Config: B={args.batch_size}, H={config.hidden_size}, "
          f"moe_intermediate={config.moe_intermediate_size}, "
          f"shared_intermediate={config.intermediate_size}")
    print(f"MoE: num_experts={config.num_experts}, top_k={config.num_experts_per_tok}")
    print(f"dtype={dtype}, warmup={args.warmup}, measure={args.measure}")
    print()

    # ---- 正确性检查 ----
    correctness_ok = True
    if not args.skip_correctness:
        print("=" * 60)
        print("  正确性检查")
        print("=" * 60)
        torch.cuda.empty_cache()
        torch.manual_seed(42)

        baseline = MoEBlockBaseline(config).to(device=device, dtype=dtype)
        initialize_weights(baseline)
        optimized = MoEBlockOptimized(config).to(device=device, dtype=dtype)
        optimized.load_state_dict(baseline.state_dict(), strict=True)

        result = check_correctness(baseline, optimized, device, dtype)

        print(f"  前向输出: {'通过 ✅' if result['fwd_ok'] else '不通过 ❌'} "
              f"(max_abs_diff={result['fwd_max']:.6e})")
        print(f"  输入梯度: {'通过 ✅' if result['ig_ok'] else '不通过 ❌'} "
              f"(cosine_sim={result['ig_cos']:.6f}, rel_l2={result['ig_rl2']:.6e})")

        for pname, ok, c, r in result["param_results"]:
            status = '通过 ✅' if ok else '不通过 ❌'
            print(f"  参数梯度({pname}): {status} (cosine_sim={c:.6f}, rel_l2={r:.6e})")

        correctness_ok = result["all_ok"]
        print(f"\n  总结: {'所有检查项通过 ✅' if correctness_ok else '存在未通过的检查项 ❌'}")

        del baseline, optimized
        torch.cuda.empty_cache()
        print()

    # ---- Benchmark 对比 ----
    print("=" * 60)
    print("  性能对比")
    print("=" * 60)
    print(f"{'SeqLen':>8s} | {'Model':>10s} | {'Peak Mem (MB)':>14s} | "
          f"{'Avg (ms)':>10s} | {'Min (ms)':>10s} | {'Max (ms)':>10s} | "
          f"{'Mem Δ':>10s} | {'Speed Δ':>10s}")
    print("-" * 106)

    results_by_seq = {}
    state_dict = None

    for seq_len in seq_lengths:
        results = {}
        results_by_seq[seq_len] = {"correct": correctness_ok}

        for label, ModelCls in [("Baseline", MoEBlockBaseline), ("Optimized", MoEBlockOptimized)]:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            model = None
            x = None

            try:
                torch.manual_seed(42)
                model = ModelCls(config).to(device=device, dtype=dtype)
                if label == "Baseline":
                    initialize_weights(model)
                    state_dict = {k: v.clone() for k, v in model.state_dict().items()}
                else:
                    model.load_state_dict(state_dict, strict=True)

                x = torch.randn(args.batch_size, seq_len, config.hidden_size, device=device, dtype=dtype)
                peak_mb, avg_ms, min_ms, max_ms = benchmark_model(model, x, device, args.warmup, args.measure)
                results[label] = (peak_mb, avg_ms, min_ms, max_ms)
                results_by_seq[seq_len][label] = (peak_mb, avg_ms)

            except torch.cuda.OutOfMemoryError:
                results[label] = None
                results_by_seq[seq_len][label] = None
            finally:
                del model, x
                torch.cuda.empty_cache()

        for label in ["Baseline", "Optimized"]:
            r = results.get(label)
            if r is None:
                print(f"{seq_len:>8d} | {label:>10s} | {'OOM':>14s} | "
                      f"{'--':>10s} | {'--':>10s} | {'--':>10s} | {'':>10s} | {'':>10s}")
            else:
                peak_mb, avg_ms, min_ms, max_ms = r
                mem_delta, speed_delta = "", ""
                if label == "Optimized" and results.get("Baseline") is not None:
                    bp, bt = results["Baseline"][:2]
                    mem_delta = f"{(bp[0] if isinstance(bp, tuple) else bp - peak_mb) / (bp[0] if isinstance(bp, tuple) else bp) * 100:+.1f}%"
                    # Simplified
                    base_peak, base_avg = results["Baseline"][0], results["Baseline"][1]
                    mem_pct = (base_peak - peak_mb) / base_peak * 100
                    speed_pct = (base_avg - avg_ms) / base_avg * 100
                    mem_delta = f"{mem_pct:+.1f}%"
                    speed_delta = f"{speed_pct:+.1f}%"
                elif label == "Optimized" and results.get("Baseline") is None:
                    mem_delta, speed_delta = "BASE OOM", "BASE OOM"

                print(f"{seq_len:>8d} | {label:>10s} | {peak_mb:>14.2f} | "
                      f"{avg_ms:>10.2f} | {min_ms:>10.2f} | {max_ms:>10.2f} | "
                      f"{mem_delta:>10s} | {speed_delta:>10s}")

    # ---- 评分 ----
    print()
    print("=" * 60)
    print("  模拟评分 (显存 60% + 速度 40%)")
    print("=" * 60)
    scores = compute_scores(results_by_seq)

    print(f"{'SeqLen':>8s} | {'显存得分':>10s} | {'速度得分':>10s} | {'加权总分':>10s} | {'备注':>12s}")
    print("-" * 70)

    total_mem_score = 0
    total_speed_score = 0
    total_score = 0
    valid_count = 0

    for s in scores:
        note = s["note"] if s["note"] else "--"
        print(f"{s['seq_len']:>8d} | {s['mem_score']:>10.2f} | {s['speed_score']:>10.2f} | "
              f"{s['total']:>10.2f} | {note:>12s}")
        total_mem_score += s["mem_score"]
        total_speed_score += s["speed_score"]
        total_score += s["total"]
        valid_count += 1

    if valid_count > 0:
        print("-" * 70)
        avg_mem = total_mem_score / valid_count
        avg_speed = total_speed_score / valid_count
        avg_total = total_score / valid_count
        print(f"{'平均':>8s} | {avg_mem:>10.2f} | {avg_speed:>10.2f} | {avg_total:>10.2f} |")
        print()
        print(f"  📊 最终模拟得分: {avg_total:.2f} / 100")
        print(f"     显存贡献: {0.6 * avg_mem:.2f} (权重60%)")
        print(f"     速度贡献: {0.4 * avg_speed:.2f} (权重40%)")

    print()
    print("说明:")
    print("  Mem Δ: 正数 = 优化版显存更低 (越大越好)")
    print("  Speed Δ: 正数 = 优化版更快 (越大越好)")
    print("  显存得分 = max(0, 显存下降百分比)")
    print("  速度得分 = max(0, 速度提升百分比)")
    print("  总分 = 0.6×显存得分 + 0.4×速度得分")


if __name__ == "__main__":
    main()
