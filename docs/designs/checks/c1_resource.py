"""C1 resource accounting: what restricted (r of R) LoRA training saves for a weak client."""
GiB = 2**30
MODELS = {  # hidden, intermediate, layers, heads, kv_heads, vocab, tied  (HF config.json; LLaMA-2 standard)
    "Qwen2.5-0.5B": (896, 4864, 24, 14, 2, 151936, True),
    "Qwen2.5-1.5B": (1536, 8960, 28, 12, 2, 151936, True),
    "Qwen2.5-3B": (2048, 11008, 36, 16, 2, 151936, True),
    "LLaMA-2-7B": (4096, 11008, 32, 32, 32, 32000, False),
}
TOKENS = 4 * 512          # micro-batch 4 x seq 512

def shapes(h, I, L, H, KV, V, tied):
    hd = h // H; kv = KV * hd
    per_layer = 2 * h * h + 2 * h * kv + 3 * h * I + 2 * h
    N = L * per_layer + V * h * (1 if tied else 2)
    lora_io = L * (9 * h + 2 * kv + 3 * I)          # sum of (d_in + d_out) over q,k,v,o,gate,up,down
    return N, lora_io, kv

def activ(h, I, L, V):
    ckpt = 2 * TOKENS * h * L                        # bf16 layer inputs kept with gradient checkpointing
    layer = 2 * TOKENS * (16 * h + 3 * I)            # one recomputed layer's working set (rough)
    logits = 4 * TOKENS * V * 2                      # fp32 logits + their gradient
    return ckpt + layer + logits

rows = []
for name, cfg in MODELS.items():
    h, I, L, H, KV, V, tied = cfg
    N, io, kv = shapes(*cfg)
    base = 2 * N
    act = activ(h, I, L, V)
    for R, r in ((64, 8), (64, 16), (16, 4)):
        PR, Pr = R * io, r * io
        full_state = 16 * PR                         # fp32 param + grad + Adam m, v
        restr_state = 16 * Pr + 4 * Pr              # trained comps + amp start copy; frozen comps folded into base
        full_peak = base + act + full_state
        restr_peak = base + act + restr_state
        flops_full = 4 * N + 6 * PR                  # per token: frozen base fwd+act-bwd, LoRA fwd+bwd
        flops_restr = 4 * N + 6 * Pr + 2 * Pr        # + amp start-term forward
        rows.append((name, N / 1e9, R, r, PR / 1e6, base / GiB, act / GiB, full_state / GiB, restr_state / GiB,
                     full_peak / GiB, restr_peak / GiB, 1 - restr_peak / full_peak, 1 - flops_restr / flops_full,
                     4 * PR / 2**20, 4 * Pr / 2**20))
print("| model | N (B) | R / r | LoRA params at R (M) | base bf16 | activations* | LoRA state at R | LoRA state at r (+amp) | weak peak at R | weak peak at r | peak saved | FLOPs saved | upload fp32 at R / r (MiB) |")
print("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
for x in rows:
    print(f"| {x[0]} | {x[1]:.2f} | {x[2]} / {x[3]} | {x[4]:.1f} | {x[5]:.2f} | {x[6]:.2f} | {x[7]:.2f} | {x[8]:.2f} | "
          f"{x[9]:.2f} | {x[10]:.2f} | {100*x[11]:.0f}% | {100*x[12]:.1f}% | {x[13]:.0f} / {x[14]:.0f} |")
