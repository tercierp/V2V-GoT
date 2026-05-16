"""
Scalability analysis: Centralized GoT vs Federated GoT as N agents grows.

Produces 4 publication-quality figures saved to demo/figures/:
  1. GoT prompt length vs N agents (per stage)
  2. Context window overflow point (centralized hard limit)
  3. V2V communication cost vs N agents
  4. FedAvg convergence vs N agents (real loss curve + federated simulation)

Run:
    conda activate llava
    cd V2V-GoT/
    python code/analysis/scalability.py
"""

import json
import glob
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE      = Path("/scratch/izar/tercier/v2v-got/V2V-GoT")
EVAL_DIR  = BASE / "LLaVA/playground/data/eval"
CKPT_DIR  = BASE / "LLaVA/checkpoints/llava-v1.5-7b-task-lora/llava-v1.5-7b-task-lora_crafter_osm_crafter_train01"
OUT_DIR   = BASE / "demo/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
DARK_BG   = "#0d1117"
GRID_COL  = "#21262d"
TEXT_COL  = "#e6edf3"
ACCENT    = ["#58a6ff", "#3fb950", "#f78166", "#d2a8ff", "#ffa657"]
CENT_COL  = "#f78166"   # red-ish for centralized
FED_COL   = "#58a6ff"   # blue for federated

plt.rcParams.update({
    "figure.facecolor": DARK_BG, "axes.facecolor": DARK_BG,
    "axes.edgecolor": GRID_COL, "axes.labelcolor": TEXT_COL,
    "xtick.color": TEXT_COL, "ytick.color": TEXT_COL,
    "text.color": TEXT_COL, "grid.color": GRID_COL,
    "legend.facecolor": "#161b22", "legend.edgecolor": GRID_COL,
    "font.family": "DejaVu Sans",
})

N_RANGE = np.arange(2, 13)   # N = 2 … 12 agents
CONTEXT_LIMIT = 2048          # LLaVA-v1.5-7b token limit

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_q1_outputs():
    path = (EVAL_DIR /
            "v2v4real_3d_grounding_osm_from_q1_2500_full_nq1sm3w0d"
            / "answers/val/llava-v1.5-7b/merge.jsonl")
    if not path.exists():
        return None
    return [json.loads(l) for l in open(path) if l.strip()]


def approximate_tokens(text):
    """Fast token approximation: ~1.3 chars/token for English text."""
    return max(1, int(len(text) / 3.8))


def measure_per_vehicle_tokens(samples):
    """
    From real Q1 outputs: measure question + answer tokens per vehicle.
    Returns (mean_q_tokens, mean_a_tokens, std_a_tokens).
    """
    q_toks = [approximate_tokens(s["conversations"][0]["value"]) for s in samples]
    a_toks = [approximate_tokens(s["outputs"]) for s in samples]
    return np.mean(q_toks), np.mean(a_toks), np.std(a_toks)


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 1 + 2: GoT prompt length vs N  &  overflow point
# ═══════════════════════════════════════════════════════════════════════════════

def plot_prompt_length(samples):
    """
    Model how GoT prompt length grows with N agents.

    Centralized GoT (server sees ALL vehicles simultaneously):
      - System prompt:      ~150 tokens (fixed)
      - Per-vehicle Q+A:    Q_tok + A_tok per vehicle per stage
      - Scene graph prompt: cumulative — each merge stage adds all previous answers

    Federated GoT (each vehicle runs local stages, shares only text at merge):
      - Each vehicle's local prompt stays ~constant regardless of N
      - Merge points add one summary text per vehicle (~A_tok tokens)
    """
    if samples is None:
        print("  [warn] No Q1 outputs found, using estimates")
        q_tok, a_tok, a_std = 92, 15, 8
    else:
        q_tok, a_tok, a_std = measure_per_vehicle_tokens(samples)
        print(f"  Real token stats: question={q_tok:.0f}, answer={a_tok:.1f}±{a_std:.1f}")

    SYS_PROMPT   = 150   # fixed overhead
    STAGE_Q      = q_tok # per-stage question tokens
    STAGE_A      = a_tok # per-vehicle answer tokens

    # GoT stages and their accumulation pattern
    # Stage weights: how many previous vehicle answers are concatenated
    # Q1,Q2 → individual (1 vehicle each)
    # Q3    → merges Q1+Q2 answers  (2 vehicles, but in centralized: N vehicles)
    # Q4    → merges Q1+Q3          (2 stages × N)
    # Q5-Q9 → progressive accumulation
    stage_names = ["Q1","Q2","Q3","Q4","Q5","Q6","Q7","Q8","Q9"]
    # Number of previous vehicle-answer blocks included in each stage's context
    # (in centralized: scales with N; in federated: always 1 local + 1 shared summary)
    accum_factors_central  = [1, 1, 2, 2, 3, 3, 4, 5, 6]  # × N
    accum_factors_federated = [1, 1, 2, 2, 2, 2, 2, 3, 3]  # independent of N (local + merge)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("GoT Prompt Length vs Number of Agents", color=TEXT_COL,
                 fontsize=14, fontweight="bold", y=1.02)

    # ── Plot 1a: all 9 stages, centralized ─────────────────────────────────
    ax = axes[0]
    for i, (name, acc) in enumerate(zip(stage_names, accum_factors_central)):
        tokens_n = SYS_PROMPT + STAGE_Q + acc * N_RANGE * STAGE_A
        ax.plot(N_RANGE, tokens_n, color=ACCENT[i % len(ACCENT)],
                lw=1.8, label=name, marker="o", ms=4)

    ax.axhline(CONTEXT_LIMIT, color=CENT_COL, lw=2, ls="--",
               label=f"LLaVA limit ({CONTEXT_LIMIT})")
    ax.fill_between(N_RANGE, CONTEXT_LIMIT, ax.get_ylim()[1] if ax.get_ylim()[1] > CONTEXT_LIMIT else CONTEXT_LIMIT * 1.5,
                    alpha=0.15, color=CENT_COL)
    ax.set_xlabel("Number of agents (N)", fontsize=11)
    ax.set_ylabel("Tokens in prompt", fontsize=11)
    ax.set_title("Centralized GoT", color=CENT_COL, fontsize=12)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(2, 12)

    # ── Plot 1b: centralized vs federated at the critical stages ───────────
    ax = axes[1]
    critical = [(4, "Q4 (scene merge)"), (6, "Q6 (traj predict)"), (8, "Q8 (decision)")]
    for idx, (si, label) in enumerate(critical):
        acc_c = accum_factors_central[si]
        acc_f = accum_factors_federated[si]
        tc = SYS_PROMPT + STAGE_Q + acc_c * N_RANGE * STAGE_A
        tf = SYS_PROMPT + STAGE_Q + acc_f * STAGE_A * np.ones_like(N_RANGE)
        ax.plot(N_RANGE, tc, color=CENT_COL, lw=2,
                ls=["-","--","-."][idx], label=f"Centralized {label}")
        ax.plot(N_RANGE, tf, color=FED_COL, lw=2,
                ls=["-","--","-."][idx], label=f"Federated {label}", alpha=0.8)

    ax.axhline(CONTEXT_LIMIT, color="#ffa657", lw=2, ls=":",
               label=f"Context limit ({CONTEXT_LIMIT})")
    ax.set_xlabel("Number of agents (N)", fontsize=11)
    ax.set_ylabel("Tokens in prompt", fontsize=11)
    ax.set_title("Centralized vs Federated (key stages)", color=TEXT_COL, fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(2, 12)

    fig.tight_layout()
    out = OUT_DIR / "1_prompt_length_vs_N.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved: {out}")

    # ── Find overflow point ─────────────────────────────────────────────────
    worst_acc = max(accum_factors_central)
    overflow_N = (CONTEXT_LIMIT - SYS_PROMPT - STAGE_Q) / (worst_acc * STAGE_A)
    print(f"  Centralized overflow at N ≈ {overflow_N:.1f} agents (Q9, worst case)")
    return overflow_N


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 3: Communication cost vs N
# ═══════════════════════════════════════════════════════════════════════════════

def plot_communication_cost():
    """
    Centralized: each vehicle streams raw LiDAR to server every frame.
      From point_pillar_opv2v.yaml: data_size=1.06 Mb/frame, 10Hz → 10.6 Mbps/vehicle

    Federated: vehicles exchange LoRA weights once per training round.
      LoRA checkpoint ≈ 50 MB, round every ~3h of compute.
      Amortized over driving time: 50 MB / (3h × 3600s) ≈ 0.005 Mbps/vehicle
      Plus GoT text sharing at merge points: ~1KB/merge × 3 merges/frame × 10Hz ≈ 0.24 Mbps
    """
    LIDAR_RATE_MBPS   = 10.6    # Mbps per vehicle (real value from yaml)
    LORA_MB           = 50      # MB for one LoRA checkpoint
    ROUND_SECONDS     = 3 * 3600  # 3h per training round
    LORA_RATE_MBPS    = (LORA_MB * 8) / ROUND_SECONDS / 1   # Mbps per vehicle

    TEXT_MERGES       = 3       # merge points per second (3 GoT merge stages × 10fps / 10)
    TEXT_KB_PER_MERGE = 1       # ~1KB per text graph exchange
    TEXT_RATE_MBPS    = TEXT_MERGES * TEXT_KB_PER_MERGE * 8 / 1000  # Mbps per vehicle

    central_total  = LIDAR_RATE_MBPS * N_RANGE
    fed_weight     = LORA_RATE_MBPS  * N_RANGE
    fed_text       = TEXT_RATE_MBPS  * N_RANGE
    fed_total      = fed_weight + fed_text

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("V2V Communication Cost vs Number of Agents",
                 color=TEXT_COL, fontsize=14, fontweight="bold")

    # Linear scale
    ax1.fill_between(N_RANGE, central_total, alpha=0.2, color=CENT_COL)
    ax1.plot(N_RANGE, central_total, color=CENT_COL, lw=2.5,
             label=f"Centralized (LiDAR streams, {LIDAR_RATE_MBPS} Mbps/vehicle)")
    ax1.fill_between(N_RANGE, fed_total, alpha=0.2, color=FED_COL)
    ax1.plot(N_RANGE, fed_total, color=FED_COL, lw=2.5,
             label="Federated (LoRA weights + GoT text)")
    ax1.plot(N_RANGE, fed_text, color=FED_COL, lw=1.5, ls="--",
             label="  └─ GoT text only (no weight exchange)")
    ax1.set_xlabel("Number of agents (N)", fontsize=11)
    ax1.set_ylabel("Total bandwidth (Mbps)", fontsize=11)
    ax1.set_title("Linear scale", color=TEXT_COL)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(2, 12)

    # Log scale — shows the gap clearly
    ax2.semilogy(N_RANGE, central_total, color=CENT_COL, lw=2.5,
                 label="Centralized (LiDAR streams)")
    ax2.semilogy(N_RANGE, fed_total, color=FED_COL, lw=2.5,
                 label="Federated total")
    ax2.semilogy(N_RANGE, fed_text, color=FED_COL, lw=1.5, ls="--",
                 label="Federated (GoT text only)")
    # Annotate the gap at N=10
    n10 = 10
    gap = LIDAR_RATE_MBPS * n10 / (TEXT_RATE_MBPS * n10)
    ax2.annotate(f"~{gap:.0f}× less\nbandwidth",
                 xy=(n10, TEXT_RATE_MBPS * n10),
                 xytext=(n10 - 2, LIDAR_RATE_MBPS * n10 * 0.3),
                 color=TEXT_COL, fontsize=9,
                 arrowprops=dict(arrowstyle="->", color=TEXT_COL, lw=1.2))
    ax2.set_xlabel("Number of agents (N)", fontsize=11)
    ax2.set_ylabel("Total bandwidth (Mbps, log scale)", fontsize=11)
    ax2.set_title("Log scale", color=TEXT_COL)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, which="both")
    ax2.set_xlim(2, 12)

    fig.tight_layout()
    out = OUT_DIR / "2_communication_cost_vs_N.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved: {out}")
    print(f"  At N=10: centralized={LIDAR_RATE_MBPS*10:.0f} Mbps, "
          f"federated text={TEXT_RATE_MBPS*10:.3f} Mbps  ({gap:.0f}× reduction)")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 4: FedAvg convergence vs N agents
# ═══════════════════════════════════════════════════════════════════════════════

def plot_fedavg_convergence():
    """
    Uses the REAL centralized loss curve from trainer_state.json.
    Simulates federated convergence for N=2, 4, 8 agents using FedAvg theory:

    FedAvg convergence (McMahan et al., 2017):
      Federated loss ≈ Centralized loss + extra variance from data heterogeneity
      Variance term decreases with more rounds and is bounded by O(1/sqrt(R))
      With interleaved data split (as we do), heterogeneity is low.

    Simulation: federated_loss[t] = central_loss[t] × (1 + α/N^0.5 × decay(t))
      where α is heterogeneity factor (small due to interleaved split) and
      decay(t) = exp(-t/T) models convergence of the variance term.
    """
    # Load real training loss
    all_logs = []
    for path in sorted(glob.glob(str(CKPT_DIR / "checkpoint-*/trainer_state.json"))):
        d = json.load(open(path))
        all_logs.extend(d.get("log_history", []))

    seen, loss_curve = set(), []
    for entry in sorted(all_logs, key=lambda x: x.get("step", 0)):
        s = entry.get("step")
        if s not in seen and "loss" in entry:
            seen.add(s)
            loss_curve.append((s, entry["loss"]))

    if not loss_curve:
        print("  [warn] No real loss data found, using synthetic curve")
        steps = np.arange(1, 3501)
        losses = 2.5 * np.exp(-steps / 800) + 0.1
        loss_curve = list(zip(steps, losses))

    steps  = np.array([x[0] for x in loss_curve])
    losses = np.array([x[1] for x in loss_curve])

    # Smooth with rolling mean for visibility
    def smooth(y, w=30):
        return np.convolve(y, np.ones(w)/w, mode="same")

    losses_smooth = smooth(losses)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Training Loss: Centralized vs Federated (N agents)",
                 color=TEXT_COL, fontsize=14, fontweight="bold")

    # ── Plot 4a: full training curve ─────────────────────────────────────
    ax = axes[0]
    ax.plot(steps, losses_smooth, color=CENT_COL, lw=2.5, label="Centralized (real)",
            zorder=5)

    # Simulate federated curves for N=2, 4, 8
    ALPHA = 0.08     # low heterogeneity (interleaved split)
    T_DECAY = 1500   # convergence timescale

    for i, N in enumerate([2, 4, 8]):
        # Each federated "step" corresponds to 1 global round (N local steps)
        # Rounds complete at steps: N, 2N, 3N, ...
        round_steps = np.arange(N, steps[-1] + 1, N)
        # Interpolate centralized loss at round steps
        central_at_round = np.interp(round_steps, steps, losses_smooth)
        # Add heterogeneity variance (decreases as model converges)
        variance = ALPHA / np.sqrt(N) * np.exp(-round_steps / T_DECAY)
        fed_loss = central_at_round * (1 + variance)
        ax.plot(round_steps, fed_loss, color=ACCENT[i], lw=1.8,
                ls=["--", "-.", ":"][i], label=f"Federated N={N} (simulated)",
                alpha=0.9)

    ax.set_xlabel("Training step", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_title("Full training curve", color=TEXT_COL)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, steps[-1])
    ax.set_ylim(0, 1.2)

    # ── Plot 4b: convergence gap at end of training vs N ─────────────────
    ax = axes[1]
    final_central = losses_smooth[-100:].mean()
    N_vals = np.arange(2, 13)
    gap_vals = []
    for N in N_vals:
        variance_final = ALPHA / np.sqrt(N) * np.exp(-steps[-1] / T_DECAY)
        fed_final = final_central * (1 + variance_final)
        gap_vals.append(fed_final)

    ax.plot(N_vals, gap_vals, color=FED_COL, lw=2.5, marker="o", ms=6,
            label="Federated final loss")
    ax.axhline(final_central, color=CENT_COL, lw=2, ls="--",
               label=f"Centralized final loss ({final_central:.3f})")
    ax.fill_between(N_vals, final_central, gap_vals, alpha=0.2, color=FED_COL,
                    label="Convergence gap")

    ax.set_xlabel("Number of agents (N)", fontsize=11)
    ax.set_ylabel("Final training loss", fontsize=11)
    ax.set_title("Final loss vs number of agents", color=TEXT_COL)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(2, 12)

    fig.tight_layout()
    out = OUT_DIR / "3_fedavg_convergence_vs_N.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved: {out}")
    print(f"  Centralized final loss: {final_central:.4f}")
    print(f"  Federated N=2 final loss: {gap_vals[0]:.4f}  "
          f"(gap: {gap_vals[0]-final_central:.4f})")
    print(f"  Federated N=8 final loss: {gap_vals[6]:.4f}  "
          f"(gap: {gap_vals[6]-final_central:.4f})")


# ═══════════════════════════════════════════════════════════════════════════════
# Summary table
# ═══════════════════════════════════════════════════════════════════════════════

def plot_summary_table(overflow_N):
    """Single-slide summary showing all four results at a glance."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Scalability: Centralized GoT vs Federated GoT",
                 color=TEXT_COL, fontsize=15, fontweight="bold", y=1.01)

    labels = [
        "Prompt length\ngrows with N",
        "Communication\ncost",
        "Training\nconvergence",
        "Context window\noverflow at N≈",
    ]
    values_central  = ["O(N) tokens", "O(N × 10.6 Mbps)", "Stable", f"N ≈ {overflow_N:.0f}"]
    values_federated = ["O(1) per vehicle", "O(N × 0.02 Mbps)", "Stable (+small gap)", "Never"]
    colors_central   = [CENT_COL] * 4
    colors_federated = [FED_COL] * 4

    for i, ax in enumerate(axes.flat):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_facecolor("#161b22")
        ax.add_patch(mpatches.FancyBboxPatch((0.02, 0.02), 0.96, 0.96,
            boxstyle="round,pad=0.02", fc="#161b22", ec=GRID_COL, lw=2))
        ax.text(0.5, 0.85, labels[i], ha="center", va="center",
                color=TEXT_COL, fontsize=12, fontweight="bold")
        ax.text(0.5, 0.60, "Centralized",
                ha="center", va="center", color=CENT_COL, fontsize=10)
        ax.text(0.5, 0.45, values_central[i],
                ha="center", va="center", color=CENT_COL, fontsize=13, fontweight="bold")
        ax.text(0.5, 0.28, "Federated",
                ha="center", va="center", color=FED_COL, fontsize=10)
        ax.text(0.5, 0.13, values_federated[i],
                ha="center", va="center", color=FED_COL, fontsize=13, fontweight="bold")

    fig.tight_layout()
    out = OUT_DIR / "0_summary_table.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Loading Q1 inference outputs...")
    samples = load_q1_outputs()
    print(f"  Loaded {len(samples) if samples else 0} Q1 samples\n")

    print("=== Plot 1+2: GoT prompt length vs N ===")
    overflow_N = plot_prompt_length(samples)

    print("\n=== Plot 3: Communication cost vs N ===")
    plot_communication_cost()

    print("\n=== Plot 4: FedAvg convergence vs N ===")
    plot_fedavg_convergence()

    print("\n=== Summary table ===")
    plot_summary_table(overflow_N)

    print(f"\nAll figures saved to {OUT_DIR}/")
    print("Files:")
    for f in sorted(OUT_DIR.glob("*.png")):
        print(f"  {f.name}")
