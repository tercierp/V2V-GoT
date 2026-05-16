"""
V2V-GoT + OSM  —  Interactive Demo
Run:  conda activate llava && python demo/app.py
Access: http://localhost:7860  (forward port via VS Code or ssh -L)
"""

import ast
import io
import json
import os
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE     = Path("/scratch/izar/tercier/v2v-got/V2V-GoT")
SAT_DIR  = Path("/scratch/izar/tercier/v2v-got/sat_images")
EVAL_DIR = BASE / "LLaVA/playground/data/eval"
PIPE_TAG = "osm_from_q1_2500"
SPLIT    = "val"
MDL      = "llava-v1.5-7b"

STAGES = [
    ("Q1", "nq1sm3w0d",  "Object detection — ego view"),
    ("Q2", "nq2sm3w0d",  "Object detection — CAV view"),
    ("Q3", "nq3sm3w0dc", "Scene graph construction"),
    ("Q4", "nq4sm3w0dc", "Scene graph merge"),
    ("Q5", "nq5sm3w1dc", "Trajectory prediction"),
    ("Q6", "nq6sm3w1dc", "Trajectory prediction (CAV)"),
    ("Q7", "nq7sm3w1dc", "Trajectory merge"),
    ("Q8", "nq8sm3w6dc", "Decision making"),
    ("Q9", "nq9sm3w6dc", "🛰 Planning + OSM context"),
]

STAGE_COLORS = [
    "#1565c0","#1976d2","#0288d1","#0097a7",
    "#00897b","#43a047","#7cb342","#f9a825","#e64a19",
]

# ── Sequence list — only the 9 inferred test sequences ───────────────────────
_TEST_SPLITS = {"test_01", "test_02", "test_03"}
_INFERRED_SCENARIOS = 9  # scenario_index 0-8

def _build_seq_list():
    seen, lst = set(), []
    for split in sorted(os.listdir(SAT_DIR)):
        if split not in _TEST_SPLITS:
            continue
        sp = SAT_DIR / split
        if not sp.is_dir():
            continue
        for seq in sorted(os.listdir(sp)):
            if seq not in seen:
                seen.add(seq)
                lst.append((split, seq))
                if len(lst) >= _INFERRED_SCENARIOS:
                    return lst
    return lst

SEQ_LIST = _build_seq_list()

# ── Data helpers ──────────────────────────────────────────────────────────────
def _load_stage(tag):
    p = (EVAL_DIR / f"v2v4real_3d_grounding_{PIPE_TAG}_full_{tag}"
         / "answers" / SPLIT / MDL / "merge.jsonl")
    if not p.exists() or p.stat().st_size == 0:
        return None
    out = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def _find_sample(data, scenario_index, timestamp_index):
    if not data:
        return None
    return next(
        (s for s in data
         if s.get("scenario_index") == scenario_index
         and s.get("local_timestamp_index") == timestamp_index),
        None)

# ── Satellite image ───────────────────────────────────────────────────────────
def _sat_image(scenario_index, timestamp_index):
    if scenario_index >= len(SEQ_LIST):
        return None
    split, seq = SEQ_LIST[scenario_index]
    frame = f"{timestamp_index * 30:06d}"
    path = SAT_DIR / split / seq / frame / "agent_0.png"
    return Image.open(path) if path.exists() else None

# ── BEV rendering (returns PIL image) ────────────────────────────────────────
def _parse_traj(s):
    try:
        return ast.literal_eval(s)
    except Exception:
        return []

def _pose_xy(flat):
    return flat[3], flat[7]

def _bev_image(sample):
    fig, ax = plt.subplots(figsize=(5, 5), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="#666")
    for sp in ax.spines.values():
        sp.set_edgecolor("#333")

    R = 70
    for v in range(-R, R + 1, 20):
        ax.axhline(v, color="#1a1a2a", lw=0.6)
        ax.axvline(v, color="#1a1a2a", lw=0.6)

    traj = _parse_traj(sample.get("future_trajectory_str_in_ego", "[]"))
    if traj:
        xs, ys = zip(*traj)
        ax.plot(xs, ys, "o--", color="#f0a500", lw=1.5, ms=4, label="Trajectory", zorder=3)

    ego = mpatches.FancyBboxPatch((-2, -2.5), 4, 5,
        boxstyle="round,pad=0.3", lw=2, ec="#00e5ff", fc="#00e5ff22", zorder=5)
    ax.add_patch(ego)
    ax.text(0, 0, "EGO", color="#00e5ff", fontsize=8, ha="center",
            va="center", fontweight="bold", zorder=6)

    ep = sample.get("cav_ego_lidar_pose")
    cp = sample.get("cav_1_lidar_pose")
    if ep and cp:
        ex, ey = _pose_xy(ep); cx, cy = _pose_xy(cp)
        dx, dy = cx - ex, cy - ey
        angle = np.arctan2(ep[4], ep[0])
        rx =  dx * np.cos(-angle) - dy * np.sin(-angle)
        ry =  dx * np.sin(-angle) + dy * np.cos(-angle)
        if abs(rx) < R and abs(ry) < R:
            cav = mpatches.FancyBboxPatch((rx-2, ry-2.5), 4, 5,
                boxstyle="round,pad=0.3", lw=2, ec="#7c4dff", fc="#7c4dff22", zorder=5)
            ax.add_patch(cav)
            ax.text(rx, ry, "CAV", color="#7c4dff", fontsize=8,
                    ha="center", va="center", fontweight="bold", zorder=6)

    ax.set_xlim(-R, R); ax.set_ylim(-R, R); ax.set_aspect("equal")
    ax.set_xlabel("lateral (m)", color="#888", fontsize=8)
    ax.set_ylabel("forward (m)", color="#888", fontsize=8)
    ax.legend(loc="upper right", fontsize=7, facecolor="#0d1117",
              labelcolor="#ccc", edgecolor="#333")
    ax.set_title("Bird's Eye View", color="#ccc", fontsize=9)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor="#0d1117")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()

# ── GoT chain HTML ────────────────────────────────────────────────────────────
def _got_html(all_data, scenario_index, timestamp_index):
    html = '<div style="font-family:monospace;padding:8px">'
    for i, (qname, tag, desc) in enumerate(STAGES):
        color = STAGE_COLORS[i]
        data = all_data.get(tag)
        osm_badge = (' <span style="background:#e64a19;color:#fff;padding:1px 6px;'
                     'border-radius:4px;font-size:11px">OSM</span>'
                     if i == 8 else "")

        if data is None:
            body = '<div style="color:#555;font-size:13px">⏳ Inference pending…</div>'
        else:
            s = _find_sample(data, scenario_index, timestamp_index)
            if s is None:
                body = '<div style="color:#555;font-size:13px">No sample for this frame.</div>'
            else:
                convs = s.get("conversations", [])
                q = convs[0]["value"] if convs else ""
                a = s.get("outputs", "—")
                body = (
                    f'<div style="background:#111827;padding:8px;border-radius:4px;'
                    f'margin-top:6px;font-size:12px;color:#9ca3af">'
                    f'<b style="color:#6b7280;font-size:10px">Q</b> {q}</div>'
                    f'<div style="background:#0d200d;border:1px solid {color}55;padding:8px;'
                    f'border-radius:4px;margin-top:4px;font-size:13px;color:#d1fae5">'
                    f'<b style="color:{color};font-size:10px">A</b> {a}</div>'
                )

        html += (
            f'<div style="border-left:3px solid {color};padding:10px 14px;'
            f'margin-bottom:10px;background:#0f1923;border-radius:0 6px 6px 0">'
            f'<b style="color:{color};font-size:14px">{qname}</b>{osm_badge}'
            f'<span style="color:#6b7280;font-size:12px;margin-left:8px">{desc}</span>'
            f'{body}</div>'
        )
    html += '</div>'
    return html

# ── Dropdown helpers ──────────────────────────────────────────────────────────
def _scenario_choices():
    return [f"[{i:02d}] {seq}" for i, (_, seq) in enumerate(SEQ_LIST)]

def _frame_choices(scenario_index):
    if scenario_index >= len(SEQ_LIST):
        return ["0"]
    split, seq = SEQ_LIST[scenario_index]
    d = SAT_DIR / split / seq
    if not d.exists():
        return ["0"]
    frames = sorted(f for f in os.listdir(d) if (d / f).is_dir())
    return [str(int(f) // 30) for f in frames] or ["0"]

def _parse_scenario(choice):
    return int(choice.split("]")[0].strip("["))

# ── Main callback ─────────────────────────────────────────────────────────────
def on_visualise(scenario_choice, frame_str):
    si = _parse_scenario(scenario_choice)
    ti = int(frame_str) if frame_str else 0

    all_data = {tag: _load_stage(tag) for _, tag, _ in STAGES}
    q1 = all_data.get("nq1sm3w0d")
    sample = _find_sample(q1, si, ti) if q1 else None
    if sample is None and q1:
        sample = q1[0]

    sat  = _sat_image(si, ti)
    bev  = _bev_image(sample) if sample else None
    got  = _got_html(all_data, si, ti)
    done = sum(1 for v in all_data.values() if v is not None)
    split, seq = SEQ_LIST[si]
    info = f"**{seq}** · split `{split}` · frame `{ti*30:06d}` · **{done}/9 stages ready**"
    return sat, bev, got, info

def on_scenario_change(scenario_choice):
    si = _parse_scenario(scenario_choice)
    choices = _frame_choices(si)
    return gr.Dropdown(choices=choices, value=choices[0])

# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="V2V-GoT + OSM") as demo:
    gr.Markdown("# 🛰 V2V-GoT + OSM — Graph-of-Thought Cooperative Perception\n"
                "**Model:** LLaVA-v1.5-7B + LoRA OSM Crafter · **Dataset:** V2V4Real")

    with gr.Row():
        scenario_dd = gr.Dropdown(
            choices=_scenario_choices(), value=_scenario_choices()[0],
            label="Scenario", scale=3)
        frame_dd = gr.Dropdown(
            choices=_frame_choices(0), value=_frame_choices(0)[0],
            label="Frame (timestamp)", scale=1)
        run_btn = gr.Button("▶  Visualise", variant="primary", scale=1)

    info_md = gr.Markdown("_Select a scenario and click Visualise._")

    with gr.Row():
        sat_out = gr.Image(label="🛰 Satellite View", type="pil", height=380)
        bev_out = gr.Image(label="🚗 Bird's Eye View", type="pil", height=380)

    got_out = gr.HTML(label="Graph-of-Thought Chain  (Q1 → Q9)")

    scenario_dd.change(on_scenario_change, scenario_dd, frame_dd)
    run_btn.click(on_visualise, [scenario_dd, frame_dd],
                  [sat_out, bev_out, got_out, info_md])

    gr.Markdown("---\n**Q9** uses satellite context (OSM). "
                "Stages marked ⏳ are still running in the background SLURM job.")

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
