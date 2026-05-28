#!/usr/bin/env python3
"""
V2V-GoT-D trajectory visualizer.

Loads phase4 per_frame.json and lets you browse every frame with a BEV plot
showing all three variants side by side.

Launch:
    python scripts/viz.py
    # then open http://localhost:7860 (or SSH-tunnel from Izar)

SSH tunnel from your laptop:
    ssh -L 7860:localhost:7860 <user>@izar1.izar.cluster
"""

import json
import math
import os
import sys

import gradio as gr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

DEFAULT_DATA = os.path.join(
    os.path.dirname(__file__), '..', '..', 'outputs', 'phase4', 'per_frame.json')
SAFETY_M = 3.0

# ── Colour palette ─────────────────────────────────────────────────────────────
COLORS = {
    'gt_ego':   '#1a1a2e',   # dark navy  — GT ego
    'gt_cav1':  '#e94560',   # red        — GT cav1
    'llm_ego':  '#0f3460',   # blue       — LLM ego
    'llm_cav1': '#e94560',   # red        — LLM cav1
    'res_ego':  '#16213e',   # dark blue  — resolved ego
    'res_cav1': '#533483',   # purple     — resolved cav1
    'conflict': '#ff000033', # translucent red
}


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data(path: str):
    with open(path) as f:
        records = json.load(f)
    # index by scenario → list of records (sorted by ts)
    by_sc: dict[int, list] = {}
    for r in records:
        sc = r['sc']
        by_sc.setdefault(sc, []).append(r)
    for sc in by_sc:
        by_sc[sc].sort(key=lambda x: x['ts'])
    return records, by_sc


# ── BEV plotting ───────────────────────────────────────────────────────────────

def _draw_traj(ax, traj, color, label, linestyle='-', marker='o', zorder=3, alpha=1.0):
    if not traj:
        return
    xs = [0.0] + [p[0] for p in traj]
    zs = [0.0] + [p[1] for p in traj]
    ax.plot(zs, xs, color=color, linestyle=linestyle, linewidth=1.8,
            marker=marker, markersize=4, label=label, zorder=zorder, alpha=alpha)
    # mark start
    ax.scatter([zs[0]], [xs[0]], s=60, color=color, zorder=zorder+1,
               edgecolors='white', linewidths=0.8, alpha=alpha)


def _conflict_zone(ax, traj_a, traj_b, threshold):
    """Shade timesteps where the two trajectories are within threshold."""
    n = min(len(traj_a), len(traj_b))
    for i in range(n):
        ax_i = traj_a[i]; bx_i = traj_b[i]
        d = math.sqrt((ax_i[0]-bx_i[0])**2 + (ax_i[1]-bx_i[1])**2)
        if d < threshold:
            cx = (ax_i[1] + bx_i[1]) / 2   # z → plot x
            cy = (ax_i[0] + bx_i[0]) / 2   # x → plot y
            circle = plt.Circle((cx, cy), threshold/2,
                                 color='red', alpha=0.18, zorder=2)
            ax.add_patch(circle)


def make_bev_figure(record: dict, show_variants: list[str]) -> plt.Figure:
    traj = record.get('traj', {})
    gt_ego   = traj.get('gt_ego',   [])
    gt_cav1  = traj.get('gt_cav1',  [])
    llm_ego  = traj.get('llm_ego',  [])
    llm_cav1 = traj.get('llm_cav1', [])
    res_ego  = traj.get('res_ego',  [])
    res_cav1 = traj.get('res_cav1', [])

    fig, axes = plt.subplots(1, 3, figsize=(15, 6), constrained_layout=True)
    fig.patch.set_facecolor('#f8f9fa')

    titles = [
        '(A) Centralized baseline\nego LLM  vs  cav1 GT',
        '(B) Peer-symmetric + resolver\nboth LLM  →  after resolver',
        '(C) Ablation\nboth LLM, ego always yields',
    ]
    variant_keys = ['A', 'B', 'C']

    for col, (ax, title, key) in enumerate(zip(axes, titles, variant_keys)):
        ax.set_facecolor('#eef2f7')
        ax.set_title(title, fontsize=9, pad=6)
        ax.set_xlabel('lateral z (m)', fontsize=8)
        ax.set_ylabel('forward x (m)', fontsize=8)
        ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
        ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(True, alpha=0.3, linewidth=0.5)

        if key not in show_variants:
            ax.text(0.5, 0.5, 'hidden', transform=ax.transAxes,
                    ha='center', va='center', color='gray', fontsize=14)
            continue

        if key == 'A':
            # ego LLM plan vs cav1 GT path
            _draw_traj(ax, llm_ego,  '#0f3460', 'ego (LLM)',  '-',  'o')
            _draw_traj(ax, gt_cav1,  '#c62a2a', 'cav1 (GT)',  '--', 's')
            _conflict_zone(ax, llm_ego, gt_cav1, SAFETY_M)
            conflict = record.get('conflict_A', False)

        elif key == 'B':
            # both LLM before resolver (faint) + after resolver (solid)
            _draw_traj(ax, llm_ego,  '#0f3460', 'ego LLM (before)',  '--', None, alpha=0.35)
            _draw_traj(ax, llm_cav1, '#c62a2a', 'cav1 LLM (before)', '--', None, alpha=0.35)
            _draw_traj(ax, res_ego,  '#0f3460', 'ego (after resolver)', '-', 'o')
            _draw_traj(ax, res_cav1, '#533483', 'cav1 (after resolver)', '-', 's')
            _conflict_zone(ax, llm_ego, llm_cav1, SAFETY_M)   # show pre-resolver conflict
            conflict = record.get('conflict_B_after', False)

        else:  # C
            ab_ego = traj.get('ab_ego', llm_ego)   # pre-computed in phase4_eval
            _draw_traj(ax, llm_ego,  '#0f3460', 'ego LLM (before)',   '--', None, alpha=0.35)
            _draw_traj(ax, ab_ego,   '#0f3460', 'ego (after ablation)', '-', 'o')
            _draw_traj(ax, llm_cav1, '#c62a2a', 'cav1 LLM', '-', 's')
            _conflict_zone(ax, llm_ego, llm_cav1, SAFETY_M)   # show pre-ablation conflict
            conflict = record.get('conflict_C', False)

        # conflict banner
        banner_color = '#ff4d4d' if conflict else '#2ecc71'
        banner_text  = '⚠ CONFLICT' if conflict else '✓ safe'
        ax.set_title(f'{title}\n{banner_text}',
                     fontsize=9, pad=6,
                     color=banner_color,
                     fontweight='bold' if conflict else 'normal')

        # GT reference (both panels)
        _draw_traj(ax, gt_ego,  '#333333', 'ego GT (ref)',  ':', None, zorder=1, alpha=0.5)
        _draw_traj(ax, gt_cav1, '#888888', 'cav1 GT (ref)', ':', None, zorder=1, alpha=0.5)

        ax.legend(fontsize=7, loc='upper left', framealpha=0.7)

    # shared title
    rule = record.get('resolver_rule', '—')
    yield_cav = record.get('resolver_yield') or '—'
    fig.suptitle(
        f"Scenario {record['sc']}  ·  frame {record['ts']}  "
        f"·  resolver rule: {rule}  ·  yielding: {yield_cav}",
        fontsize=10, y=1.01
    )
    return fig


# ── Metrics table for one frame ────────────────────────────────────────────────

def frame_metrics_html(record: dict) -> str:
    def yn(v): return '⚠ yes' if v else '✓ no'
    rows = [
        ('Variant', 'Conflict?', 'Min dist', 'Notes'),
        ('(A) Centralized', yn(record.get('conflict_A')),
         f"{record.get('min_dist_A') or '—'} m", 'ego LLM vs cav1 GT'),
        ('(B) Before resolver', yn(record.get('conflict_B_before')),
         f"{record.get('min_dist_B_before') or '—'} m", '—'),
        ('(B) After resolver', yn(record.get('conflict_B_after')),
         f"{record.get('min_dist_B_after') or '—'} m",
         record.get('resolver_rule') or '—'),
        ('(C) Ablation', yn(record.get('conflict_C')), '—', 'ego yields by default'),
    ]
    html = '<table style="width:100%;border-collapse:collapse;font-size:13px">'
    for i, row in enumerate(rows):
        bg = '#dde' if i == 0 else ('#f8f8f8' if i % 2 == 0 else 'white')
        html += f'<tr style="background:{bg}">'
        for cell in row:
            tag = 'th' if i == 0 else 'td'
            html += f'<{tag} style="padding:4px 8px;border:1px solid #ccc">{cell}</{tag}>'
        html += '</tr>'
    html += '</table>'
    return html


# ── Gradio app ─────────────────────────────────────────────────────────────────

def build_app(data_path: str):
    records, by_sc = load_data(data_path)
    scenario_ids   = sorted(by_sc.keys())
    sc_choices     = [str(s) for s in scenario_ids]

    def update(sc_str, frame_idx, show_A, show_B, show_C):
        sc    = int(sc_str)
        recs  = by_sc[sc]
        frame_idx = min(int(frame_idx), len(recs) - 1)
        rec   = recs[frame_idx]

        show = []
        if show_A: show.append('A')
        if show_B: show.append('B')
        if show_C: show.append('C')

        fig  = make_bev_figure(rec, show)
        html = frame_metrics_html(rec)

        # update frame slider max
        slider_update = gr.Slider(maximum=len(recs) - 1, value=frame_idx)
        return fig, html, slider_update

    def on_scenario_change(sc_str):
        sc = int(sc_str)
        n  = len(by_sc[sc])
        return gr.Slider(minimum=0, maximum=n - 1, value=0, step=1,
                         label=f'Frame  (0–{n-1})')

    conflict_frames = {sc: [r['ts'] for r in recs if r.get('conflict_B_before')]
                       for sc, recs in by_sc.items()}

    with gr.Blocks(title='V2V-GoT-D Visualizer') as app:
        gr.Markdown('## V2V-GoT-D — trajectory comparison')
        gr.Markdown(
            'Compare centralized baseline **(A)**, peer-symmetric + resolver **(B)**, '
            'and ablation **(C)** across all validation frames.'
        )

        with gr.Row():
            with gr.Column(scale=1):
                sc_dd = gr.Dropdown(
                    choices=sc_choices, value=sc_choices[0],
                    label='Scenario', interactive=True)

                frame_sl = gr.Slider(
                    minimum=0, maximum=len(by_sc[scenario_ids[0]]) - 1,
                    value=0, step=1,
                    label=f'Frame  (0–{len(by_sc[scenario_ids[0]])-1})',
                    interactive=True)

                gr.Markdown('**Conflict frames in this scenario (B, before resolver):**')
                conflict_info = gr.Markdown(
                    ', '.join(str(t) for t in conflict_frames.get(scenario_ids[0], [])) or '—'
                )

                gr.Markdown('**Show variants:**')
                show_A = gr.Checkbox(value=True,  label='(A) Centralized baseline')
                show_B = gr.Checkbox(value=True,  label='(B) Peer-symmetric + resolver')
                show_C = gr.Checkbox(value=False, label='(C) Ablation (no resolver)')

                jump_btn = gr.Button('Jump to next conflict ▶', variant='secondary')

            with gr.Column(scale=3):
                bev_plot = gr.Plot(label='BEV trajectories')
                metrics  = gr.HTML(label='Per-frame metrics')

        # ── Interactions ────────────────────────────────────────────────────────
        inputs  = [sc_dd, frame_sl, show_A, show_B, show_C]
        outputs = [bev_plot, metrics, frame_sl]

        sc_dd.change(fn=on_scenario_change, inputs=[sc_dd], outputs=[frame_sl])
        sc_dd.change(fn=lambda sc: ', '.join(str(t) for t in conflict_frames.get(int(sc), [])) or '—',
                     inputs=[sc_dd], outputs=[conflict_info])
        frame_sl.change(fn=update, inputs=inputs, outputs=outputs)
        show_A.change(fn=update, inputs=inputs, outputs=outputs)
        show_B.change(fn=update, inputs=inputs, outputs=outputs)
        show_C.change(fn=update, inputs=inputs, outputs=outputs)

        # Jump to next conflict frame
        _conflict_ptr = [0]

        def jump_conflict(sc_str, frame_idx, show_A, show_B, show_C):
            sc = int(sc_str)
            cf = conflict_frames.get(sc, [])
            if not cf:
                return update(sc_str, frame_idx, show_A, show_B, show_C)
            recs = by_sc[sc]
            ts_list = [r['ts'] for r in recs]
            current_ts = recs[min(int(frame_idx), len(recs)-1)]['ts']
            # find next conflict timestamp after current
            future = [t for t in cf if t > current_ts]
            next_ts = future[0] if future else cf[0]   # wrap around
            next_idx = ts_list.index(next_ts) if next_ts in ts_list else 0
            return update(sc_str, next_idx, show_A, show_B, show_C)

        jump_btn.click(fn=jump_conflict, inputs=inputs, outputs=outputs)

        # initial render
        app.load(fn=update, inputs=inputs, outputs=outputs)

    return app


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=DEFAULT_DATA,
                        help='Path to phase4 per_frame.json')
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument('--share', action='store_true',
                        help='Create a public Gradio link')
    args = parser.parse_args()

    print(f'Loading {args.data}')
    app = build_app(args.data)
    app.launch(server_port=args.port, share=args.share,
               server_name='0.0.0.0', theme=gr.themes.Soft())
