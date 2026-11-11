"""Success curves for one or more runs: evaluation return, failed clips, and training return vs training time.

The time axis is training time with the evaluation overhead deducted: checkpoint evaluations share the
GPU with training and slow it down, so every logged interval that overlaps an evaluation is counted at
the run's median clean interval duration (an estimate; the raw wall-clock is in train_log.csv).

Usage: python tools/plot.py logs/run_a [logs/run_b ...] [--labels A B] --out results/success_curve.png
"""

import argparse
import csv
import glob
import os
import re
import statistics
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
REF = "#8a8984"


def read(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _ts(s):
    return time.mktime(time.strptime(s, "%Y-%m-%d %H:%M:%S"))


def load_run(run_dir):
    """Rows of the train and eval logs, plus a map from raw wall hours to eval-deducted hours."""
    tr = read(os.path.join(run_dir, "train_log.csv"))
    ev = sorted(read(os.path.join(run_dir, "eval.csv")), key=lambda r: int(r["iteration"]))
    # evaluation windows in the training clock (hours since training start)
    dur = {}
    log = os.path.join(run_dir, "eval_stdout.log")
    if os.path.exists(log):
        for m in re.finditer(r"\[eval\] it (\d+) .*\((\d+)s\)", open(log).read()):
            dur[int(m.group(1))] = int(m.group(2))
    windows = []
    for r in ev:
        t0 = _ts(r["saved_at"]) - float(r["train_wall_h"]) * 3600  # absolute time of training start
        end = (_ts(r["eval_at"]) - t0) / 3600
        windows.append((end - dur.get(int(r["iteration"]), 200) / 3600, end))
    wall = [float(r["wall_h"]) for r in tr]
    dts = [wall[0]] + [b - a for a, b in zip(wall, wall[1:])]
    hit = [any(a < w1 and b > w0 for w0, w1 in windows) for a, b in zip([0.0] + wall[:-1], wall)]
    clean = [d for d, h, r in zip(dts, hit, tr) if not h and float(r["pi_loss"]) != 0.0]
    med = statistics.median(clean) if clean else 0.0
    adj, acc = [], 0.0
    for d, h in zip(dts, hit):
        acc += min(d, med) if h and med > 0 else d
        adj.append(acc)
    its = [int(r["iteration"]) for r in tr]

    def hours_at(iteration):
        k = min(range(len(its)), key=lambda i: abs(its[i] - iteration))
        return adj[k]

    return {"tr": tr, "ev": ev, "adj": adj, "hours_at": hours_at, "deducted_h": (wall[-1] - adj[-1]) if wall else 0.0}


def style(ax, title, ylabel):
    ax.set_facecolor(SURF)
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.set_xlabel("training time (h), 1 GPU, evaluation overhead deducted", color=INK2, fontsize=9)
    ax.grid(True, color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9c8c2")
    ax.tick_params(colors=INK2, labelsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--target", type=float, default=0.82)
    a = ap.parse_args()
    labels = a.labels or [os.path.basename(os.path.normpath(d)) for d in a.run_dirs]
    runs = [load_run(d) for d in a.run_dirs]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), facecolor=SURF)
    style(axes[0], "(a) Evaluation normalized return, 40 LAFAN1 clips", "normalized return")
    style(axes[1], "(b) Failed clips in evaluation (of 40)", "clips terminated early")
    style(axes[2], "(c) Training-episode normalized return", "mean normalized return per episode")
    notes = []
    for run, label, color in zip(runs, labels, SERIES):
        ev, tr = run["ev"], run["tr"]
        if ev:
            x = [run["hours_at"](int(r["iteration"])) for r in ev]
            y = [float(r["norm_return"]) for r in ev]
            nf = [int(r["n_failed"]) for r in ev]
            axes[0].plot(x, y, color=color, lw=2, marker="o", ms=4.5, mec=SURF, mew=1.2, zorder=3, label=label)
            axes[0].annotate(f"{y[-1]:.3f}", (x[-1], y[-1]), xytext=(5, 5), textcoords="offset points", fontsize=8, color=INK)
            axes[1].plot(x, nf, color=color, lw=2, marker="o", ms=4.5, mec=SURF, mew=1.2, zorder=3, label=label)
            axes[1].annotate(f"{nf[-1]}", (x[-1], nf[-1]), xytext=(5, 5), textcoords="offset points", fontsize=8, color=INK)
            hit = next((i for i, v in enumerate(y) if v >= a.target), None)
            zero = next((i for i, v in enumerate(nf) if v == 0), None)
            note = f"{label}: "
            note += f"return {a.target} at {x[hit]:.2f} h (it {ev[hit]['iteration']})" if hit is not None else f"return {a.target} not reached"
            note += f"; 0/40 failed at {x[zero]:.2f} h (it {ev[zero]['iteration']})" if zero is not None else "; 0/40 failed not reached"
            note += f"; {run['deducted_h'] * 60:.0f} min of evaluation overhead deducted"
            notes.append(note)
        if tr:
            y = [float(r["ep_ret_norm"]) for r in tr]
            k = 10
            ys = [sum(y[max(0, i - k) : i + 1]) / len(y[max(0, i - k) : i + 1]) for i in range(len(y))]
            axes[2].plot(run["adj"], ys, color=color, lw=2, zorder=3, label=label)
    axes[0].axhline(a.target, color=REF, lw=1, ls="--", zorder=1)
    axes[0].text(0.01, a.target + 0.012, f"target {a.target} (paper Fig. 4a: reached at 1.2 h on an H100)",
                 transform=axes[0].get_yaxis_transform(), fontsize=8, color=INK2)  # fmt: skip
    axes[0].set_ylim(0, 1)
    axes[1].set_ylim(-0.5, 40.5)
    axes[2].set_ylim(0, 1)
    if len(runs) > 1:
        for ax in axes:
            ax.legend(loc="lower right" if ax is not axes[1] else "upper right", fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.05 + 0.035 * len(notes), 1, 1))
    foot = "Time axis: training time only. Checkpoint evaluations share the GPU with training; intervals overlapping an evaluation are counted at the run's median clean speed (estimate)."
    fig.text(0.008, 0.012, "\n".join(notes + [foot]), fontsize=8, color=INK2, va="bottom", ha="left")
    out = a.out or os.path.join(a.run_dirs[0], "success_curve.png")
    fig.savefig(out, dpi=140, facecolor=SURF)
    print("wrote", out)
    for n in notes:
        print(n)


if __name__ == "__main__":
    main()
