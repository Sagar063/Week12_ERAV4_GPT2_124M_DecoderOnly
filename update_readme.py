#!/usr/bin/env python3
"""update_readme.py

Fills README.md placeholders from a training run directory.

Expected files under --out_dir:
  logs/train.log
  logs/train_metrics.csv
  samples/samples.txt

Usage:
  python update_readme.py --out_dir out
  python update_readme.py --out_dir out_sanity --readme README.md

Notes:
- This script ONLY edits content between these markers in README.md:
  <!--AUTO:SUMMARY:START--> ... <!--AUTO:SUMMARY:END-->
  <!--AUTO:LOGS:START-->    ... <!--AUTO:LOGS:END-->
  <!--AUTO:METRICS:START--> ... <!--AUTO:METRICS:END-->
  <!--AUTO:SAMPLES:START--> ... <!--AUTO:SAMPLES:END-->
  <!--AUTO:PLOT:START-->    ... <!--AUTO:PLOT:END-->
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

AUTO_BLOCKS = {
    "SUMMARY": ("<!--AUTO:SUMMARY:START-->", "<!--AUTO:SUMMARY:END-->"),
    "LOGS": ("<!--AUTO:LOGS:START-->", "<!--AUTO:LOGS:END-->"),
    "METRICS": ("<!--AUTO:METRICS:START-->", "<!--AUTO:METRICS:END-->"),
    "SAMPLES": ("<!--AUTO:SAMPLES:START-->", "<!--AUTO:SAMPLES:END-->"),
    "PLOT": ("<!--AUTO:PLOT:START-->", "<!--AUTO:PLOT:END-->"),
}


def read_tail_text(p: Path, n_lines: int) -> str:
    if not p.exists():
        return f"[missing] {p.as_posix()}"
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    tail = lines[-n_lines:] if len(lines) > n_lines else lines
    return "\n".join(tail)


def read_samples_tail(p: Path, max_chars: int) -> str:
    if not p.exists():
        return f"[missing] {p.as_posix()}"
    txt = p.read_text(encoding="utf-8", errors="replace")
    if len(txt) <= max_chars:
        return txt.strip()
    return txt[-max_chars:].strip()


def load_metrics(csv_path: Path) -> dict:
    if not csv_path.exists():
        return {"best_loss": None, "best_step": None, "last_loss": None, "n_rows": 0}

    df = pd.read_csv(csv_path)

    step_col = None
    for c in ["step", "global_step", "iter", "iteration"]:
        if c in df.columns:
            step_col = c
            break

    loss_col = None
    for c in ["loss", "train_loss", "total_loss"]:
        if c in df.columns:
            loss_col = c
            break

    if loss_col is None:
        num_cols = [c for c in df.columns if df[c].dtype.kind in "if"]
        if step_col in num_cols:
            num_cols.remove(step_col)
        loss_col = num_cols[0] if num_cols else None

    if step_col is None:
        step_col = df.columns[0]

    if loss_col is None:
        return {"best_loss": None, "best_step": None, "last_loss": None, "n_rows": int(len(df))}

    best_idx = df[loss_col].idxmin()
    best_loss = float(df.loc[best_idx, loss_col])
    best_step = int(df.loc[best_idx, step_col]) if step_col in df.columns else int(best_idx)
    last_loss = float(df.iloc[-1][loss_col])

    return {
        "best_loss": best_loss,
        "best_step": best_step,
        "last_loss": last_loss,
        "n_rows": int(len(df)),
        "step_col": step_col,
        "loss_col": loss_col,
    }


def save_loss_plot(csv_path: Path, out_png: Path, title: str, max_points: int = 5000) -> bool:
    """Generate a simple loss curve plot from train_metrics.csv.

    Returns True if plot is generated, False otherwise.
    """
    if not csv_path.exists():
        return False

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return False

    # Identify step and loss columns
    step_col = None
    for c in ["step", "global_step", "iter", "iteration"]:
        if c in df.columns:
            step_col = c
            break

    loss_col = None
    for c in ["loss", "train_loss", "total_loss"]:
        if c in df.columns:
            loss_col = c
            break

    if loss_col is None:
        num_cols = [c for c in df.columns if df[c].dtype.kind in "if"]
        if step_col in num_cols:
            num_cols.remove(step_col)
        loss_col = num_cols[0] if num_cols else None

    if step_col is None:
        step_col = df.columns[0]

    if loss_col is None:
        return False

    if len(df) > max_points:
        stride = max(1, len(df) // max_points)
        df = df.iloc[::stride].copy()

    x = df[step_col].astype(float).to_numpy()
    y = df[loss_col].astype(float).to_numpy()

    try:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.figure()
        plt.plot(x, y)
        plt.xlabel(step_col)
        plt.ylabel(loss_col)
        plt.title(title)
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.tight_layout()
        plt.savefig(out_png)
        plt.close()
        return True
    except Exception:
        try:
            plt.close()
        except Exception:
            pass
        return False


def replace_block(md: str, start: str, end: str, new_body: str) -> str:
    s = md.find(start)
    e = md.find(end)
    if s == -1 or e == -1 or e < s:
        raise ValueError(f"Could not find markers: {start} ... {end}")
    before = md[: s + len(start)]
    after = md[e:]
    return before + "\n" + new_body.rstrip() + "\n" + after


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True, help="Training run output directory (e.g., out\\runs\\<run_name>)")
    ap.add_argument("--readme", default="README.md", help="README path to update")
    ap.add_argument("--log_tail", type=int, default=80, help="Lines from end of logs/train.log")
    ap.add_argument("--samples_tail_chars", type=int, default=2200, help="Characters from end of samples/samples.txt")
    ap.add_argument("--plot_max_points", type=int, default=5000, help="Max points to plot (downsample if larger)")
    ap.add_argument("--plot_name", default="loss_curve.png", help="Loss plot filename saved under logs/")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    readme_path = Path(args.readme)

    log_path = out_dir / "logs" / "train.log"
    csv_path = out_dir / "logs" / "train_metrics.csv"
    samples_path = out_dir / "samples" / "samples.txt"
    plot_path = out_dir / "logs" / args.plot_name

    metrics = load_metrics(csv_path)
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")

    # SUMMARY block
    summary_lines = [
        f"- Updated: **{now}**",
        f"- Run directory: `{out_dir.as_posix()}`",
    ]
    if metrics.get("best_loss") is not None:
        summary_lines.append(f"- Best loss: **{metrics['best_loss']:.6f}** at step **{metrics['best_step']}**")
        summary_lines.append(f"- Last logged loss: **{metrics['last_loss']:.6f}**")
        summary_lines.append(f"- Total logged rows: **{metrics['n_rows']}**")
    else:
        summary_lines.append(f"- Metrics: could not parse `{csv_path.as_posix()}`")

    summary_md = "\n".join(summary_lines)

    # LOGS block
    logs_tail = read_tail_text(log_path, args.log_tail)
    logs_md = "```text\n" + logs_tail + "\n```"

    # METRICS block
    if metrics.get("best_loss") is not None:
        metrics_md = (
            "| Metric | Value |\n"
            "|---|---:|\n"
            f"| Best loss | {metrics['best_loss']:.6f} |\n"
            f"| Step at best loss | {metrics['best_step']} |\n"
            f"| Last logged loss | {metrics['last_loss']:.6f} |\n"
            f"| Total logged rows | {metrics['n_rows']} |\n"
        )
    else:
        metrics_md = (
            "| Metric | Value |\n"
            "|---|---|\n"
            "| Best loss | (unavailable) |\n"
            "| Step at best loss | (unavailable) |\n"
            "| Last logged loss | (unavailable) |\n"
            f"| Total logged rows | {metrics.get('n_rows', 0)} |\n"
        )

    # SAMPLES block
    samples_tail = read_samples_tail(samples_path, args.samples_tail_chars)
    samples_md = "```text\n" + samples_tail + "\n```"

    # PLOT block (IMPORTANT: inside main so csv_path/plot_path exist)
    plot_ok = save_loss_plot(
        csv_path=csv_path,
        out_png=plot_path,
        title=f"Training loss vs step ({out_dir.name})",
        max_points=args.plot_max_points,
    )
    if plot_ok:
        plot_rel = plot_path.as_posix()
        plot_md = f"![Training loss curve]({plot_rel})"
    else:
        plot_md = f"_Loss plot unavailable (missing or unparseable `{csv_path.as_posix()}`)._"

    # Patch README
    if not readme_path.exists():
        raise FileNotFoundError(f"README not found: {readme_path.as_posix()}")
    md = readme_path.read_text(encoding="utf-8", errors="replace")

    md = replace_block(md, *AUTO_BLOCKS["SUMMARY"], summary_md)
    md = replace_block(md, *AUTO_BLOCKS["LOGS"], logs_md)
    md = replace_block(md, *AUTO_BLOCKS["METRICS"], metrics_md)
    md = replace_block(md, *AUTO_BLOCKS["SAMPLES"], samples_md)
    md = replace_block(md, *AUTO_BLOCKS["PLOT"], plot_md)

    readme_path.write_text(md, encoding="utf-8")
    print(f"Updated {readme_path.as_posix()} from {out_dir.as_posix()}")


if __name__ == "__main__":
    main()
