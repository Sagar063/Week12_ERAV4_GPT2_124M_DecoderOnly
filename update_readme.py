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
"""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import datetime as dt

AUTO_BLOCKS = {
    "SUMMARY": ("<!--AUTO:SUMMARY:START-->", "<!--AUTO:SUMMARY:END-->"),
    "LOGS": ("<!--AUTO:LOGS:START-->", "<!--AUTO:LOGS:END-->"),
    "METRICS": ("<!--AUTO:METRICS:START-->", "<!--AUTO:METRICS:END-->"),
    "SAMPLES": ("<!--AUTO:SAMPLES:START-->", "<!--AUTO:SAMPLES:END-->"),
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
        return {
            "best_loss": None,
            "best_step": None,
            "last_loss": None,
            "n_rows": 0,
        }
    df = pd.read_csv(csv_path)
    # Try common column names
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
        # fallback: first numeric column besides step
        num_cols = [c for c in df.columns if df[c].dtype.kind in "if"]
        if step_col in num_cols:
            num_cols.remove(step_col)
        loss_col = num_cols[0] if num_cols else None

    if step_col is None:
        step_col = df.columns[0]

    if loss_col is None:
        return {"best_loss": None, "best_step": None, "last_loss": None, "n_rows": len(df)}

    best_idx = df[loss_col].idxmin()
    best_loss = float(df.loc[best_idx, loss_col])
    best_step = int(df.loc[best_idx, step_col]) if step_col in df.columns else int(best_idx)
    last_loss = float(df.iloc[-1][loss_col])
    return {"best_loss": best_loss, "best_step": best_step, "last_loss": last_loss, "n_rows": int(len(df))}

def replace_block(md: str, start: str, end: str, new_body: str) -> str:
    s = md.find(start)
    e = md.find(end)
    if s == -1 or e == -1 or e < s:
        raise ValueError(f"Could not find markers: {start} ... {end}")
    before = md[: s + len(start)]
    after = md[e:]
    # Ensure exactly one newline after start and before end
    return before + "\n" + new_body.rstrip() + "\n" + after

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True, help="Training run output directory (e.g., out or out_sanity)")
    ap.add_argument("--readme", default="README.md", help="README path to update")
    ap.add_argument("--log_tail", type=int, default=80, help="Number of lines to include from the end of train.log")
    ap.add_argument("--samples_tail_chars", type=int, default=2200, help="Max characters from the end of samples.txt")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    readme_path = Path(args.readme)

    log_path = out_dir / "logs" / "train.log"
    csv_path = out_dir / "logs" / "train_metrics.csv"
    samples_path = out_dir / "samples" / "samples.txt"

    metrics = load_metrics(csv_path)
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")

    # Build blocks
    summary_lines = []
    summary_lines.append(f"- Updated: **{now}**")
    summary_lines.append(f"- Run directory: `{out_dir.as_posix()}`")
    if metrics["best_loss"] is not None:
        summary_lines.append(f"- Best loss: **{metrics['best_loss']:.6f}** at step **{metrics['best_step']}**")
        summary_lines.append(f"- Last logged loss: **{metrics['last_loss']:.6f}**")
        summary_lines.append(f"- Total logged rows: **{metrics['n_rows']}**")
    else:
        summary_lines.append(f"- Metrics: could not parse `{csv_path.as_posix()}`")

    summary_md = "\n".join(summary_lines)

    logs_tail = read_tail_text(log_path, args.log_tail)
    logs_md = "```text\n" + logs_tail + "\n```"


    # Metrics table
    if metrics["best_loss"] is not None:
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
            f"| Total logged rows | {metrics['n_rows']} |\n"
        )

    samples_tail = read_samples_tail(samples_path, args.samples_tail_chars)
    samples_md = "```text\n" + samples_tail + "\n```"


    # Load README and patch blocks
    if not readme_path.exists():
        raise FileNotFoundError(f"README not found: {readme_path.as_posix()}")
    md = readme_path.read_text(encoding="utf-8", errors="replace")

    md = replace_block(md, *AUTO_BLOCKS["SUMMARY"], summary_md)
    md = replace_block(md, *AUTO_BLOCKS["LOGS"], logs_md)
    md = replace_block(md, *AUTO_BLOCKS["METRICS"], metrics_md)
    md = replace_block(md, *AUTO_BLOCKS["SAMPLES"], samples_md)

    readme_path.write_text(md, encoding="utf-8")
    print(f"Updated {readme_path.as_posix()} from {out_dir.as_posix()}")

if __name__ == "__main__":
    main()
