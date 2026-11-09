#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PATTERN = re.compile(r"^(gpt|lstm)_ctx(64|128|256|512)$")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize context-length runs and plot final validation bpb."
    )
    parser.add_argument("root", type=Path)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for metrics_path in sorted(args.root.glob("*/metrics.json")):
        metrics = json.loads(metrics_path.read_text())
        match = PATTERN.match(str(metrics["name"]))
        if not match:
            continue
        rows.append(
            {
                "architecture": match.group(1),
                "context_length": int(match.group(2)),
                "seed": int(metrics["seed"]),
                "val_bpb": float(metrics["val_bpb"]),
                "params": int(metrics["params"]),
                "tokens_seen": int(metrics["tokens_seen"]),
                "train_time_sec": float(metrics["train_time_sec"]),
            }
        )

    if not rows:
        raise SystemExit(f"no context-length runs found under {args.root}")

    raw = pd.DataFrame(rows)
    summary = (
        raw.groupby(["architecture", "context_length"], as_index=False)
        .agg(
            val_bpb_mean=("val_bpb", "mean"),
            val_bpb_std=("val_bpb", "std"),
            n_seeds=("seed", "count"),
            params=("params", "first"),
            tokens_seen=("tokens_seen", "first"),
            train_time_sec_mean=("train_time_sec", "mean"),
        )
        .sort_values(["architecture", "context_length"])
    )

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.csv, index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for architecture, group in summary.groupby("architecture"):
        group = group.sort_values("context_length")
        ax.errorbar(
            group["context_length"],
            group["val_bpb_mean"],
            yerr=group["val_bpb_std"].fillna(0),
            marker="o",
            capsize=3,
            label=architecture.upper(),
        )
    ax.set_xlabel("context length (tokens)")
    ax.set_ylabel("validation bits per byte (lower is better)")
    ax.set_xticks([64, 128, 256, 512])
    ax.set_xscale("log", base=2)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=160)
    plt.close(fig)

    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nwrote {args.csv}")
    print(f"wrote {args.figure}")


if __name__ == "__main__":
    main()
