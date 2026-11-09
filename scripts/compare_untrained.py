#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd
import yaml


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare near-untrained validation loss with ln(vocabulary size)."
    )
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for metrics_path in sorted(args.root.glob("*/metrics.json")):
        run_dir = metrics_path.parent
        metrics = json.loads(metrics_path.read_text())
        config = yaml.safe_load((run_dir / "config.yaml").read_text())
        vocab_size = int(config["model"]["vocab_size"])
        expected = math.log(vocab_size)
        observed = float(metrics["val_loss"])
        rows.append(
            {
                "name": metrics["name"],
                "tokenizer": metrics["tokenizer"],
                "vocab_size": vocab_size,
                "ln_vocab": expected,
                "val_loss": observed,
                "loss_minus_ln_vocab": observed - expected,
                "val_bpb": float(metrics["val_bpb"]),
                "run_dir": str(run_dir),
            }
        )

    if not rows:
        raise SystemExit(f"no completed runs found under {args.root}")

    df = pd.DataFrame(rows).sort_values("name")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
