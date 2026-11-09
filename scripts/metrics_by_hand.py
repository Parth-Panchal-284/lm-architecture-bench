#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute LM metrics from one nanolm metrics.json file."
    )
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    metrics = json.loads(args.metrics.read_text())
    loss = float(metrics["val_loss"])
    tokens = int(metrics["val_tokens"])
    nbytes = int(metrics["val_bytes"])

    ppl = math.exp(loss)
    bits_per_token = loss / math.log(2)
    bpb = bits_per_token * tokens / nbytes

    result = {
        "source": str(args.metrics),
        "val_loss": loss,
        "val_tokens": tokens,
        "val_bytes": nbytes,
        "recomputed_val_ppl": ppl,
        "logged_val_ppl": float(metrics["val_ppl"]),
        "ppl_abs_error": abs(ppl - float(metrics["val_ppl"])),
        "bits_per_token": bits_per_token,
        "recomputed_val_bpb": bpb,
        "logged_val_bpb": float(metrics["val_bpb"]),
        "bpb_abs_error": abs(bpb - float(metrics["val_bpb"])),
    }

    print(json.dumps(result, indent=2))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
