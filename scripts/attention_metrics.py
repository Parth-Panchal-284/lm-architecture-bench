#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from nanolm.viz import attention_maps, load_run


def current_word_starts(text: str) -> list[tuple[int, int]]:
    """Return (query_position, current_word_start) for non-initial word chars."""
    pairs: list[tuple[int, int]] = []
    start = None
    for index, char in enumerate(text):
        if char.isalnum() or char == "'":
            if start is None:
                start = index
            elif index > start:
                pairs.append((index, start))
        else:
            start = None
    return pairs


def score_text(run_dir: str, label: str, text: str) -> list[dict]:
    _, model, tokenizer, _, _ = load_run(run_dir)
    token_ids = tokenizer.encode(text)
    if len(token_ids) != len(text):
        raise ValueError(
            "attention_metrics.py expects a character tokenizer so text positions "
            "and token positions are identical"
        )

    idx = torch.tensor([token_ids], dtype=torch.long)
    maps = attention_maps(model, idx)
    word_pairs = current_word_starts(text)
    rows = []

    for layer, layer_map in enumerate(maps):
        for head in range(layer_map.shape[0]):
            weights = layer_map[head]
            previous = [weights[t, t - 1] for t in range(1, len(text))]
            word_start = [weights[t, s] for t, s in word_pairs]

            lookbacks = []
            for query in range(len(text)):
                keys = np.arange(query + 1)
                lookbacks.append(
                    float(np.sum((query - keys) * weights[query, : query + 1]))
                )

            rows.append(
                {
                    "text": label,
                    "layer": layer,
                    "head": head,
                    "previous_token_weight": float(np.mean(previous)),
                    "word_start_weight": (
                        float(np.mean(word_start)) if word_start else float("nan")
                    ),
                    "average_lookback": float(np.mean(lookbacks)),
                    "n_characters": len(text),
                    "n_word_start_positions": len(word_pairs),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantify interpretable attention patterns for a character Transformer."
    )
    parser.add_argument("--run", required=True)
    parser.add_argument("--texts", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    texts = json.loads(args.texts.read_text())
    rows = []
    for label, text in texts.items():
        rows.extend(score_text(args.run, label, text))

    df = pd.DataFrame(rows)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.csv, index=False)

    aggregate = (
        df.groupby(["layer", "head"], as_index=False)
        .agg(
            previous_token_weight=("previous_token_weight", "mean"),
            word_start_weight=("word_start_weight", "mean"),
            average_lookback=("average_lookback", "mean"),
        )
    )
    prev = aggregate.nlargest(5, "previous_token_weight")
    word = aggregate.nlargest(5, "word_start_weight")
    far = aggregate.nlargest(5, "average_lookback")

    lines = [
        "# Attention-head measurements",
        "",
        "Weights are averaged across the three analysis texts.",
        "",
        "## Strongest previous-token heads",
        "",
        prev.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Strongest current-word-start heads",
        "",
        word.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Longest average look-back heads",
        "",
        far.to_markdown(index=False, floatfmt=".4f"),
        "",
        "Definitions:",
        "- previous-token weight: attention mass on position t-1, averaged over t > 0",
        "- word-start weight: attention mass on the first character of the current word, excluding the first character itself",
        "- average look-back: attention-weighted distance from each query to its attended keys",
        "",
    ]
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text("\n".join(lines))

    print("Top previous-token heads:")
    print(prev.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nTop word-start heads:")
    print(word.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nwrote {args.csv}")
    print(f"wrote {args.summary}")


if __name__ == "__main__":
    main()
