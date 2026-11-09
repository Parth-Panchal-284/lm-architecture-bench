from pathlib import Path
import json
import sys

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

UPSTREAM = (
    ROOT.parent
    / "_deps"
    / "DeepDataMiningLearning"
    / "DeepDataMiningLearning"
    / "llm"
)

sys.path.insert(0, str(UPSTREAM))

from nanolm.generate import load_run
from nanolm.evaluate import generate


RUN = (
    ROOT
    / "outputs"
    / "nanolm"
    / "sweeps"
    / "architecture"
    / "gpt_modern_s1337"
)

OUT_DIR = ROOT / "results" / "diversity"
FIG_DIR = ROOT / "figures"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

TEMPERATURES = [0.5, 0.8, 1.0, 1.2]
TOP_PS = [0.8, 0.9, 0.95, 1.0]

SEEDS = [0, 1, 2, 3, 4]

PROMPT = "ROMEO:"
TOKENS = 300


def distinct_4gram_fraction(text):
    if len(text) < 4:
        return 0.0

    grams = [
        text[i:i + 4]
        for i in range(len(text) - 3)
    ]

    return len(set(grams)) / len(grams)


metrics = json.loads(
    (RUN / "metrics.json").read_text()
)

fixed_bpb = metrics["val_bpb"]

cfg, model, tok, dinfo, _ = load_run(
    str(RUN)
)

ctx = cfg.eval.eval_seq_len or cfg.train.seq_len

rows = []

for temp in TEMPERATURES:
    for top_p in TOP_PS:
        for seed in SEEDS:

            text = generate(
                model=model,
                tok=tok,
                dinfo=dinfo,
                prompt=PROMPT,
                max_new=TOKENS,
                ctx_len=ctx,
                temperature=temp,
                top_k=0,
                seed=seed,
                top_p=top_p,
            )

            if text.startswith(PROMPT):
                generated = text[len(PROMPT):]
            else:
                generated = text

            diversity = distinct_4gram_fraction(
                generated
            )

            rows.append({
                "temperature": temp,
                "top_p": top_p,
                "seed": seed,
                "distinct_4gram_fraction": diversity,
                "model_val_bpb": fixed_bpb,
                "generated_chars": len(generated),
            })

            print(
                f"temp={temp:.2f} "
                f"top_p={top_p:.2f} "
                f"seed={seed} "
                f"diversity={diversity:.4f}"
            )


df = pd.DataFrame(rows)

raw_path = OUT_DIR / "diversity_raw.csv"
df.to_csv(raw_path, index=False)

summary = (
    df.groupby(
        ["temperature", "top_p"],
        as_index=False,
    )
    .agg(
        diversity_mean=(
            "distinct_4gram_fraction",
            "mean",
        ),
        diversity_std=(
            "distinct_4gram_fraction",
            "std",
        ),
        model_val_bpb=(
            "model_val_bpb",
            "first",
        ),
    )
)

summary_path = OUT_DIR / "diversity_summary.csv"
summary.to_csv(summary_path, index=False)

fig, ax = plt.subplots(figsize=(7, 5))

for top_p in TOP_PS:
    sub = summary[
        summary["top_p"] == top_p
    ].sort_values("temperature")

    ax.plot(
        sub["temperature"],
        sub["diversity_mean"],
        marker="o",
        label=f"top-p={top_p}",
    )

ax.set_xlabel("Temperature")
ax.set_ylabel(
    "Distinct 4-gram fraction"
)
ax.set_title(
    "Sample diversity vs decoding settings"
)
ax.legend()
ax.grid(alpha=0.25)

fig.tight_layout()

fig_path = (
    FIG_DIR
    / "diversity_vs_sampling.png"
)

fig.savefig(
    fig_path,
    dpi=200,
)

print()
print("Fixed model validation bpb:",
      fixed_bpb)
print()
print(summary.to_string(index=False))
print()
print("wrote", raw_path)
print("wrote", summary_path)
print("wrote", fig_path)
