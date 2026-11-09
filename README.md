# LM Architecture Bench

A reproducible benchmark of small recurrent and Transformer language models on
Tiny Shakespeare. The project focuses on metric interpretation, controlled
architecture comparisons, context-length scaling, and attention-head analysis.

## Scope

The main experiments measure:

- cross-entropy loss, perplexity, bits/token, and bits/byte;
- near-random model loss versus `ln(vocabulary size)`;
- vanilla RNN, GRU, LSTM, larger LSTM, and two Transformer recipes;
- LSTM and modern Transformer behavior at context lengths 64, 128, 256, and 512;
- quantitative attention patterns such as previous-token and current-word-start heads.

Raw checkpoints and datasets are intentionally excluded from Git. Compact result
tables, figures, configs, and analysis scripts are kept in this repository.

## Repository layout

```text
lm-architecture-bench/
├── configs/                 # reproducible model and sweep definitions
├── scripts/                 # setup, execution, and analysis helpers
├── references/              # upstream pin and analysis text inputs
├── results/                 # compact generated tables and environment records
├── figures/                 # generated plots selected for reporting
├── report/                  # final write-up material
├── PROJECT_PLAN.md
├── requirements.txt
└── README.md
```

The external reference implementation is deliberately kept as a sibling dependency:

```text
<workspace>/
├── lm-architecture-bench/          # this repository
└── _deps/
    └── DeepDataMiningLearning/     # reference implementation
```

From the root of this repository, the dependency therefore lives at:

```text
../_deps/DeepDataMiningLearning
```

## Reference implementation

This project uses `nanolm` from
`lkk688/DeepDataMiningLearning`, located inside the upstream checkout at:

```text
../_deps/DeepDataMiningLearning/DeepDataMiningLearning/llm/nanolm
```

The upstream code is not copied into this repository. The first bootstrap resolves
an exact upstream Git SHA into `references/upstream_commit.txt`; after a successful
smoke test, commit that file so subsequent machines use the same source revision.

## First-time setup

Run all commands below from the repository root unless a command explicitly changes
directories.

### 1. Clone this repository

```bash
git clone https://github.com/Parth-Panchal-284/lm-architecture-bench.git
cd lm-architecture-bench
```

### 2. Clone and resolve the reference implementation

```bash
bash scripts/bootstrap_upstream.sh
```

This creates `../_deps/DeepDataMiningLearning` automatically.

### 3. Create or activate a Python environment

Use Python 3.10 or newer and a PyTorch build compatible with the machine. On a GPU
cluster, prefer the site's CUDA/PyTorch module or environment rather than replacing
it blindly with a generic wheel.

Then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

### 4. Validate the environment

```bash
bash scripts/smoke.sh
```

The smoke stage prints Python/PyTorch/GPU information, runs the upstream `nanolm`
tests, and performs a two-step end-to-end training run.

After it succeeds, preserve the resolved reference revision:

```bash
git add references/upstream_commit.txt
git commit -m "Pin nanolm reference revision"
git push
```

You can also record the full software environment:

```bash
bash scripts/capture_environment.sh
```

## Experiment stages

### Stage 1 - metric checks

Run two near-random tokenizer configurations and compare validation loss with
`ln(V)`:

```bash
bash scripts/run_01_metrics.sh
```

The output table is written to:

```text
results/metrics/untrained_tokenizers.csv
```

The trained-model hand calculation is intentionally reused from the modern
Transformer in Stage 2 so no duplicate training run is needed.

### Stage 2 - architecture benchmark

```bash
bash scripts/run_02_architecture.sh
```

This runs six configurations with seeds 1337 and 42. Every run processes
2,000 x 32 x 256 = 16,384,000 training tokens.

Generated compact artifacts include:

```text
results/architecture/summary.csv
results/architecture/summary.md
results/metrics/trained_char_metrics.json
figures/architecture_learning_curves.png
figures/architecture_final_bpb.png
```

The trained metric JSON recomputes:

```text
perplexity     = exp(val_loss)
bits/token     = val_loss / ln(2)
bits/byte      = bits/token * val_tokens / val_bytes
```

and reports the numerical difference from the values logged by `nanolm`.

### Stage 3 - context-length scaling

```bash
bash scripts/run_03_context.sh
```

For every context length, batch size is chosen so that `batch_size x seq_len = 8192`:

| context | batch | tokens/step |
| ---: | ---: | ---: |
| 64 | 128 | 8192 |
| 128 | 64 | 8192 |
| 256 | 32 | 8192 |
| 512 | 16 | 8192 |

This keeps the total training-token budget constant while changing only available
context length.

Key outputs:

```text
results/context/context_summary.csv
figures/context_length_bpb.png
```

### Stage 4 - attention analysis

After Stage 2 has produced the modern Transformer checkpoint:

```bash
bash scripts/run_04_attention.sh
```

The script runs the reference visualization command on dialogue, prose, and structured
text, preserves each heatmap, and computes numerical head measurements.

Key outputs:

```text
figures/attention_dialogue.png
figures/attention_prose.png
figures/attention_structured.png
results/attention/head_metrics.csv
results/attention/summary.md
```

The numerical analysis measures the average attention mass on the immediately
previous character and on the first character of the current word, plus each head's
attention-weighted look-back distance.

## Multi-GPU sweep execution

`nanolm.sweep` can schedule one run per listed GPU. The stage wrappers forward extra
arguments directly to the sweep command. For example, on an allocated four-GPU node:

```bash
bash scripts/run_02_architecture.sh --gpus 0,1,2,3
```

For a single allocated GPU, use the wrapper with no `--gpus` option.

## Resuming interrupted sweeps

The reference sweep runner skips runs that already contain `metrics.json`, so the
same stage command can normally be rerun after an interruption. If a killed process
left a stale `CLAIM` file and no other sweep process is active, use the reference
runner's `--reclaim` flag.

## Reproducibility notes

- The two paired seeds are 1337 and 42.
- Raw runs are stored under `outputs/nanolm/` and ignored by Git.
- Every completed `nanolm` run stores its resolved config, metadata, learning curves,
  final metrics, generated sample, and checkpoint in its run directory.
- The compact files in `results/` and `figures/` are the artifacts intended to be
  versioned and shared.
- Do not commit datasets or checkpoints.

## References

The experiment runner and model implementation come from the public `nanolm` code in
`lkk688/DeepDataMiningLearning`. See `references/README.md` for dependency layout and
revision-pinning details.
