# Experiment plan

The project studies small character-level language models under controlled training
budgets. The workflow is divided into four measured stages.

1. Validate the environment and reference implementation with tests and a smoke run.
2. Check language-model metrics and random-initialization behavior across tokenizers.
3. Compare recurrent architectures and Transformers using equal training-token budgets.
4. Measure context-length scaling and inspect learned attention patterns.

## Fair-comparison rules

- Tiny Shakespeare is used for the main experiments.
- Architecture comparisons use the same training-token budget.
- Paired configurations use the same two random seeds: 1337 and 42.
- Context-length runs preserve 8,192 training tokens per optimizer step by changing
  batch size inversely with sequence length.
- Final claims should use mean and standard deviation across seeds and should not
  over-interpret differences smaller than the observed seed noise.

## Generated outputs

Raw training runs stay under `outputs/` and are not committed because they contain
checkpoints. Compact tables and figures are copied into `results/` and `figures/`.
