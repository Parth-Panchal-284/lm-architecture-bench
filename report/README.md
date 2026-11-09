# Report working outline

Keep the main report concise and evidence-driven. Every numerical claim should trace
back to a saved run or a derived table in `results/`.

## 1. Question and hypotheses

- Expected ordering of recurrent and Transformer models.
- Expected effect of model size on the larger LSTM.
- Expected effect of increasing context length for LSTM and Transformer.

## 2. Experimental setup

Record:

- dataset and tokenizer;
- model configurations and parameter counts;
- equal training-token budget;
- seeds;
- hardware and software environment;
- upstream and project Git revisions.

## 3. Metrics sanity checks

Include the hand calculation from `results/metrics/trained_char_metrics.json` and the
near-random tokenizer comparison from `results/metrics/untrained_tokenizers.csv`.

## 4. Architecture results

Use:

- `figures/architecture_learning_curves.png`;
- `figures/architecture_final_bpb.png`;
- `results/architecture/summary.csv`.

Discuss the vanilla RNN gap and whether the larger LSTM improves or overfits.

## 5. Context-length results

Use `figures/context_length_bpb.png` and `results/context/context_summary.csv`.
Identify the point at which each architecture stops receiving a meaningful benefit
from additional context, with seed variation taken into account.

## 6. Attention analysis

Use the three attention figures and `results/attention/summary.md`. Report at least
two interpretable head behaviors with measured average attention weights.

## 7. Conclusion and limitations

Summarize what the controlled runs support, what remains unresolved by two seeds,
and any hardware/runtime constraints that affected the study.
