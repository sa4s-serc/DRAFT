# Human Evaluation

Blinded human evaluation of four approaches (**Prompting**, **RAFG**, **Finetuning**, **DRAFT**) on two tasks: Context → Decision (**CD**) and Title → Body (**TB**).

## Study Design

| Study | Samples | Raters |
|---|---|---|
| Author | 64 test ADRs (90% confidence, 10% margin, N = 870) | 2 authors |
| Expert | 14-sample subset of the 64 (90% confidence, 20% margin, N = 64) | 1 senior architect (all 14); 2 PhD students in software architecture (samples 1–7 and 8–14), merged into one *junior architect* rater |

Each output received two 5-point ratings: **Closeness to Ground Truth** and **Architectural Correctness**. Outputs were blinded and randomly shuffled; raters saw the input and the ground truth.

Each approach is represented by its best model by BERTScore F1 on the test set:

| Approach | CD | TB |
|---|---|---|
| Prompting | Gemini-2.5-Flash | Gemini-2.5-Flash |
| RAFG | Qwen3-30B | Gemma-3-4B |
| Finetuning | Qwen3-30B | Qwen3-30B |
| DRAFT | Qwen3-30B | Qwen3-30B |

Evaluation apps: [CD](https://script.google.com/macros/s/AKfycbzBiiQRMRX6CJUglgm-UHKYN3wMU_Mq1kAuqL18WaFD5S2aBMZ9KhwKI_EnnVh7VvFj/exec) · [TB](https://script.google.com/macros/s/AKfycbzwUH1Did2TQ9SnjgiM1CFMab62Rx4llsnOHz5njWFnvEBdeQRmjnMJVu7IJoIo8Q/exec)

## Files

| File | Content |
|---|---|
| `HumanEval.ipynb` | Sampling (seed 42) and construction of the evaluation datasets |
| `sampled_keys.json` | Sampled primary keys (64 author, 14 expert) |
| `{cd,tb}_64_samples.jsonl`, `{cd,tb}_14_samples*.jsonl` | Evaluation datasets |
| `{CD,TB} Authors.xlsx` | Author ratings (one sheet per author) |
| `{CD,TB} Expert.xlsx` | Senior architect ratings |
| `{CD,TB} Expert 1to7.xlsx`, `{CD,TB} Expert 8to14.xlsx` | Junior architect ratings |
| `analysis.py` | Statistical analysis |
| `{CD,TB}_{author,expert,combined}_results.txt` | Analysis reports |

## Analysis

```bash
python analysis.py
```

Computes, per study and task: mean scores; quadratic weighted Cohen's κ; Friedman test over the four approaches (per-sample scores averaged over raters); Holm-corrected Wilcoxon signed-rank tests of DRAFT vs. each baseline when Friedman p < 0.05. The combined report covers authors on the 14 expert samples, all four raters on the 14 samples, and all ratings pooled over the 64 samples.

## Results

Mean scores (Closeness / Correctness, 1–5). Bold: highest per group and column.

| Raters | Approach | CD | TB |
|---|---|---|---|
| Authors (n = 64) | Prompting | 2.89 / 3.38 | 2.41 / 2.98 |
| | RAFG | 2.99 / 3.55 | 2.44 / 3.09 |
| | Finetuning | 2.95 / 3.48 | 2.63 / 3.09 |
| | DRAFT | **3.20** / **3.68** | **2.72** / **3.25** |
| Experts (n = 14) | Prompting | 2.96 / 3.46 | **2.46** / 2.50 |
| | RAFG | **3.39** / **3.82** | **2.46** / **3.57** |
| | Finetuning | 2.75 / 2.93 | 2.11 / 2.32 |
| | DRAFT | 2.68 / 2.82 | 2.29 / 2.32 |
| All raters (n = 14) | Prompting | 2.89 / 3.36 | 2.45 / 2.89 |
| | RAFG | **3.29** / **3.64** | 2.55 / **3.39** |
| | Finetuning | 2.89 / 3.23 | 2.32 / 2.82 |
| | DRAFT | 2.96 / 3.30 | **2.57** / 2.84 |

Agreement and significance (Closeness / Correctness):

| Raters | Task | κ | Friedman p | Significant pairwise |
|---|---|---|---|---|
| Authors | CD | 0.352 / 0.205 | 0.049 / 0.028 | DRAFT > Prompting, Correctness (p = 0.024) |
| | TB | 0.258 / 0.259 | 0.065 / 0.165 | — |
| Experts | CD | 0.569 / 0.472 | 0.169 / 0.031 | none |
| | TB | 0.683 / 0.407 | 0.239 / 0.002 | RAFG > DRAFT, Correctness (p = 0.011) |
| All raters | CD | 0.368 / 0.267 (mean pairwise) | 0.196 / 0.151 | — |
| | TB | 0.374 / 0.233 (mean pairwise) | 0.601 / 0.421 | — |

The expert differences stem from the senior architect (Friedman p = 0.010 and < 0.001 for Correctness on CD and TB); junior architect ratings show no significant differences (all p > 0.25).
