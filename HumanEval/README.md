# Human Evaluation Framework

This directory contains the human evaluation study for assessing the quality of ADR generation across four approaches: **Prompting**, **RAFG**, **Finetuning**, and **DRAFT**. Two domain experts (authors) independently rated generated outputs on two tasks using custom Google Web Apps.

---

## Study Design

The human evaluation consists of two hierarchical studies:

- **Author Study**: 64 sampled ADR instances evaluated by domain experts
- **Architect Study**: 14 instances (subset of Author Study) evaluated by architects for deeper analysis

Each study covers two generation tasks:
- **Context-to-Decision (CD)**: Evaluating decision generation quality given a context
- **Title-to-Body (TB)**: Evaluating full ADR body generation given only a title

For each generation, evaluators assigned two 5-point Likert ratings:
- **Closeness to Ground Truth**: How similar the generation is to the reference solution
- **Architectural Correctness**: How well the output follows sound software architecture principles

Outputs from all four approaches were **blinded and randomly shuffled** to prevent evaluator bias.

---

## Model Selection

Each approach was represented by its **top-performing model** as determined by automated BertScore evaluation on the test set:

| Approach | Model | Task Type |
|----------|-------|-----------|
| Prompting | Gemini 2.5 Flash | Both CD & TB |
| Finetuning | Qwen3 30B A3B Instruct | Both CD & TB |
| RAFG | Model varies by task | CD: Qwen3; TB: Gemma 3 4B |
| DRAFT | Qwen3 30B A3B Instruct | Both CD & TB |

---

## Files

| File | Description |
|------|-------------|
| `HumanEval.ipynb` | Dataset construction notebook (sampling, building JSONL files) |
| `analysis.py` | Statistical analysis script (means, IRA, significance tests) |
| `CD Authors.xlsx` | Raw evaluator scores for the CD task (2 sheets, one per evaluator) |
| `TB Authors.xlsx` | Raw evaluator scores for the TB task (2 sheets, one per evaluator) |
| `CD Expert.xlsx` / `TB Expert.xlsx` | Senior architect scores for all 14 architect samples |
| `CD Expert 1to7.xlsx` / `TB Expert 1to7.xlsx` | Student architect 1 scores for architect samples 1–7 |
| `CD Expert 8to14.xlsx` / `TB Expert 8to14.xlsx` | Student architect 2 scores for architect samples 8–14 |
| `CD_author_results.txt` | Generated analysis report for the CD task (author study) |
| `TB_author_results.txt` | Generated analysis report for the TB task (author study) |
| `CD_expert_results.txt` | Generated analysis report for the CD task (expert study) |
| `TB_expert_results.txt` | Generated analysis report for the TB task (expert study) |
| `CD_combined_results.txt` | Authors + experts combined report for the CD task |
| `TB_combined_results.txt` | Authors + experts combined report for the TB task |
| `sampled_keys.json` | Sampled primary keys (author: 64, architect: 14) |
| `cd_64_samples.jsonl` | CD evaluation dataset for author study (64 samples) |
| `tb_64_samples.jsonl` | TB evaluation dataset for author study (64 samples) |
| `cd_14_samples.jsonl` | CD evaluation dataset for architect study (14 samples) |
| `tb_14_samples.jsonl` | TB evaluation dataset for architect study (14 samples) |
| `cd_14_samples_1to7.jsonl` / `cd_14_samples_8to14.jsonl` | CD architect samples split into halves for the two student architects |
| `tb_14_samples_1to7.jsonl` / `tb_14_samples_8to14.jsonl` | TB architect samples split into halves for the two student architects |

---

## Workflow

### 1. Dataset Preparation (`HumanEval.ipynb`)

The notebook automates the entire evaluation dataset construction:

- **Sample 64 keys** from the test set (reproducible with seed 42)
- **Build CD dataset**: Aggregate context, ground truth decisions, and generated decisions from all four approaches
- **Build TB dataset**: Aggregate titles, ground truth bodies, and generated bodies from all four approaches
- **Sample 14 architect keys**: Randomly select a subset from the 64 author samples
- **Create 14-sample datasets**: Filter CD and TB files to architect subset

Output files:
- `sampled_keys.json` — Lists of sampled primary keys (author and architect)
- `cd_64_samples.jsonl` — CD samples for author study
- `tb_64_samples.jsonl` — TB samples for author study
- `cd_14_samples.jsonl` — CD samples for architect study
- `tb_14_samples.jsonl` — TB samples for architect study

### 2. Human Evaluation

Evaluators accessed the Google Web Apps to rate all generated outputs on both metrics. Blinding and random shuffling ensured unbiased assessment. Each evaluator's scores are stored as a separate sheet in the corresponding `.xlsx` workbook, keyed by `Decision_ID` and approach position columns (`Model_A`–`Model_D`).

The Web App links are:
- CD : https://script.google.com/macros/s/AKfycbzBiiQRMRX6CJUglgm-UHKYN3wMU_Mq1kAuqL18WaFD5S2aBMZ9KhwKI_EnnVh7VvFj/exec
- TB : https://script.google.com/macros/s/AKfycbzwUH1Did2TQ9SnjgiM1CFMab62Rx4llsnOHz5njWFnvEBdeQRmjnMJVu7IJoIo8Q/exec

### 3. Results Analysis (`analysis.py`)

After collecting human ratings, run:

```bash
python analysis.py
```

This runs two analyses:

- **Author study** — reads `CD Authors.xlsx` and `TB Authors.xlsx` (2 authors × 64 samples) and writes `CD_author_results.txt` and `TB_author_results.txt`.
- **Expert study** — reads `{CD,TB} Expert.xlsx`, `{CD,TB} Expert 1to7.xlsx` and `{CD,TB} Expert 8to14.xlsx` and writes `CD_expert_results.txt` and `TB_expert_results.txt`. A senior architect rated all 14 samples; two student architects each rated a disjoint half (1–7 and 8–14). The two students are merged into a single *Student Architect* rater, so every sample has exactly one senior and one student rating. Per-student means are also reported for information but are not used for agreement or significance tests.

- **Combined** — writes `CD_combined_results.txt` and `TB_combined_results.txt` with three sections:
  - **A.** Authors restricted to the 14 expert samples (like-for-like comparison with the experts)
  - **B.** All four raters (2 authors, senior architect, student architect) on the 14 expert samples
  - **C.** All raters pooled over the 64 author samples; for significance tests each sample's score is the mean of all raters who rated it (4 raters for the 14 expert samples, 2 otherwise)

For each study the script computes:

1. **Performance means** — per-evaluator and combined mean closeness & correctness for each approach
2. **Weighted Cohen's Kappa** (quadratic) — pairwise inter-rater agreement on raw Likert scores
3. **Friedman omnibus test** — whether any significant difference exists across the 4 approaches
4. **Wilcoxon signed-rank tests** (Holm-Bonferroni corrected) — pairwise significance of DRAFT vs. each baseline

---

## Results Summary

### Context-to-Decision (CD) — Combined Means (out of 5)

| Approach   | Closeness | Correctness |
|------------|:---------:|:-----------:|
| Prompting  | 2.891     | 3.375       |
| RAFG       | 2.992     | 3.547       |
| Finetuning | 2.945     | 3.477       |
| **DRAFT**  | **3.195** | **3.680**   |

**Inter-Rater Agreement (CD):** Cohen's κ = 0.352 (Closeness) / 0.205 (Correctness)

**Statistical Significance (CD):**
- *Closeness*: Friedman p = 0.0495 (significant omnibus); pairwise DRAFT vs. all baselines — Not Significant after Holm-Bonferroni correction
- *Correctness*: Friedman p = 0.0281; **DRAFT vs. Prompting — Significant** (p = 0.0238); DRAFT ranks highest on all comparisons

---

### Title-to-Body (TB) — Combined Means (out of 5)

| Approach   | Closeness | Correctness |
|------------|:---------:|:-----------:|
| Prompting  | 2.414     | 2.984       |
| RAFG       | 2.438     | 3.086       |
| Finetuning | 2.625     | 3.094       |
| **DRAFT**  | **2.719** | **3.250**   |

**Inter-Rater Agreement (TB):** Cohen's κ = 0.258 (Closeness) / 0.259 (Correctness)

**Statistical Significance (TB):**
- *Closeness*: Friedman p = 0.0655 — No statistically significant difference
- *Correctness*: Friedman p = 0.1645 — No statistically significant difference

---

### Expert Study — Context-to-Decision (CD), Combined Means (14 samples, senior + student)

| Approach   | Closeness | Correctness |
|------------|:---------:|:-----------:|
| Prompting  | 2.964     | 3.464       |
| **RAFG**   | **3.393** | **3.821**   |
| Finetuning | 2.750     | 2.929       |
| DRAFT      | 2.679     | 2.821       |

**Inter-Rater Agreement (CD, senior vs. student):** Cohen's κ = 0.569 (Closeness) / 0.472 (Correctness), n = 56

**Statistical Significance (CD):**
- *Closeness*: Friedman p = 0.1694 — No statistically significant difference
- *Correctness*: Friedman p = 0.0312; pairwise DRAFT vs. all baselines — Not Significant after Holm-Bonferroni correction

---

### Expert Study — Title-to-Body (TB), Combined Means (14 samples, senior + student)

| Approach   | Closeness | Correctness |
|------------|:---------:|:-----------:|
| Prompting  | 2.464     | 2.500       |
| **RAFG**   | **2.464** | **3.571**   |
| Finetuning | 2.107     | 2.321       |
| DRAFT      | 2.286     | 2.321       |

**Inter-Rater Agreement (TB, senior vs. student):** Cohen's κ = 0.683 (Closeness) / 0.407 (Correctness), n = 56

**Statistical Significance (TB):**
- *Closeness*: Friedman p = 0.2386 — No statistically significant difference
- *Correctness*: Friedman p = 0.0016; **RAFG > DRAFT — Significant** (p = 0.0111); DRAFT vs. Prompting / Finetuning — Not Significant

---

### Combined (Authors + Experts) — Combined Means

| Approach   | CD Closeness (14) | CD Correctness (14) | CD Closeness (64) | CD Correctness (64) | TB Closeness (14) | TB Correctness (14) | TB Closeness (64) | TB Correctness (64) |
|------------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Prompting  | 2.893 | 3.357 | 2.904 | 3.391 | 2.446 | 2.893 | 2.423 | 2.897 |
| RAFG       | 3.286 | 3.643 | 3.064 | 3.596 | 2.554 | 3.393 | 2.442 | 3.173 |
| Finetuning | 2.893 | 3.232 | 2.910 | 3.378 | 2.321 | 2.821 | 2.532 | 2.955 |
| DRAFT      | 2.964 | 3.304 | 3.103 | 3.526 | 2.571 | 2.839 | 2.641 | 3.083 |

(14) = all four raters on the 14 expert samples; (64) = all ratings pooled over the 64 author samples. No Friedman test is significant in either combined setting (all p > 0.15), so no pairwise tests are reported. See `*_combined_results.txt` for per-rater means and pairwise κ.

---

## Data Sources

- Test ADRs: `Data/ADR-data/test.jsonl`
- Automated results: Individual `Results/` directories under Prompting, Finetune, RAFG, and DRAFT
- Human ratings: `CD Authors.xlsx`, `TB Authors.xlsx` (output from Google Web Apps, post-processed)

