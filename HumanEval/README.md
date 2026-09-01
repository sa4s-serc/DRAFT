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
| `CD_results.txt` | Generated analysis report for the CD task |
| `TB_results.txt` | Generated analysis report for the TB task |
| `sampled_keys.json` | Sampled primary keys (author: 64, architect: 14) |
| `cd_64_samples.jsonl` | CD evaluation dataset for author study (64 samples) |
| `tb_64_samples.jsonl` | TB evaluation dataset for author study (64 samples) |
| `cd_14_samples.jsonl` | CD evaluation dataset for architect study (14 samples) |
| `tb_14_samples.jsonl` | TB evaluation dataset for architect study (14 samples) |

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

This reads `CD Authors.xlsx` and `TB Authors.xlsx` and writes `CD_results.txt` and `TB_results.txt`. The script computes:

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

**Inter-Rater Agreement (TB):** Cohen's κ = 0.266 (Closeness) / 0.272 (Correctness)

**Statistical Significance (TB):**
- *Closeness*: Friedman p = 0.0727 — No statistically significant difference
- *Correctness*: Friedman p = 0.1826 — No statistically significant difference

---

## Data Sources

- Test ADRs: `Data/ADR-data/test.jsonl`
- Automated results: Individual `Results/` directories under Prompting, Finetune, RAFG, and DRAFT
- Human ratings: `CD Authors.xlsx`, `TB Authors.xlsx` (output from Google Web Apps, post-processed)

