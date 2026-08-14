# Human Evaluation Framework

This directory contains the human evaluation study for assessing the quality of ADR generation across four approaches: Prompting, Finetuning, RAFG, and DRAFT. Two complementary studies were conducted with domain experts and architects using custom Google Web Apps.

---

## Study Design

The human evaluation consists of two hierarchical studies:

- **Author Study**: 64 sampled ADR instances evaluated by domain experts
- **Architect Study**: 14 instances (subset of Author Study) evaluated by architects for deeper analysis

Each study covers two generation tasks:
- **Context-to-Decision (CD)**: Evaluating decision generation quality given a context
- **Title-to-Body (TB)**: Evaluating full ADR body generation given only a title

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

## Evaluation Platform

Four custom Google Web Apps were built to conduct the human evaluations:

1. **Author CD App**: Context-Decision task for 64 author samples
2. **Author TB App**: Title-Body task for 64 author samples
3. **Architect CD App**: Context-Decision task for 14 expert architect samples
4. **Architect TB App**: Title-Body task for 14 expert architect samples

Each app presented:
- The source material (Context + ground truth Decision for CD; Title + ground truth Body for TB)
- Four blinded, randomly shuffled generated outputs (one from each approach)
- For each generation, evaluators assigned two 5-star ratings:
  - **Architectural Correctness**: How well the decision/body follows sound software architecture principles
  - **Closeness to Ground Truth**: How similar the generation is to the reference solution

---

## Workflow

### 1. Dataset Preparation (HumanEval.ipynb)

The notebook automates the entire evaluation dataset construction:

- **Sample 64 keys** from the test set (reproducible with seed 42)
- **Build CD dataset**: Aggregate context, ground truth decisions, and generated decisions from all four approaches
- **Build TB dataset**: Aggregate titles, ground truth bodies, and generated bodies from all four approaches
- **Sample 14 architect keys**: Randomly select a subset from the 64 author samples
- **Create 14-sample datasets**: Filter CD and TB files to architect subset

Output files:
- `sampled_keys.json` - Lists of sampled primary keys (author and architect)
- `cd_64_samples.jsonl` - CD samples for author study
- `tb_64_samples.jsonl` - TB samples for author study
- `cd_14_samples.jsonl` - CD samples for architect study
- `tb_14_samples.jsonl` - TB samples for architect study

### 2. Human Evaluation

Evaluators accessed the Google Web Apps to rate all generated outputs on both metrics. Blinding and random shuffling ensured unbiased assessment.

### 3. Results Analysis

After collecting human ratings, the results are aggregated and analyzed to compute:
- Mean scores for each approach per metric
- Statistical significance tests comparing approaches
- Correlation with automated metrics (BertScore)
- Qualitative patterns and failure modes

Final results are stored in `Results/` directory.

---

## Data Sources

- Test ADRs: `Data/ADR-data/test.jsonl`
- Automated results: Individual `Results/` directories under Prompting, Finetune, RAFG, and DRAFT
- Human ratings: Output from Google Web Apps (post-processed and stored in `Results/`)

---

## Related Sections

- [DRAFT](../DRAFT) - DRAFT approach implementation
- [Finetune](../Finetune) - Finetuning approach
- [RAFG](../RAFG) - RAFG approach
- [Prompting](../Prompting) - Prompting approach
- [Results](../Results) - Consolidated result analysis and visualizations
- [Data](../Data) - ADR dataset and preprocessing pipeline
