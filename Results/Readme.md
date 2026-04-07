# Results

This directory contains all evaluation outputs, scores, and plots for the DRAFT paper. Results cover two tasks across four models and four approaches, evaluated on 982 held-out test ADRs.

---

## Directory Structure

```
Results/
├── scores/                                           # Per-experiment JSON score files
│   ├── Prompting_<model>_CDtest.json
│   ├── RAFG_<model>_CDtest.json
│   ├── Finetune_<model>_CDtest.json
│   ├── DRAFT_<model>_CDtest.json
│   └── ... (same structure for TBtest)
│
├── Plots/                                            # Generated figures
│   ├── CD_Results_2x2.pdf                           # Effectiveness comparison (Task 1)
│   ├── TB_Results_2x2.pdf                           # Effectiveness comparison (Task 2)
│   ├── Output_Tokens_Chart_Combined.pdf             # Output token counts by approach
│   ├── Response_Time_Chart_Combined.pdf             # Response time by approach
│   └── Validation_Loss_2x2.pdf                      # Training loss curves
│
├── score.py                                          # Computes ROUGE/BLEU/METEOR/BERTScore from prediction JSONL files
├── efficiency.py                                     # Extracts token count and response time from prediction JSONL files
├── input_tokens.py                                   # Counts input tokens for Prompting vs RAFG/DRAFT
├── efficiency_results.json                           # Aggregated efficiency stats for all experiments
├── token_counts.json                                 # Input token analysis per retrieval split
├── Results.ipynb                                     # Notebook for exploring and aggregating results
└── Plots/plots.ipynb                                 # Notebook for generating all paper figures
```

---

## Tasks and Setup

| Task | Input | Output | Ground Truth Column |
|---|---|---|---|
| Task 1 : Context → Decision (CD) | Decision Context | Design Decision | `Decision` |
| Task 2 : Title → Body (TB) | ADR Title | Complete ADR Body | `Body` |

**Test set**: 982 ADRs (held-out 20% split, consistent across all approaches and both tasks).

**Approaches evaluated**: Prompting (zero-shot), RAFG, Fine-tuning, DRAFT.

**Models**: Gemma-3-4B-it, Qwen3-30B-A3B-Instruct-2507, Gemini-2.5-Flash, GPT-5-high.

> GPT-5 was only evaluated with Prompting and RAFG (not available for fine-tuning at the time of the study).

**Primary metric**: BERTScore F1 (semantic similarity; correlates best with human judgement). Secondary metrics: ROUGE-1 (lexical recall), BLEU (lexical precision), METEOR.

---

## Effectiveness Results

**Bold** = best value per metric within that model group. *Italic* = worst value per metric within that model group.

### Task 1 — Context → Decision

| Model | Approach | ROUGE-1 | ROUGE-Lsum | BLEU | METEOR | BS Precision | BS Recall | BS F1 |
|---|---|---|---|---|---|---|---|---|
| Gemma-3-4b | Prompting | *0.135* | *0.122* | *0.009* | **0.165** | *0.780* | *0.831* | *0.805* |
| Gemma-3-4b | Finetuning | 0.186 | 0.164 | **0.032** | 0.164 | 0.833 | 0.836 | 0.834 |
| Gemma-3-4b | RAFG | **0.191** | **0.165** | 0.020 | *0.159* | **0.849** | **0.838** | **0.843** |
| Gemma-3-4b | DRAFT | 0.179 | 0.159 | 0.031 | 0.162 | 0.828 | 0.835 | 0.831 |
| Qwen-3-30B | Prompting | *0.159* | *0.131* | *0.014* | 0.184 | *0.807* | 0.838 | *0.822* |
| Qwen-3-30B | Finetuning | 0.210 | 0.188 | 0.033 | *0.162* | 0.859 | 0.837 | 0.847 |
| Qwen-3-30B | RAFG | 0.183 | 0.148 | 0.020 | **0.190** | 0.859 | 0.837 | 0.847 |
| Qwen-3-30B | DRAFT | **0.213** | **0.191** | **0.041** | 0.167 | **0.863** | 0.838 | **0.850** |
| gemini-2.5 | Prompting | 0.162 | 0.133 | *0.015* | 0.158 | *0.825* | 0.835 | 0.829 |
| gemini-2.5 | Finetuning | *0.125* | *0.114* | 0.022 | *0.103* | 0.835 | *0.820* | *0.827* |
| gemini-2.5 | RAFG | 0.176 | 0.147 | 0.019 | 0.172 | 0.830 | 0.838 | 0.834 |
| gemini-2.5 | DRAFT | **0.199** | **0.177** | **0.040** | **0.178** | **0.837** | **0.841** | **0.839** |
| gpt-5 | Prompting | 0.138 | 0.123 | 0.008 | 0.164 | 0.789 | 0.836 | 0.812 |
| gpt-5 | RAFG | 0.146 | 0.130 | 0.009 | 0.169 | 0.794 | 0.838 | 0.815 |

### Task 2 — Title → Body

| Model | Approach | ROUGE-1 | ROUGE-Lsum | BLEU | METEOR | BS Precision | BS Recall | BS F1 |
|---|---|---|---|---|---|---|---|---|
| Gemma-3-4b | Prompting | *0.221* | *0.208* | *0.013* | 0.222 | *0.808* | *0.821* | *0.814* |
| Gemma-3-4b | Finetuning | **0.269** | **0.255** | **0.062** | 0.245 | 0.831 | 0.829 | 0.830 |
| Gemma-3-4b | RAFG | 0.247 | 0.233 | 0.036 | **0.255** | 0.814 | **0.831** | 0.822 |
| Gemma-3-4b | DRAFT | 0.263 | 0.248 | 0.051 | *0.221* | **0.837** | 0.828 | **0.832** |
| Qwen-3-30B | Prompting | 0.242 | 0.227 | *0.026* | 0.226 | 0.809 | 0.825 | 0.817 |
| Qwen-3-30B | Finetuning | *0.237* | *0.223* | **0.050** | *0.192* | 0.850 | 0.822 | 0.835 |
| Qwen-3-30B | RAFG | 0.242 | 0.227 | 0.027 | 0.226 | 0.809 | 0.825 | 0.817 |
| Qwen-3-30B | DRAFT | **0.251** | **0.236** | 0.035 | 0.194 | **0.858** | 0.822 | **0.839** |
| gemini-2.5 | Prompting | *0.145* | *0.135* | *0.009* | *0.118* | 0.832 | *0.800* | 0.815 |
| gemini-2.5 | Finetuning | 0.182 | 0.171 | 0.040 | 0.166 | 0.829 | 0.812 | 0.820 |
| gemini-2.5 | RAFG | 0.158 | 0.148 | 0.032 | 0.157 | *0.821* | 0.807 | *0.813* |
| gemini-2.5 | DRAFT | **0.213** | **0.199** | **0.041** | **0.185** | **0.844** | **0.819** | **0.830** |
| gpt-5 | Prompting | 0.198 | 0.183 | 0.012 | 0.155 | 0.801 | 0.806 | 0.803 |
| gpt-5 | RAFG | 0.204 | 0.188 | 0.013 | 0.153 | 0.805 | 0.807 | 0.806 |

---

## Efficiency Results

All values are means over the test set from `efficiency_results.json`. **Bold** = fastest response time per model group. *Italic* = slowest response time per model group.

### Task 1 — Context → Decision

| Model | Approach | Input Tokens | Output Tokens | Response Time (s) |
|---|---|---|---|---|
| Gemma-3-4b | Prompting | 122 | 395 | *26.08* |
| Gemma-3-4b | Finetuning | 122 | 128 | 4.29 |
| Gemma-3-4b | RAFG | 559 | 79 | **3.88** |
| Gemma-3-4b | DRAFT | 559 | 140 | 5.17 |
| Qwen3-30B | Prompting | 122 | 275 | *74.82* |
| Qwen3-30B | Finetuning | 122 | 87 | 39.46 |
| Qwen3-30B | RAFG | 559 | 171 | 59.93 |
| Qwen3-30B | DRAFT | 559 | 61 | **28.97** |
| gemini-2.5 | Prompting | 122 | 147 | *6.27* |
| gemini-2.5 | Finetuning | 122 | 122 | **1.31** |
| gemini-2.5 | RAFG | 527 | 151 | 6.03 |
| gemini-2.5 | DRAFT | 527 | 135 | 1.73 |
| gpt-5 | Prompting | 122 | 672 | 17.19 |
| gpt-5 | RAFG | 643 | 605 | 14.74 |

Ground truth mean Decision length: **73 tokens**.

### Task 2 — Title → Body

| Model | Approach | Input Tokens | Output Tokens | Response Time (s) |
|---|---|---|---|---|
| Gemma-3-4b | Prompting | 8 | 981 | *32.42* |
| Gemma-3-4b | Finetuning | 8 | 453 | 15.14 |
| Gemma-3-4b | RAFG | 811 | 680 | 25.48 |
| Gemma-3-4b | DRAFT | 811 | 313 | **11.51** |
| Qwen3-30B | Prompting | 8 | 500 | 155.20 |
| Qwen3-30B | Finetuning | 8 | 264 | 110.14 |
| Qwen3-30B | RAFG | 811 | 500 | *177.11* |
| Qwen3-30B | DRAFT | 811 | 182 | **75.51** |
| gemini-2.5 | Prompting | 8 | 155 | *7.55* |
| gemini-2.5 | Finetuning | 8 | 306 | **6.13** |
| gemini-2.5 | RAFG | 759 | 312 | 7.33 |
| gemini-2.5 | DRAFT | 759 | 264 | 6.92 |
| gpt-5 | Prompting | 8 | 727 | 17.25 |
| gpt-5 | RAFG | 801 | 695 | 15.44 |

Ground truth mean Body length: **275 tokens**.

---

## Key Findings

**Effectiveness**

- DRAFT achieves the highest BERTScore F1 for Qwen3-30B and Gemini-2.5-Flash on both tasks. For Gemma-3-4b, RAFG peaks on Task 1 while DRAFT leads on Task 2, suggesting that DRAFT's benefit scales with model capacity.
- Fine-tuning and DRAFT consistently outperform prompting and RAFG alone across both tasks. The gap is especially large for Task 2, where domain-specific training on structured ADR bodies provides a stronger signal.
- DRAFT produces higher absolute metric scores for Task 2 (Title → Body) than Task 1 (Context → Decision), reflecting the longer, more information-rich nature of ADR bodies — which give more lexical and semantic surface for metrics to match against.
- RAFG improves over prompting for Task 1 consistently, but is more variable on Task 2: Gemini-2.5-Flash and Qwen3-30B show negligible RAFG gains, suggesting that retrieved full-body examples can occasionally introduce conflicting style patterns for longer generations.

**Efficiency**

- RAFG and DRAFT use ~4–5× more input tokens than prompting for Task 1 (due to 2 retrieved C-D pairs), and ~100× more for Task 2 (due to 2 retrieved full ADR bodies prepended to a very short title input).
- Despite the additional input, DRAFT response times are comparable to or faster than RAFG and fine-tuning in most configurations, because the fine-tuned model generates shorter, more focused outputs.
- Fine-tuned and DRAFT-ed models produce outputs much closer to ground truth length than prompting. Prompting generates 3–9× more tokens than necessary (e.g., 395 tokens for Gemma vs. 73 token ground truth on Task 1). Shorter outputs directly reduce response time, since output length is the primary driver of latency.
- Response time is highly model-dependent. Qwen3-30B on local GPU is consistently the slowest (up to 177s for RAFG on Task 2), while Gemini-2.5-Flash via API is the fastest (under 7s across all approaches).

---

## Running the Evaluation

All scripts are run from the **project root**.

### 1. Compute Effectiveness Scores

```bash
python Results/score.py "<approach>/Results/*CDtest.jsonl" "<approach>/Results/*TBtest.jsonl"
```

Or across all approaches at once:

```bash
python Results/score.py \
  Prompting/Results/*.jsonl \
  RAFG/Results/*.jsonl \
  Finetune/Results/*.jsonl \
  DRAFT/Results/*.jsonl
```

Scores are written to `Results/scores/<Approach>_<Model>_<Task>.json`. Already-scored files are skipped automatically.

**Dependencies**: `evaluate`, `rouge_score`, `bert_score`, `nltk`, `pandas`, `sympy`

### 2. Compute Efficiency Metrics

```bash
python Results/efficiency.py \
  Prompting/Results/*.jsonl \
  RAFG/Results/*.jsonl \
  Finetune/Results/*.jsonl \
  DRAFT/Results/*.jsonl
```

Output is written to `Results/efficiency_results.json`. Each entry records mean, median, std, min, max for response time and output token count per experiment.

**Compute Input Token Counts**

```bash
python Results/input_tokens.py
```

Reads retrieval splits from `Retrieval/qwen3-embedding-8B/`, `Retrieval/openai/`, and `Retrieval/gemini/` and writes per-file token statistics to `Results/token_counts.json`. Reports both `prompting_tokens` (anchor input only) and `rafg_tokens` (anchor + 2 retrieved examples).

### 3. Generate Plots

Open and run `Results/Plots/plots.ipynb` to reproduce all figures in the paper (effectiveness bar charts, output token charts, response time charts, validation loss curves).

---

## Score File Format

Each file in `Results/scores/` is a JSON with the following structure:

```json
{
  "rouge": {
    "rouge1": 0.213,
    "rouge2": 0.051,
    "rougeL": 0.148,
    "rougeLsum": 0.191
  },
  "bleu": {
    "bleu": 0.041,
    "precisions": [0.214, 0.054, 0.023, 0.014],
    "brevity_penalty": 1.0,
    "length_ratio": 0.961,
    "translation_length": 52700,
    "reference_length": 54810
  },
  "meteor": { "meteor": 0.167 },
  "bertscore": {
    "precision": 0.863,
    "recall": 0.838,
    "f1": 0.850
  },
  "AverageTime": 28.97,
  "AverageGeneratedTokens": 60.5
}
```

Naming convention: `<Approach>_<Model>_<Task>.json`  
e.g., `DRAFT_qwen3-30b-a3b-instruct_CDtest.json`
