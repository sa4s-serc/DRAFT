# DRAFT-ing Architectural Design Decisions using LLMs

<!-- **TOSEM 2026** | [Paper](TOSEM_2026_DRAFT.pdf) | [GitHub](https://github.com/sa4s-serc/DRAFT) -->

> Rudra Dhar · Adyansh Kakran · Amey Karan · Vasudeva Varma · Karthik Vaidhyanathan
> IIIT Hyderabad, India

---

## Overview

Architectural Decision Records (ADRs) capture key architectural decisions, but their adoption is limited by the manual effort of writing them. This repository contains the implementation and evaluation of **DRAFT** (Domain-specific Retrieval Augmented Few-shot Tuning), an approach that combines Retrieval Augmented Few-shot Generation (RAFG) with Fine-tuning to assist architects in drafting ADRs.

DRAFT is evaluated on two complementary tasks:

| Task | Input | Output |
|---|---|---|
| **Task 1 — Decision Making Support** | Decision Context (C) | Design Decision (D) |
| **Task 2 — Complete ADR Generation** | ADR Title (T) | ADR Body (B) |

Evaluated on **4,344 ADRs** with **4 LLMs** (GPT-5, Gemini-2.5-Flash, Qwen3-30B, Gemma-3-4B), DRAFT produces outputs most closely aligned with real-world ADRs on automated metrics. A human evaluation by the authors and external architects shows no uniformly preferred approach.

---

## How DRAFT Works

DRAFT operates in two phases:

**Offline Phase** — A foundational LLM is fine-tuned on retrieval-augmented prompts. For each training ADR, the top-k most semantically similar ADR pairs are retrieved from a vector database and prepended to the anchor input, forming a few-shot prompt. The model is trained on these augmented prompts so it learns to exploit retrieved examples.

**Online Phase** — At inference time, the same retrieval step is applied to the user's input. The DRAFT-ed model receives a few-shot prompt (retrieved examples + query) in the same format it was trained on, combining internalized domain knowledge with dynamic contextual grounding.

```
Offline:
  Training ADR ──► Retrieval ──► Few-shot Prompt ──► Fine-tune LLM ──► DRAFT-ed Model

Online:
  Query ──► Retrieval ──► Few-shot Prompt ──► DRAFT-ed Model ──► Generated ADR
```

See [DRAFT/Readme.md](DRAFT/Readme.md) for full implementation details.

---

## Repository Structure

```
.
├── Data/           # Dataset collection, processing, filtering, and splitting pipeline
├── LLM/            # Model and embedding model selection rationale
├── Retrieval/      # Vector DB construction and retrieval (shared by RAFG and DRAFT)
├── Prompting/      # Zero-shot prompting baseline experiments
├── RAFG/           # Retrieval-Augmented Few-shot Generation experiments
├── Finetune/       # Fine-tuning baseline experiments
├── DRAFT/          # DRAFT training and inference (main contribution)
├── HumanEval/      # Human evaluation study, analysis scripts, and results
├── Results/        # Evaluation scripts, scores, efficiency metrics, and plots
└── Diagrams/       # Architecture figures used in the paper
```

---

## Data

The dataset comprises **4,344 ADRs** from open-source GitHub repositories, sourced from a Mining Software Repositories (MSR) study of 921 ADR-using repositories. The pipeline collects, processes, and prepares Context–Decision (C-D) pairs for Task 1 and Title–Body (T-B) pairs for Task 2.

**Pipeline stages:**

1. **Aggregate** — Extract repository metadata and ADR file paths from MSR JSON files into `data.csv`
2. **Scrape** — Download raw ADR markdown files from GitHub (with `main`/`master`/API branch fallback)
3. **Dataset Creation** — Parse markdown into structured JSONL (title, body, context, decision, token counts)
4. **Filter** — Remove ADRs outside token limits (body: 10–1000 tokens), non-English content (<90% ASCII), and high URL density (>20%)
5. **De-duplicate** — Remove near-identical ADRs using `all-MiniLM-L6-v2` cosine similarity (threshold ≥ 0.98)
6. **Split** — Randomly shuffle and partition into train / val / test at 70% / 10% / 20% (3,040 / 434 / 870 ADRs)

See [Data/README.md](Data/README.md) for full details.



---

## Models

Generative models were selected from the [LM Arena text leaderboard](https://lmarena.ai/leaderboard/text) (as of Sep 8, 2025): **GPT-5-high** and **Gemini-2.5-Flash** (proprietary), and **Qwen3-30B-A3B-Instruct-2507** and **Gemma-3-4B-it** (open source). GPT-5 was evaluated with Prompting and RAFG only — not available for fine-tuning at the time of the study.

Embedding models follow a vendor-alignment strategy: proprietary LLMs are paired with their provider's embedding model (OpenAI / Google), while both open-source models use **Qwen3-Embedding-8B** (MTEB Retrieval #1, open source).

See [LLM/Readme.md](LLM/Readme.md) for selection rationale and leaderboard snapshots.

---

## Experimental Approaches

All four approaches are evaluated on both tasks across all models.

| Approach | Description | Directory |
|---|---|---|
| **Prompting** | Zero-shot: model receives only the input (C or T) with a system prompt | [Prompting/](Prompting/) |
| **RAFG** | Few-shot with top-2 retrieved examples from a VDB; no model training | [RAFG/](RAFG/) |
| **Fine-tuning** | Model trained on C-D or T-B pairs; zero-shot at inference | [Finetune/](Finetune/) |
| **DRAFT** | Model trained on retrieval-augmented prompts; same retrieval at inference | [DRAFT/](DRAFT/) |

### Setup

1. Clone the repository and install dependencies:
   ```bash
   pip install transformers datasets peft unsloth google-genai python-dotenv torch \
               evaluate rouge_score bert_score nltk sentence-transformers faiss-cpu
   ```

2. Configure environment variables in a `.env` file at the project root:
   ```
   HUGGINGFACE=your_hf_token           # For Gemma and Qwen (gated models)
   GCP_PROJECT_ID=your_project_id      # For Gemini
   LOCATION=us-central1
   OPENAI_API_KEY=your_api_key         # For GPT-5
   BUCKET_NAME=your_gcs_bucket         # GCS bucket for Gemini fine-tuning data
   VERTEXAI_API_KEY=your_api_key       # For Gemini inference after fine-tuning
   ```

### Running Experiments

Each module (`Prompting/`, `RAFG/`, `Finetune/`, `DRAFT/`) contains a `Code/` directory with one script per model per task, named `[training_|inference_]<model>_<CD|TB>.py`. Prompting and RAFG are single-step; Fine-tuning and DRAFT have separate training and inference scripts. All scripts resume automatically if interrupted. See each module's README for details.

---

## Results

Evaluated on the held-out test set of **870 ADRs** using ROUGE-1, BLEU, METEOR, and BERTScore F1 (primary metric — semantic similarity correlating best with human judgement) for effectiveness, and token counts and response time for efficiency.

The `Results/` directory contains scripts (`score.py`, `efficiency.py`, `input_tokens.py`) to compute all metrics from the prediction JSONL files produced by each approach, and a notebook (`Results/Plots/plots.ipynb`) to reproduce all paper figures. See [Results/Readme.md](Results/Readme.md) for the full tables and workflow.

### Human Evaluation

Outputs were rated blind on Closeness to Ground Truth and Architectural Correctness (5-point scale) by two authors (64 test ADRs) and by external experts (14-sample subset; one senior architect, two PhD students in software architecture).

| Raters | Result |
|---|---|
| Authors | DRAFT highest on both tasks and criteria; DRAFT > Prompting on CD Correctness (p = 0.024) |
| Experts | RAFG highest; RAFG > DRAFT on TB Correctness (p = 0.011), driven by the senior architect |
| All raters | No significant differences |

See [HumanEval/README.md](HumanEval/README.md) for scores, inter-rater agreement, and significance tests.

---

## Key Findings

- **Automated metrics:** DRAFT achieves the highest BERTScore F1 for Qwen3-30B and Gemini-2.5-Flash on both tasks. For Gemma-3-4B, RAFG leads on Task 1 and DRAFT on Task 2.
- **Task dependence:** DRAFT's gains over all baselines are significant for every model on Task 2, but model-dependent on Task 1.
- **Output length:** DRAFT-ed models produce outputs close to ground-truth length (61–140 tokens on Task 1 vs. 147–672 for prompting); output length is the main driver of response time.
- **Deployment:** DRAFT-ed open-source models (Qwen3-30B, Gemma-3-4B) achieve performance comparable to larger proprietary models and can be hosted on-premises.
- **Human evaluation:** Similarity to real ADRs does not imply higher perceived quality. Authors preferred DRAFT, the senior architect preferred the more elaborate RAFG outputs, and junior architects found little difference, noting that DRAFT outputs often omit rationale.


---

## Citation

```bibtex
@article{dhar2026draft,
  title     = {DRAFT-ing Architectural Design Decisions using LLMs},
  author    = {Dhar, Rudra and Kakran, Adyansh and Karan, Amey and Varma, Vasudeva and Vaidhyanathan, Karthik},
  journal   = {ACM Transactions on Software Engineering and Methodology},
  year      = {2026},
  publisher = {ACM},
  doi       = {10.1145/nnnnnnn.nnnnnnn}
}
```

---

## Related Work

This paper builds on our prior empirical study on LLM-based ADR generation:

> Rudra Dhar, Karthik Vaidhyanathan, and Vasudeva Varma. *Can LLMs Generate Architectural Design Decisions? — An Exploratory Empirical Study.* 2024.
