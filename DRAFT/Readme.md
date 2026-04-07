# DRAFT — Domain-specific Retrieval Augmented Few-shot Tuning

This directory implements DRAFT: a novel approach that combines Retrieval Augmented Few-shot Generation (RAFG) with fine-tuning. Rather than treating retrieval and fine-tuning as separate techniques, DRAFT trains the model *on retrieval-augmented prompts*, teaching it to leverage few-shot examples from similar historical ADRs at inference time. This gives the model both internalized domain knowledge (from fine-tuning) and dynamic contextual grounding (from retrieval).

Two tasks are supported:
- **Context-to-Decision (CD)**: Generate a Decision section given a Context and retrieved few-shot examples
- **Title-to-Body (TB)**: Generate a complete ADR Body given a Title and retrieved few-shot examples

---

## How DRAFT Works

DRAFT integrates two complementary techniques. RAFG improves output quality by supplying contextually relevant examples at inference time, but the underlying model has no domain-specific knowledge of ADRs. Fine-tuning internalizes domain knowledge into the model's weights, but at inference time the model sees only the raw input with no examples to guide generation.

DRAFT resolves this tension by fine-tuning the model *on retrieval-augmented prompts*. During the offline phase, for each training ADR, the top-2 most semantically similar Context–Decision pairs are retrieved from a vector database and prepended to the anchor context to form a few-shot prompt. The model is trained on these augmented prompts, so it learns to exploit the retrieved examples. During the online phase, the same retrieval step is applied at inference time, producing a prompt in the same format the model was trained on. This means the DRAFT-ed model benefits from both internalized domain knowledge and dynamic retrieval at every generation step.

---

## Supported Models

1. Gemini 2.5 Flash (via Vertex AI Supervised Fine-Tuning API)
2. Qwen3 30B A3B Instruct (via Unsloth + LoRA, 4-bit quantized)
3. Gemma 3 4B Instruct (full-parameter fine-tuning via HuggingFace Transformers)

> GPT-5 was not DRAFT-ed as it was not available for fine-tuning at the time of the study.

---

## Directory Structure

```
DRAFT/
├── Code/                                              # Training and inference scripts
│   ├── training_gemini-2.5-flash_CD.py               # Gemini DRAFT training for Context->Decision
│   ├── training_gemini-2.5-flash_TB.py               # Gemini DRAFT training for Title->Body
│   ├── training_gemma-3-4b-it_CD.py                  # Gemma DRAFT training for Context->Decision
│   ├── training_gemma-3-4b-it_TB.py                  # Gemma DRAFT training for Title->Body
│   ├── training_qwen3-30b-a3b-instruct-2507_CD.py    # Qwen DRAFT training for Context->Decision
│   ├── training_qwen3-30b-a3b-instruct-2507_TB.py    # Qwen DRAFT training for Title->Body
│   ├── inference_gemini-2.5-flash_CD.py              # Inference with DRAFT-ed Gemini (CD)
│   ├── inference_gemini-2.5-flash_TB.py              # Inference with DRAFT-ed Gemini (TB)
│   ├── ...
│
├── Output/                                            # Training artifacts
│   ├── loss_log_gemini-2.5-flash_CD.jsonl            # Per-epoch train/val loss for Gemini (CD)
│   ├── loss_log_gemini-2.5-flash_TB.jsonl            # Per-epoch train/val loss for Gemini (TB)
│   ├── ...
│
├── Results/                                           # Inference outputs on the test set
│   ├── gemini-2.5-flash_CDtest.jsonl                  # Outputs generated with DRAFT-ed gemini-2.5-flash for Context->Decision on test set
│   ├── gemini-2.5-flash_TBtest.jsonl                  # Outputs generated with DRAFT-ed gemini-2.5-flash for Title->Body on test set
│   ├── gemma-3-4b-it-CDtest.jsonl
│   ├── gemma-3-4b-it-TBtest.jsonl
│   ├── qwen3-30b-a3b-instruct-CDtest.jsonl
│   └── qwen3-30b-a3b-instruct-TBtest.jsonl
│
└── Readme.md
```

---

## Workflow

DRAFT operates in two phases: an **offline training phase** and an **online inference phase**. Both phases use retrieval-augmented few-shot prompts built from the VDB constructed in `Retrieval/`.

### Offline Phase: Training

**Data source**: Retrieval-augmented splits from `Retrieval/<embedding_model>/` (each entry contains the anchor ADR plus the top-2 most similar ADRs retrieved from the training VDB).

**Input format** (per entry in `Retrieval/qwen3-embedding-8B/CDtrain.jsonl`):
```json
{
  "Anchor": {
    "PrimaryKey": "...",
    "Context": "...",
    "Decision": "..."
  },
  "Retrieved": [
    {"Context": "...", "Decision": "..."},
    {"Context": "...", "Decision": "..."}
  ]
}
```

**Prompt construction**: Each training example is formatted as a multi-turn chat-template sequence where the two retrieved pairs appear as prior exchanges before the anchor:
```
[system]: You are an expert software architect... Below are a few examples...
[user]:   ## Context: <retrieved_context_1>
[assistant]: ## Decision: <retrieved_decision_1>
[user]:   ## Context: <retrieved_context_2>
[assistant]: ## Decision: <retrieved_decision_2>
[user]:   ## Context: <anchor_context>
[assistant]: ## Decision: <anchor_decision>   ← training target
```
For TB, replace Context/Decision with Title/Body throughout.

**Model-specific training strategies**:

| Model | Strategy | Context length | Batch size |
|---|---|---|---|
| Gemma 3 4B | Full-parameter, bfloat16 | 3072 tokens | 1 (grad accum ×8) |
| Qwen3 30B | LoRA (r=32, α=16) via Unsloth, 4-bit | 3072 tokens | 1 (grad accum ×8) |
| Gemini 2.5 Flash | Vertex AI SFT API (cloud-managed) | — | — |

The context window is larger than in plain fine-tuning (3072 vs 1024 tokens) to accommodate the two retrieved examples prepended to each prompt.

All models train for up to **5 epochs**. The best checkpoint is selected by lowest validation loss.

**Loss logging**: Per-epoch train and validation loss is written to `Output/loss_log_<model>_<task>.jsonl` after each epoch.

### Online Phase: Inference

The best checkpoint is loaded and run over the test set. The same retrieval-augmented prompt format used during training is applied at inference — retrieved pairs from `Retrieval/<embedding_model>/CDtest.jsonl` are prepended to the anchor context before passing to the DRAFT-ed model.

**Inference prompt** (same few-shot format as training):
```
[system]: You are an expert software architect... Below are a few examples...
[user]:   ## Context: <retrieved_context_1>
[assistant]: ## Decision: <retrieved_decision_1>
[user]:   ## Context: <retrieved_context_2>
[assistant]: ## Decision: <retrieved_decision_2>
[user]:   ## Context: <anchor_context>
```

The model generates the Decision (or Body for TB) as its response.

---

## Running the Experiments

### Setup

1. Install dependencies:
   ```bash
   pip install transformers datasets peft unsloth google-genai python-dotenv torch
   ```

2. Configure environment variables in `.env` at the project root:
   ```
   HUGGINGFACE=your_hf_token          # For Gemma and Qwen (gated models)
   GCP_PROJECT_ID=your_project_id     # For Gemini
   LOCATION=us-central1
   BUCKET_NAME=your_gcs_bucket        # GCS bucket holding training data for Gemini
   VERTEXAI_API_KEY=your_api_key      # For Gemini inference
   ```

3. Ensure retrieval-augmented dataset splits exist (produced by `Retrieval/`):
   - `Retrieval/qwen3-embedding-8B/CDtrain.jsonl`, `CDval.jsonl`, `CDtest.jsonl`
   - `Retrieval/qwen3-embedding-8B/TBtrain.jsonl`, `TBval.jsonl`, `TBtest.jsonl`
   - Corresponding paths for Gemini and Gemma (vendor-aligned embedding models)

### Training

```bash
# Gemma — full fine-tuning on retrieval-augmented prompts
python DRAFT/Code/training_gemma-3-4b-it_CD.py
python DRAFT/Code/training_gemma-3-4b-it_TB.py

# Qwen — LoRA via Unsloth on retrieval-augmented prompts
python DRAFT/Code/training_qwen3-30b-a3b-instruct-2507_CD.py
python DRAFT/Code/training_qwen3-30b-a3b-instruct-2507_TB.py

# Gemini — submits a Vertex AI SFT job (runs asynchronously in the cloud)
python DRAFT/Code/training_gemini-2.5-flash_CD.py
python DRAFT/Code/training_gemini-2.5-flash_TB.py
```

Checkpoints are saved after each epoch to `DRAFT/Output/`. Update `checkpoint_dir` / `MODEL_ID` in the inference scripts to point to the best checkpoint before running inference.

### Inference

```bash
python DRAFT/Code/inference_gemma-3-4b-it_CD.py
python DRAFT/Code/inference_qwen3-30b-a3b-instruct-2507_TB.py
python DRAFT/Code/inference_gemini-2.5-flash_CD.py
```

**Resuming interrupted runs**: Gemini inference scripts automatically skip already-processed entries; re-run to continue from where it left off.

---

## Output Format

Results are saved as JSONL in `Results/`, one entry per test ADR:

```json
{
  "PrimaryKey": "repo/path/to/adr.md",
  "Decision": "We will use NPM instead of Yarn for this project.",
  "GeneratedTokens": 61,
  "Time": 28.97
}
```

Loss logs in `Output/` record per-epoch metrics:

```json
{"epoch": 0, "train_loss": 3.951, "val_loss": 3.874}
{"epoch": 1, "train_loss": 2.241, "val_loss": 2.103}
```
