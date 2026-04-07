# Fine-tuning for ADR Generation

This directory implements fine-tuning of LLMs to generate Architectural Decision Records (ADRs). Fine-tuning embeds ADR domain knowledge directly into the model's parameters, enabling it to produce outputs that are more aligned with real-world architectural documentation than zero-shot prompting alone.

Two tasks are supported:
- **Context-to-Decision (CD)**: Generate a Decision section given a Context
- **Title-to-Body (TB)**: Generate a complete ADR Body given a Title

Each task has dedicated training and inference scripts per model.

---

## Supported Models

1. Gemini 2.5 Flash (via Vertex AI Supervised Fine-Tuning API)
2. Qwen3 30B A3B Instruct (via Unsloth + LoRA, 4-bit quantized)
3. Gemma 3 4B Instruct (full-parameter fine-tuning via HuggingFace Transformers)

> GPT-5 was not fine-tuned as it was not available for fine-tuning at the time of the study.

---

## Directory Structure

```
Finetune/
├── Code/                                              # Training and inference scripts
│   ├── training_gemini-2.5-flash_CD.py               # Gemini fine-tuning for Context->Decision
│   ├── training_gemini-2.5-flash_TB.py               # Gemini fine-tuning for Title->Body
│   ├── training_gemma-3-4b-it_CD.py                  # Gemma full fine-tuning for Context->Decision
│   ├── training_gemma-3-4b-it_TB.py                  # Gemma full fine-tuning for Title->Body
│   ├── training_qwen3-30b-a3b-instruct-2507_CD.py    # Qwen LoRA fine-tuning for Context->Decision
│   ├── training_qwen3-30b-a3b-instruct-2507_TB.py    # Qwen LoRA fine-tuning for Title->Body
│   ├── inference_gemini-2.5-flash_CD.py              # Inference with fine-tuned Gemini (CD)
│   ├── inference_gemini-2.5-flash_TB.py              # Inference with fine-tuned Gemini (TB)
│   ├── ...
│
├── Output/                                            # Training artifacts
│   ├── loss_log_gemini-2.5-flash_CD.jsonl            # Per-epoch train/val loss for Gemini (CD)
│   ├── loss_log_gemini-2.5-flash_TB.jsonl            # Per-epoch train/val loss for Gemini (TB)
│   ├── ...
│
├── Results/                                           # Inference outputs on the test set
│   ├── gemini-2.5-flash_CDtest.jsonl                  # Outputs generated with Finetuned gemini-2.5-flash for Context->Decision on test set
│   ├── gemini-2.5-flash_TBtest.jsonl                  # Outputs generated with Finetuned gemini-2.5-flash for Title->Body on test set
│   ├── ...
│
└── Readme.md
```

---

## Workflow

Fine-tuning operates in two phases: an **offline training phase** and an **online inference phase**.

### Offline Phase: Training

**Data source**: Training and validation splits from `Retrieval/` (plain zero-shot C-D or T-B pairs — no retrieved examples).

**Input format** (per entry in `Retrieval/CDtrain.jsonl`):
```json
{
  "Anchor": {
    "PrimaryKey": "...",
    "Context": "...",
    "Decision": "..."
  }
}
```

**Prompt construction**: Each training example is formatted as a chat-template message sequence:
```
[system]: You are an expert software architect...
[user]:   ## Context: <context>
[assistant]: ## Decision: <decision>
```
For TB, replace Context/Decision with Title/Body.

**Model-specific training strategies**:

| Model | Strategy | Context length | Batch size |
|---|---|---|---|
| Gemma 3 4B | Full-parameter, bfloat16 | 1024 tokens | 2 (grad accum ×4) |
| Qwen3 30B | LoRA (r=32, α=16) via Unsloth, 4-bit | 1024 tokens | 4 (grad accum ×2) |
| Gemini 2.5 Flash | Vertex AI SFT API (cloud-managed) | — | — |

All models train for up to **5 epochs**. The best checkpoint is selected by lowest validation loss.

**Loss logging**: Per-epoch train and validation loss is written to `Output/loss_log_<model>_<task>.jsonl` after each epoch, enabling training curve analysis.

### Online Phase: Inference

The best checkpoint (lowest validation loss) is loaded and run over the test set (`Retrieval/CDtest.jsonl` or `Retrieval/TBtest.jsonl`).

**Inference prompt** (same zero-shot format as training — no retrieved examples):
```
[system]: You are an expert software architect...
[user]:   ## Context: <context>
```

For Gemini, the tuned model endpoint is called via the `google-genai` client using Vertex AI. For Gemma and Qwen, the saved checkpoint is loaded locally and run on GPU.

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

3. Ensure the dataset splits exist:
   - `Retrieval/CDtrain.jsonl`, `Retrieval/CDval.jsonl`, `Retrieval/CDtest.jsonl`
   - `Retrieval/TBtrain.jsonl`, `Retrieval/TBval.jsonl`, `Retrieval/TBtest.jsonl`
   - For Qwen: `Retrieval/qwen3-embedding-8B/CDtrain.jsonl` etc.

### Training

```bash
# Gemma — full fine-tuning
python Finetune/Code/training_gemma-3-4b-it_CD.py
python Finetune/Code/training_gemma-3-4b-it_TB.py

# Qwen — LoRA via Unsloth
python Finetune/Code/training_qwen3-30b-a3b-instruct-2507_CD.py
python Finetune/Code/training_qwen3-30b-a3b-instruct-2507_TB.py

# Gemini — submits a Vertex AI SFT job (runs asynchronously in the cloud)
python Finetune/Code/training_gemini-2.5-flash_CD.py
python Finetune/Code/training_gemini-2.5-flash_TB.py
```

Checkpoints are saved after each epoch to `Finetune/Output/`. Update the `checkpoint_dir` / `MODEL_ID` in the inference scripts to point to the best checkpoint before running inference.

### Inference

```bash
python Finetune/Code/inference_gemma-3-4b-it_CD.py
python Finetune/Code/inference_qwen3-30b-a3b-instruct-2507_TB.py
python Finetune/Code/inference_gemini-2.5-flash_CD.py
```

**Resuming interrupted runs**: Gemini inference scripts automatically skip already-processed entries; re-run to continue from where it left off.

---

## Output Format

Results are saved as JSONL in `Results/`, one entry per test ADR:

```json
{
  "PrimaryKey": "repo/path/to/adr.md",
  "Decision": "We will use Jest as our testing framework.",
  "GeneratedTokens": 128,
  "Time": 4.29
}
```

Loss logs in `Output/` record per-epoch metrics:

```json
{"epoch": 0, "train_loss": 3.812, "val_loss": 3.754}
{"epoch": 1, "train_loss": 2.103, "val_loss": 1.987}
```
