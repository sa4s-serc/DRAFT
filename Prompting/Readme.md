# Prompting Engine for ADR Generation

This directory contains scripts for generating Architectural Decision Records (ADRs) using various Large Language Models (LLMs). The pipeline supports two generation tasks:
- **Context-to-Decision (CD)**: Generate a Decision section given a Context
- **Title-to-Body (TB)**: Generate a complete Body given a Title

Each task is implemented across multiple state-of-the-art models to evaluate performance and quality.

---

## Supported Models

1. Gemini 2.5 Flash
2. GPT-5
3. Qwen3 30B (A3B Instruct)
4. Gemma 3 4B (Instruct)

---

## Directory Structure

```
Prompting/
├── Code/                             # Scripts to construct prompts and run generation experiments
│   ├── gemini-2.5-flash_CD.py        # Context->Decision generation with Gemini
│   ├── gemini-2.5-flash_TB.py        # Title->Body generation with Gemini
│   ├── ...
│
├── Results/                          # Generated outputs in jsonl format
│   ├── gemini-2.5-flash_CDtest.jsonl # Outputs generated with gemini-2.5-flash for Context->Decision on test set
│   ├── gemini-2.5-flash_TBtest.jsonl # Outputs generated with gemini-2.5-flash for Title->Body on test set
│   ├── ...
│
└── Readme.md
```

---

## Workflow

### Task 1: Context-to-Decision (CD)

**Objective**: Generate a Decision section based on a given Context section.

**Process**:
- Load test data from `Retrieval/[model]/CDtest.jsonl` (contains Context field)
- For each context, prompt the LLM with a system instruction to generate a coherent decision
- Store results with metadata including:
  - Generated Decision text
  - Model name and parameters
  - Execution time
  - Original Context (for reference)


### Task 2: Title-to-Body (TB)

**Objective**: Generate a complete Body section (including Context and Decision) given only a Title.

**Process**:
- Load test data from `Retrieval/[model]/TBtest.jsonl` (contains Title field)
- For each title, prompt the LLM to generate a full ADR body
- Store results with metadata including:
  - Generated Body text
  - Generated Context (if structured)
  - Generated Decision (if structured)
  - Model name and parameters
  - Execution time


---

## Running an Experiment

### Setup

Configure environment variables in `.env`:
```
GCP_PROJECT_ID=your_project_id      # For Gemini
LOCATION=us-central1
OPENAI_API_KEY=your_api_key         # For OpenAI
```

### Execute

Run any model-task script directly (no arguments needed):

```powershell
python Prompting/Code/gemini-2.5-flash_CD.py
python Prompting/Code/gpt_5_TB.py
python Prompting/Code/qwen3-30b-a3b-instruct-2507_CD.py
```

**Resuming interrupted runs**: Scripts automatically skip already-processed entries; re-run to continue.

---

## Output Format

Results are saved as JSONL with per-entry metadata:

```json
{
  "Model": "gemini-2.5-flash",
  "Task": "CD",
  "Input_ID": 42,
  "Input_Text": "The project requires managing multiple microservices...",
  "Generated_Output": "We decided to adopt a service mesh architecture...",
  "Tokens_Generated": 187,
  "Execution_Time": 2.34,
  "Timestamp": "2026-04-02T14:23:45"
}
```

For Title->Body tasks, `Generated_Output` contains the full ADR body including Context and Decision.
