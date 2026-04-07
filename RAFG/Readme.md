# RAFG — Retrieval-Augmented Few-Shot Generation

This directory implements RAFG: combining retrieval of few-shot examples with LLM generation to produce Architectural Decision Records (ADRs). It integrates outputs from the `Retrieval/` component to evaluate multiple LLMs on two tasks: Context->Decision (CD) and Title->Body (TB).

Rather than relying on a static, predefined set of few-shot examples, RAFG retrieves contextually similar examples from a vector database (VDB) built from historical ADRs. For each input (a Context or a Title), the top-N most similar ADR pairs are retrieved using embedding-based similarity search and used to construct a few-shot prompt. This prompt is then passed to the LLM to generate the output. The retrieval step is pre-computed and stored in the `Retrieval/` directory; the scripts here consume those results and handle prompt construction and generation.

---

## Supported Models

1. Gemini 2.5 Flash
2. GPT-5
3. Qwen3 30B (A3B Instruct)
4. Gemma 3 4B (Instruct)

---

## Directory Structure

```
RAFG/
├── Code/                                              # Model-specific generation scripts
│   ├── gemini-2.5-flash_CD.py                         # Context->Decision generation with Gemini
│   ├── gemini-2.5-flash_TB.py                         # Title->Body generation with Gemini
│   ├── ...
│
├── Results/                                           # Generated outputs in jsonl format
│   ├── gemini-2.5-flash_CDtest.jsonl                  # Outputs generated with gemini-2.5-flash for Context->Decision on test set
│   ├── gemini-2.5-flash_TBtest.jsonl                  # Outputs generated with gemini-2.5-flash for Title->Body on test set
│   ├── ...
│
└── Readme.md
```

---

## Workflow

### Task 1: Context-to-Decision (CD)

**Objective**: Generate a Decision section given a Context and retrieved few-shot examples.

**Process**:
- Load retrieval output from `Retrieval/[model]/CDtest.jsonl` (contains Context + top-N similar Context–Decision pairs)
- For each entry, construct a few-shot prompt using the retrieved Context–Decision pairs as examples
- Prompt the LLM to generate a Decision consistent with the anchor Context
- Store results with metadata including:
  - Generated Decision text
  - Model name and parameters
  - Execution time
  - Original Context (for reference)


### Task 2: Title-to-Body (TB)

**Objective**: Generate a complete ADR Body section given a Title and retrieved few-shot examples.

**Process**:
- Load retrieval output from `Retrieval/[model]/TBtest.jsonl` (contains Title + top-N similar Title–Body pairs)
- For each entry, construct a few-shot prompt using the retrieved Title–Body pairs as examples
- Prompt the LLM to generate a full ADR body for the anchor Title
- Store results with metadata including:
  - Generated Body text
  - Model name and parameters
  - Execution time


---

## Running an Experiment

### Setup

1. Ensure retrieval outputs exist:
   - `Retrieval/[model]/CDtest.jsonl` for Context->Decision
   - `Retrieval/[model]/TBtest.jsonl` for Title->Body

2. Configure environment variables in `.env`:
   ```
   GCP_PROJECT_ID=your_project_id      # For Gemini
   LOCATION=us-central1
   OPENAI_API_KEY=your_api_key         # For OpenAI
   ```

### Execute

Run any model-task script directly (no arguments needed):

```powershell
python RAFG/Code/gemini-2.5-flash_CD.py
python RAFG/Code/gpt_5_TB.py
python RAFG/Code/qwen3-30b-a3b-instruct-2507_CD.py
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
