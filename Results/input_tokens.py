import json
import tiktoken
from pathlib import Path
from tqdm import tqdm

# ---------- CONFIG ----------
INPUT_FOLDERS = ["Retrieval/qwen3-embedding-8B/", "Retrieval/openai/", "Retrieval/gemini/"]
OUTPUT_JSON = "Results/token_counts.json"
MODEL = "gpt-5"  # or another model you care about
# ----------------------------

# Load tokenizer
encoding = tiktoken.encoding_for_model(MODEL)

def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(encoding.encode(text))

results = {}

for folder in INPUT_FOLDERS:
    input_path = Path(folder)
    # Find all JSONL files in the folder
    jsonl_files = list(input_path.glob("*.jsonl"))
    for jsonl_file in jsonl_files:
        INPUT_JSONL = str(jsonl_file)
        jsonl_results = []
        with open(INPUT_JSONL, "r", encoding="utf-8") as f:
            for line_number, line in tqdm(enumerate(f, start=1), desc=f"Processing", total=sum(1 for _ in open(INPUT_JSONL))):
                record = json.loads(line)

                anchor = record.get("Anchor", {})
                retrieved = record.get("Retrieved", [])

                anchor_context = anchor.get("Context", anchor.get("Title", ""))

                # 1. Tokens for Anchor["Context"]
                anchor_context_tokens = count_tokens(anchor_context)

                # 2. Tokens for Anchor["Context"] + all Retrieved Context + Decision
                retrieved_text_parts = []
                for item in retrieved:
                    retrieved_text_parts.append(item.get("Context", item.get("Title", "")))
                    retrieved_text_parts.append(item.get("Decision", item.get("Body", "")))

                combined_text = anchor_context + "\n".join(retrieved_text_parts)
                combined_tokens = count_tokens(combined_text)

                jsonl_results.append({
                    "line_number": line_number,
                    "anchor_primary_key": anchor.get("PrimaryKey"),
                    "prompting_tokens": anchor_context_tokens,
                    "rafg_tokens": combined_tokens
                })
                
        jsonl_analysis = {
            # "file": str(jsonl_file),
            "total_records": len(jsonl_results),
            "rafg": {
                "mean": sum(r["rafg_tokens"] for r in jsonl_results) / len(jsonl_results) if jsonl_results else 0,
                "max": max(r["rafg_tokens"] for r in jsonl_results) if jsonl_results else 0,
                "min": min(r["rafg_tokens"] for r in jsonl_results) if jsonl_results else 0,
                "median": sorted(r["rafg_tokens"] for r in jsonl_results)[len(jsonl_results)//2] if jsonl_results else 0,
            },
            "prompting": {
                "mean": sum(r["prompting_tokens"] for r in jsonl_results) / len(jsonl_results) if jsonl_results else 0,
                "max": max(r["prompting_tokens"] for r in jsonl_results) if jsonl_results else 0,
                "min": min(r["prompting_tokens"] for r in jsonl_results) if jsonl_results else 0,
                "median": sorted(r["prompting_tokens"] for r in jsonl_results)[len(jsonl_results)//2] if jsonl_results else 0,
            }
        }

        results[str(jsonl_file)] = jsonl_analysis

# Write output JSON
with open(OUTPUT_JSON, "w", encoding="utf-8") as out:
    json.dump(results, out, indent=2)

print(f"Wrote token counts for {len(results)} records to {OUTPUT_JSON}")
