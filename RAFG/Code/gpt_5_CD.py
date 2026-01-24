import json
import time
from dotenv import load_dotenv
import os
from openai import OpenAI
from tqdm import tqdm

# ------------------------------------------------------------------------------
# Environment & Client
# ------------------------------------------------------------------------------

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_MODEL = "gpt-5"

SYSTEM_INSTRUCTION = "You are an expert software architect responsible for maintaining and thoroughly documenting all architectural decisions. You are writing an Architectural Decision Record for a software. Below are a few examples of Context and the corresponding Decision. Following the examples, provide only the ## Decision for the final ## Context provided by the user. Provide only the Decision in about 2-400 words. Do not add any explanations, introductions, or additional responses."

try:
    client = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    print(f"--- FAILED: {e} ---")
    exit()

# ------------------------------------------------------------------------------
# JSONL Helpers
# ------------------------------------------------------------------------------

def load_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data, file_path, overwrite=False):
    mode = "w" if overwrite else "a"
    with open(file_path, mode, encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

# ------------------------------------------------------------------------------
# Field Extractors
# ------------------------------------------------------------------------------

def extract_context(entry):
    return entry["Anchor"]["Context"]


def extract_retrieved_context(entry):
    return [doc["Context"] for doc in entry["Retrieved"]]


def extract_retrieved_decision(entry):
    return [doc["Decision"] for doc in entry["Retrieved"]]


def extract_primary_key(entry):
    return entry["Anchor"]["PrimaryKey"]

# ------------------------------------------------------------------------------
# Prompt Formatting
# ------------------------------------------------------------------------------

def context_formatter(retrieved_contexts, retrieved_decisions, context):
    """
    Builds a few-shot conversation for ChatGPT.
    """
    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},

        {"role": "user", "content": f"## Context:\n{retrieved_contexts[0]}"},
        {"role": "assistant", "content": f"## Decision:\n{retrieved_decisions[0]}"},

        {"role": "user", "content": f"## Context:\n{retrieved_contexts[1]}"},
        {"role": "assistant", "content": f"## Decision:\n{retrieved_decisions[1]}"},

        {"role": "user", "content": f"## Context:\n{context}"}
    ]
    return messages

# ------------------------------------------------------------------------------
# Generation
# ------------------------------------------------------------------------------

def generate_response(messages, client_obj, model_name):
    start_time = time.time()

    try:
        response = client_obj.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=1,
            max_completion_tokens=1024,
            reasoning_effort="low"
        )

        if not response.choices:
            print("[!] Empty response from OpenAI")
            return "", 0, 0

        decision = response.choices[0].message.content.strip()
        gen_tokens = response.usage.completion_tokens or 0

    except Exception as e:
        print(f"API Error: {e}")
        return f"API_ERROR: {str(e)}", 0, 0

    elapsed = time.time() - start_time
    return decision, gen_tokens, elapsed

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

input_file = "Retrieval/openai/CDtest.jsonl"
output_file = "RAFG/Results/gpt_5_CDtest.jsonl"

entries = load_jsonl(input_file)
results = []

if os.path.exists(output_file):
    existing_results = load_jsonl(output_file)
    results = existing_results.copy()

    for res in existing_results:
        if res["GeneratedTokens"] == 0:
            entry = next(
                (e for e in entries if extract_primary_key(e) == res["PrimaryKey"]),
                None,
            )
            if entry:
                results.remove(res)
            else:
                print(
                    f"Warning: Could not find entry for PrimaryKey "
                    f"{res['PrimaryKey']} to reprocess."
                )

    processed_keys = {res["PrimaryKey"] for res in results}
    entries = [
        entry for entry in entries
        if extract_primary_key(entry) not in processed_keys
    ]

    print(f"Resuming from existing results. {len(entries)} entries left to process.")

for i, entry in tqdm(enumerate(entries), total=len(entries)):
    try:
        primary_key = extract_primary_key(entry)

        messages = context_formatter(
            extract_retrieved_context(entry),
            extract_retrieved_decision(entry),
            extract_context(entry),
        )

        decision, tokens, elapsed = generate_response(
            messages, client, BASE_MODEL
        )

        results.append({
            "PrimaryKey": primary_key,
            "Decision": decision,
            "GeneratedTokens": tokens,
            "Time": elapsed
        })

        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}")

    except Exception as e:
        print(f"Error {i}: {e}")
        results.append({
            "PrimaryKey": primary_key,
            "Decision": f"ERROR: {e}",
            "GeneratedTokens": 0,
            "Time": 0,
        })

save_jsonl(results, output_file, overwrite=True)
print(f"Results saved to {output_file}")