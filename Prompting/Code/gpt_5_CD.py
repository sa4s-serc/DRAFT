import json
import time
from dotenv import load_dotenv
import os
from openai import OpenAI
from tqdm import tqdm

# ------------------------------------------------------------------------------
# Environment & Client Setup
# ------------------------------------------------------------------------------

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_MODEL = "gpt-5" 

SYSTEM_INSTRUCTION = "You are an expert software architect responsible for maintaining and thoroughly documenting all architectural decisions. You are writing an Architectural Decision Record for a software. Give a ## Decision corresponding to the ## Context provided by the User. Provide only the Decision in about 2-400 words. Do not add any explanations, introductions, or additional responses."


try:
    client = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    print(f"--- FAILED to initialize OpenAI client: {e} ---")
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
    """Append list of dicts to a JSONL file."""
    mode = "w" if overwrite else "a"
    with open(file_path, mode, encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

# ------------------------------------------------------------------------------
# OpenAI Response Generation
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

        body = response.choices[0].message.content.strip()
        gen_tokens = response.usage.completion_tokens or 0

    except Exception as e:
        print(f"API Error: {e}")
        return f"API_ERROR: {str(e)}", 0, 0

    elapsed = time.time() - start_time
    return body, gen_tokens, elapsed

# ------------------------------------------------------------------------------
# Entry Helpers
# ------------------------------------------------------------------------------

def extract_context(entry):
    """Extract context string from one JSONL entry."""
    return entry["Anchor"]["Context"]

def extract_primary_key(entry):
    """Extract primary key from one JSONL entry."""
    return entry["Anchor"]["PrimaryKey"]

def context_formator(context):
    messages = [
        { "role": "system", "content": SYSTEM_INSTRUCTION },
        { "role": "user", "content": f"## Context: {context}" }
    ]
    return messages

# ------------------------------------------------------------------------------
# Main Processing Loop
# ------------------------------------------------------------------------------

input_file = "Retrieval/gemini/CDtest.jsonl"
output_file = "Prompting/Results/gpt_5_CDtest.jsonl"
entries = load_jsonl(input_file)
results = []

if os.path.exists(output_file):
    existing_results = load_jsonl(output_file)
    results = existing_results.copy()
    for res in existing_results:
        if res["GeneratedTokens"] == 0:
            entry = next((e for e in entries if extract_primary_key(e) == res["PrimaryKey"]), None)
            if entry:
                results.remove(res)
            else:
                print(f"Warning: Could not find entry for PrimaryKey {res['PrimaryKey']} to reprocess.")
    processed_keys = {res["PrimaryKey"] for res in results}
    entries = [entry for entry in entries if extract_primary_key(entry) not in processed_keys]
    print(f"Resuming from existing results. {len(entries)} entries left to process.")


for i, entry in tqdm(enumerate(entries), total=len(entries)):
    primary_key = extract_primary_key(entry)
    context = extract_context(entry)
    user_prompt = context_formator(context)

    try:
        decision, gen_tokens, elapsed = generate_response(
            user_prompt, client, BASE_MODEL
        )

        result = {
            "PrimaryKey": primary_key,
            "Decision": decision,
            "GeneratedTokens": gen_tokens,
            "Time": elapsed,
        }
        results.append(result)
        
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1} entries")

    except Exception as e:
        print(f"Error processing entry {i} (PrimaryKey: {primary_key}): {e}")
        results.append(
            {
                "PrimaryKey": primary_key,
                "Decision": f"ERROR: {e}",
                "GeneratedTokens": 0,
                "Time": 0,
            }
        )

save_jsonl(results, output_file, overwrite=True)
print(f"Results saved to {output_file}")
