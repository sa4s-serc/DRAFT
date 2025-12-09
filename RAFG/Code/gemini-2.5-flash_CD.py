import json
import time
from dotenv import load_dotenv
import os
from google import genai
from google.genai import types
from tqdm import tqdm

load_dotenv()
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
LOCATION = os.getenv("LOCATION")
BASE_MODEL = "gemini-2.5-flash"

SYSTEM_INSTRUCTION = "You are an expert software architect responsible for maintaining and thoroughly documenting all architectural decisions. You are writing an Architectural Decision Record for a software. Below are a few examples of Context and the corresponding Decision. Following the examples, provide only the ## Decision for the final ## Context provided by the user. Provide only the Decision in about 2-400 words. Do not add any explanations, introductions, or additional responses."

try:
    genai_client = genai.Client(
        vertexai=True, project=GCP_PROJECT_ID, location=LOCATION
    )
except Exception as e:
    print(f"--- FAILED: {e} ---")
    exit()

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

def extract_context(entry):
    return entry["Anchor"]["Context"]

def extract_retrieved_context(entry):
    return [doc["Context"] for doc in entry["Retrieved"]]

def extract_retrieved_decision(entry):
    return [doc["Decision"] for doc in entry["Retrieved"]]

def extract_primary_key(entry):
    return entry["Anchor"]["PrimaryKey"]

def context_formator(retrieved_contexts, retrieved_decisions, context):
    # Construct few-shot history for Gemini
    messages = [
        {"role": "user", "parts": [{"text": SYSTEM_INSTRUCTION + f"\n\n## Context: {retrieved_contexts[0]}"}]},
        {"role": "model", "parts": [{"text": f"## Decision: {retrieved_decisions[0]}"}]},
        {"role": "user", "parts": [{"text": f"## Context: {retrieved_contexts[1]}"}]},
        {"role": "model", "parts": [{"text": f"## Decision: {retrieved_decisions[1]}"}]},
        {"role": "user", "parts": [{"text": f"## Context: {context}"}]}
    ]
    return messages

def generate_response(user_prompt_contents, client_obj, model_name):
    """
    Generates a response using the genai.Client object.
    """
    safety_settings = [
        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
    ]

    config = types.GenerateContentConfig(
        max_output_tokens=1024,
        safety_settings=safety_settings,
        system_instruction=SYSTEM_INSTRUCTION,
        temperature=1
    )

    start_time = time.time()
    
    # 2. Remove the infinite loop. If it blocks once, it will block again.
    try:
        response = client_obj.models.generate_content(
            model=model_name,
            contents=user_prompt_contents, # Now a simple string
            config=config,
        )
        
        # 3. Check if candidates exist before accessing text
        if not response.candidates:
            
            # Check Prompt Feedback (Did the INPUT trigger a block?)
            if response.prompt_feedback:
                reason = f"PROMPT_BLOCKED: {response.prompt_feedback.block_reason}"
            
            # Check Usage Metadata (Did it run but get wiped?)
            elif response.usage_metadata and response.usage_metadata.total_token_count > 0:
                reason = "RECITATION_OR_OTHER (Response wiped by filter)"
            
            else:
                reason = "UNKNOWN_FAILURE (Empty response)"

            print(f"\n[!] Blocked Entry {i} (Key: {primary_key}) -> {reason}")

        decision = response.text.strip()
        # Use total_token_count as fallback if candidates_token_count is missing
        gen_tokens = response.usage_metadata.candidates_token_count or 0
        
    except Exception as e:
        print(f"API Error: {e}")
        return f"API_ERROR: {str(e)}", 0, 0

    end_time = time.time()
    elapsed = end_time - start_time

    return decision, gen_tokens, elapsed

input_file = "Retrieval/gemini/CDtest.jsonl"
output_file = "RAFG/Results/gemini-2.5-flash_CDtest.jsonl"
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
    try:
        primary_key = extract_primary_key(entry)
        msgs = context_formator(
            extract_retrieved_context(entry), 
            extract_retrieved_decision(entry), 
            extract_context(entry)
        )
        decision, tokens, elapsed = generate_response(msgs, genai_client, BASE_MODEL)
        results.append({
            "PrimaryKey": primary_key,
            "Decision": decision,
            "GeneratedTokens": tokens,
            "Time": elapsed
        })
        if (i+1) % 50 == 0: print(f"Processed {i+1}")
    except Exception as e:
        print(f"Error {i}: {e}")
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