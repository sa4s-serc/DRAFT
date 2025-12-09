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

def save_jsonl(data, file_path):
    with open(file_path, "a", encoding="utf-8") as f:
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

def generate_response(messages, client_obj, model_name):
    config = types.GenerateContentConfig(max_output_tokens=1024, temperature=0.1)
    start_time = time.time()
    response = client_obj.models.generate_content(
        model=model_name, contents=messages, config=config
    )
    end_time = time.time()
    return response.text.strip(), response.usage_metadata.candidates_token_count, end_time - start_time

input_file = "Retrieval/gemini/CDtest.jsonl"
output_file = "RAFG/Results/gemini-2.5-flash_CDtest.jsonl"
entries = load_jsonl(input_file)
results = []

for i, entry in tqdm(enumerate(entries), total=len(entries)):
    try:
        msgs = context_formator(
            extract_retrieved_context(entry), 
            extract_retrieved_decision(entry), 
            extract_context(entry)
        )
        decision, tokens, elapsed = generate_response(msgs, genai_client, BASE_MODEL)
        results.append({
            "PrimaryKey": extract_primary_key(entry),
            "Decision": decision,
            "GeneratedTokens": tokens,
            "Time": elapsed
        })
        if (i+1) % 50 == 0: print(f"Processed {i+1}")
    except Exception as e:
        print(f"Error {i}: {e}")

save_jsonl(results, output_file)
print(f"Results saved to {output_file}")