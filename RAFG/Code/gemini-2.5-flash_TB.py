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

SYSTEM_INSTRUCTION = "You are an expert software architect responsible for maintaining and thoroughly documenting all architectural decisions. You are writing an Architectural Decision Record for a software. Below are a few examples of Title and the corresponding Body of an ADR. Following the examples, provide only the Body for the final # Title provided by the user. Provide only the ADR content in about 10-800 words. Do not add any additional responses—only the ADR content."

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

def extract_title(entry):
    return entry["Anchor"]["Title"]

def extract_retrieved_titles(entry):
    return [doc["Title"] for doc in entry["Retrieved"]]

def extract_retrieved_body(entry):
    return [doc["Body"] for doc in entry["Retrieved"]]

def extract_primary_key(entry):
    return entry["Anchor"]["PrimaryKey"]

def title_formator(retrieved_titles, retrieved_bodies, title):
    messages = [
        {"role": "user", "parts": [{"text": SYSTEM_INSTRUCTION + f"\n\n# {retrieved_titles[0]}"}]},
        {"role": "model", "parts": [{"text": retrieved_bodies[0]}]},
        {"role": "user", "parts": [{"text": f"# {retrieved_titles[1]}"}]},
        {"role": "model", "parts": [{"text": retrieved_bodies[1]}]},
        {"role": "user", "parts": [{"text": f"# {title}"}]}
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

input_file = "Retrieval/gemini/TBtest.jsonl"
output_file = "RAFG/Results/gemini-2.5-flash_TBtest.jsonl"
entries = load_jsonl(input_file)
results = []

for i, entry in tqdm(enumerate(entries), total=len(entries)):
    try:
        msgs = title_formator(
            extract_retrieved_titles(entry), 
            extract_retrieved_body(entry), 
            extract_title(entry)
        )
        body, tokens, elapsed = generate_response(msgs, genai_client, BASE_MODEL)
        results.append({
            "PrimaryKey": extract_primary_key(entry),
            "Body": body,
            "GeneratedTokens": tokens,
            "Time": elapsed
        })
        if (i+1) % 50 == 0: print(f"Processed {i+1}")
    except Exception as e:
        print(f"Error {i}: {e}")

save_jsonl(results, output_file)
print(f"Results saved to {output_file}")