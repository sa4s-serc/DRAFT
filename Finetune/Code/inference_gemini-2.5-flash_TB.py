import json
import time
from dotenv import load_dotenv
import os
from google import genai
from google.genai import types

load_dotenv()
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
LOCATION = os.getenv("LOCATION")
# Update MODEL_ID with the specific Endpoint ID obtained after training
MODEL_ID = "YOUR_TB_TUNED_MODEL_ENDPOINT_ID" 
TUNED_MODEL_NAME = f"projects/{GCP_PROJECT_ID}/locations/{LOCATION}/endpoints/{MODEL_ID}"

SYSTEM_INSTRUCTION = "You are an expert software architect responsible for maintaining and thoroughly documenting all architectural decisions. You are writing an Architectural Decision Record for a software. Write the ADR corresponding to the ADR Title provided by the User. Provide only the ADR content in about 10-800 words. Do not add any additional responses—only the ADR content."

try:
    genai_client = genai.Client(
        vertexai=True, project=GCP_PROJECT_ID, location=LOCATION
    )
    print("--- genai.Client initialized successfully using ADC. ---")
except Exception as e:
    print(f"--- FAILED to initialize genai.Client: {e} ---")
    exit()

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

def generate_response(user_prompt_contents, client_obj, model_name):
    config = types.GenerateContentConfig(
        max_output_tokens=1024,
        temperature=0.1
    )
    start_time = time.time()
    response = client_obj.models.generate_content(
        model=model_name,
        contents=user_prompt_contents,
        config=config,
    )
    end_time = time.time()
    elapsed = end_time - start_time
    
    body = response.text.strip()
    gen_tokens = response.usage_metadata.candidates_token_count
    return body, gen_tokens, elapsed

def extract_title(entry):
    return entry["Anchor"]["Title"]

def extract_primary_key(entry):
    return entry["Anchor"]["PrimaryKey"]

def title_formator(title):
    full_prompt = f"{SYSTEM_INSTRUCTION}\n\n# Title: {title}"
    messages = [
        {
            "role": "user",
            "parts": [{"text": full_prompt}],
        }
    ]
    return messages

input_file = "Retrieval/TBtest.jsonl"
output_file = "Finetune/Results/gemini-2.5-flash_TBtest.jsonl"
entries = load_jsonl(input_file)
results = []

for i, entry in enumerate(entries):
    primary_key = extract_primary_key(entry)
    title = extract_title(entry)
    user_prompt = title_formator(title)

    try:
        body, gen_tokens, elapsed = generate_response(
            user_prompt, genai_client, TUNED_MODEL_NAME
        )
        result = {
            "PrimaryKey": primary_key,
            "Body": body,
            "GeneratedTokens": gen_tokens,
            "Time": elapsed,
        }
        results.append(result)
    except Exception as e:
        print(f"Error processing entry {i}: {e}")

save_jsonl(results, output_file, overwrite=True)
print(f"Results saved to {output_file}")