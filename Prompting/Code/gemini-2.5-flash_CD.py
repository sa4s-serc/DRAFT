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

SYSTEM_INSTRUCTION = "You are an expert software architect responsible for maintaining and thoroughly documenting all architectural decisions. You are writing an Architectural Decision Record for a software. Give a ## Decision corresponding to the ## Context provided by the User. Provide only the Decision in about 2-400 words. Do not add any explanations, introductions, or additional responses."

try:
    genai_client = genai.Client(
        vertexai=True, project=GCP_PROJECT_ID, location=LOCATION
    )
    print("--- genai.Client initialized successfully using ADC. ---")
except Exception as e:
    print(f"--- FAILED to initialize genai.Client: {e} ---")
    exit()

def load_jsonl(file_path):
    """Load a JSONL file and return list of dicts."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def save_jsonl(data, file_path):
    """Append list of dicts to a JSONL file."""
    with open(file_path, "a", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def generate_response(user_prompt_contents, client_obj, model_name):
    """
    Generates a response using the genai.Client object.
    """
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

    decision = response.text.strip()
    gen_tokens = response.usage_metadata.candidates_token_count

    return decision, gen_tokens, elapsed

def extract_context(entry):
    """Extract context string from one JSONL entry."""
    return entry["Anchor"]["Context"]

def extract_primary_key(entry):
    """Extract primary key from one JSONL entry."""
    return entry["Anchor"]["PrimaryKey"]

def context_formator(context):
    """
    Formats the context into the list of types.Content objects
    required by the genai client.
    """
    full_prompt = f"{SYSTEM_INSTRUCTION}\n\n## Context: {context}"
    
    messages = [
        {
            "role": "user",
            "parts": [{"text": full_prompt}],
        }
    ]
    return messages

"""
Main processing loop
"""
def main():
    input_file = "Retrieval/gemini/CDtest.jsonl"
    output_file = "Prompting/Results/gemini-2.5-flash_CDtest.jsonl"
    entries = load_jsonl(input_file)

    results = []

    for i, entry in tqdm(enumerate(entries), total=len(entries)):
        primary_key = extract_primary_key(entry)
        context = extract_context(entry)
        user_prompt = context_formator(context)

        try:
            decision, gen_tokens, elapsed = generate_response(
                user_prompt, genai_client, BASE_MODEL
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

    save_jsonl(results, output_file)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()