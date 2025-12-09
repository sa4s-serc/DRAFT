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
VERTEXAI_API_KEY: str | None = os.getenv("VERTEXAI_API_KEY")
MODEL_ID = "100265564848521216"
TUNED_MODEL_NAME = f"projects/{GCP_PROJECT_ID}/locations/{LOCATION}/endpoints/{MODEL_ID}"

SYSTEM_INSTRUCTION = "You are an expert software architect responsible for maintaining and thoroughly documenting all architectural decisions. You are writing an Architectural Decision Record for a software. Write the ADR corresponding to the ADR Title provided by the User. Below are a few examples of Title and the corresponding ADR. Provide only the ADR content in about 10-800 words. Do not add any additional responses—only the ADR content."

try:
    genai_client = genai.Client(
        vertexai=True, project=GCP_PROJECT_ID, location=LOCATION
    )
    print("--- genai.Client initialized successfully using ADC. ---")
except Exception as e:
    print(f"--- FAILED to initialize genai.Client: {e} ---")
    print(
        "Please ensure you are authenticated (e.g., via 'gcloud auth application-default login')"
    )
    exit()


# def test_model_connection(client_obj, model_name):
#     """Sends a simple test prompt to see if the model connection is valid."""
#     print("--- RUNNING CONNECTION TEST ---")
#     try:
#         test_prompt = "Hello, who are you?"

#         response = client_obj.models.generate_content(
#             model=model_name, contents=test_prompt
#         )
#         print("--- TEST SUCCESSFUL ---")
#         print(f"Test Response: {response.text}\n")
#         return True
#     except Exception as e:
#         print(f"--- TEST FAILED: {e} ---")
#         return False

# if not test_model_connection(genai_client, TUNED_MODEL_NAME):
#     print("Exiting due to test failure. Check location, model ID, or ADC permissions.")
#     exit()


def load_jsonl(file_path):
    """Load a JSONL file and return list of dicts."""
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
        temperature=0.1
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

def extract_title(entry):
    """Extract title from one JSONL entry."""
    return entry["Anchor"]["Title"]

def extract_retrieved_title(entry):
    """Extract retrieved title from one JSONL entry."""
    titles = [doc["Title"] for doc in entry["Retrieved"]]
    return titles

def extract_retrieved_body(entry):
    """Extract retrieved body from one JSONL entry."""
    bodies = [doc["Body"] for doc in entry["Retrieved"]]
    return bodies

def extract_primary_key(entry):
    """Extract primary key from one JSONL entry."""
    return entry["Anchor"]["PrimaryKey"]


def title_formator(retrieved_titles, retrieved_bodies, title):
    """
    Formats the context into the list of types.Content objects
    required by the genai client.
    """
    rag_context = "\n\n".join([f"# {t}\n{b}" for t, b in zip(retrieved_titles, retrieved_bodies)])

    full_prompt = f"{rag_context}\n\n# {title}"
    
    # messages = [
    #     {
    #         "role": "user",
    #         "parts": [{"text": full_prompt}],
    #     }
    # ]
    return full_prompt

"""
Load test data and run inference
"""
input_file = "Retrieval/gemini/TBtest.jsonl"
output_file = "DRAFT/Results/gemini-2.5-flash_TBtest.jsonl"
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
    title = extract_title(entry)
    retrieved_title = extract_retrieved_title(entry)
    retrieved_body = extract_retrieved_body(entry)
    messages = title_formator(retrieved_title, retrieved_body, title)

    try:
        decision, gen_tokens, elapsed = generate_response(
            messages, genai_client, TUNED_MODEL_NAME
        )

        # print(f"Processed entry {i}: PrimaryKey={primary_key}")

        # Store result
        result = {
            "PrimaryKey": primary_key,
            "Decision": decision,
            "GeneratedTokens": gen_tokens,
            "Time": elapsed,
        }
        results.append(result)

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
