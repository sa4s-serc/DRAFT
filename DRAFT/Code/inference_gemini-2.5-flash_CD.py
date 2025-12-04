import json
import time
from dotenv import load_dotenv
import os
from google import genai
from google.genai import types
from tqdm.auto import tqdm

load_dotenv()
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
LOCATION = os.getenv("LOCATION")
VERTEXAI_API_KEY: str | None = os.getenv("VERTEXAI_API_KEY")
MODEL_ID = "69760714247503872"
TUNED_MODEL_NAME = f"projects/{GCP_PROJECT_ID}/locations/{LOCATION}/endpoints/{MODEL_ID}"

SYSTEM_INSTRUCTION = "You are an expert software architect responsible for maintaining and thoroughly documenting all architectural decisions. You are writing an Architectural Decision Record for a software. Below are a few examples of Context and the corresponding Decision. Following the examples, provide only the ## Decision for the final ## Context provided by the user. Provide only the Decision in about 2-400 words. Do not add any explanations, introductions, or additional responses."

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
    config = types.GenerateContentConfig(
        max_output_tokens=1024,
        # temperature=0.1
    )

    start_time = time.time()
    
    # print("user_prompt_contents:", user_prompt_contents)
    
    while True:
        response = client_obj.models.generate_content(
            model=model_name,
            contents=user_prompt_contents,
            config=config,
        )
        if response.text is not None:
            break
    # print(response)

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

def extract_retrieved_context(entry):
    """Extract retrieved context string from one JSONL entry."""
    contexts = [doc["Context"] for doc in entry["Retrieved"]]
    return contexts

def extract_retrieved_decision(entry):
    """Extract retrieved decision string from one JSONL entry."""
    decisions = [doc["Decision"] for doc in entry["Retrieved"]]
    return decisions


def context_formator(retrieved_contexts, retrieved_decisions, context):
    """
    Formats the context into the list of types.Content objects
    required by the genai client.
    """
    rag_context = "\n\n".join([f"##Context: {c}\n\n##Decision: {d}" for c, d in zip(retrieved_contexts, retrieved_decisions)])
    
    full_prompt = f"{SYSTEM_INSTRUCTION}\n\n{rag_context}\n\n## Context: {context}"
    
    messages = [
        {
            "role": "user",
            "parts": [{"text": full_prompt}],
        }
    ]
    return messages

"""
Load test data and run inference
"""
input_file = "Retrieval/gemini/CDtest.jsonl"
output_file = "DRAFT/Results/gemini-2.5-flash_CDtest.jsonl"
entries = load_jsonl(input_file)

results = []

for i, entry in tqdm(enumerate(entries), total=len(entries)):
    primary_key = extract_primary_key(entry)
    context = extract_context(entry)
    retrieved_contexts = extract_retrieved_context(entry)
    retrieved_decisions = extract_retrieved_decision(entry)
    messages = context_formator(retrieved_contexts, retrieved_decisions, context)

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
