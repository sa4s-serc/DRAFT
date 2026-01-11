from unsloth import FastLanguageModel
import torch
import json
import time
from tqdm.auto import tqdm
from dotenv import load_dotenv
import os

model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"
cache_dir = "/research/ug/ug2k21dual/csd/adyansh.kakran/DRAFT/cache"
output_dir = "DRAFT/Output/"

load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE")

checkpoint_path = os.getenv("CHECKPOINT_PATH", "Finetune/Output/checkpoint-760")
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

model, tokenizer = FastLanguageModel.from_pretrained(
    checkpoint_path,  # local checkpoint directory
    cache_dir=cache_dir,
    token=HUGGINGFACE_TOKEN,
    max_seq_length=1024,
    dtype=torch.bfloat16,
    # load_in_4bit=True,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Move the model to the chosen device
model.to(device)

# Set the model to evaluation mode
model.eval()


"""
Helper functions
"""
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
        
    

def generate_response(model, tokenizer, messages, device):
    """Generate model response given messages."""
    formatted_chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Tokenize
    inputs = tokenizer(formatted_chat, return_tensors="pt").to(device)
    input_length = inputs["input_ids"].shape[1]

    # max_new_tokens=500 for Context-Decision; and 1000 for Title-Body
    outputs = model.generate(**inputs, max_new_tokens=1000)

    generated_ids = outputs[0][input_length:]  # slice only new tokens
    response = tokenizer.decode(generated_ids, skip_special_tokens=True) # Decode only the generated text
    generated_tokens = generated_ids.shape[0] # Number of generated tokens
    return response, generated_tokens


def extract_title(entry):
    """Extract title from one JSONL entry."""
    return entry["Anchor"]["Title"]

def extract_primary_key(entry):
    """Extract primary key from one JSONL entry."""
    return entry["Anchor"]["PrimaryKey"]

def title_formator(title):
    messages = [
        {"role": "system", "content": "You are an expert software architect responsible for maintaining and thoroughly documenting all architectural decisions. You are writing an Architectural Decision Record for a software. Write the ADR corresponding to the ADR Title provided by the User. Provide only the ADR content in about 10-800 words. Do not add any additional responses—only the ADR content."},
        {"role": "user", "content": f"# {title}"}
    ]
    return messages


"""
Load test data and run inference
"""
input_file = "Retrieval/qwen3-embedding-8B/TBtest.jsonl"
output_file = "Finetune/Results/qwen3-30b-a3b-instruct-TBtest.jsonl"
entries = load_jsonl(input_file)

results = []

# Iterate over entries
for i, entry in tqdm(enumerate(entries), total=len(entries)): # limit to first 3 for demo
    primary_key = extract_primary_key(entry)
    title = extract_title(entry)
    messages = title_formator(title)

    start_time = time.time()
    response, gen_tokens = generate_response(model, tokenizer, messages, device)
    elapsed = time.time() - start_time


    result = {
        "PrimaryKey": primary_key,
        "Body": response,
        "GeneratedTokens": gen_tokens,
        "Time": elapsed
    }
    results.append(result)

# Save all results to output JSONL
save_jsonl(results, output_file)
print(f"Results saved to {output_file}")
