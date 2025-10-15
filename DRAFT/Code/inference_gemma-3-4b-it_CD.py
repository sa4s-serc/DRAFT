from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import time


checkpoint_dir = "DRAFT/Output/checkpoint-380"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
model = AutoModelForCausalLM.from_pretrained(
    checkpoint_dir,
    device_map="auto",
    dtype=torch.bfloat16
)

# Check for GPU availability
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
    outputs = model.generate(**inputs, max_new_tokens=500)

    generated_ids = outputs[0][input_length:]  # slice only new tokens
    response = tokenizer.decode(generated_ids, skip_special_tokens=True) # Decode only the generated text
    generated_tokens = generated_ids.shape[0] # Number of generated tokens
    return response, generated_tokens


def extract_context(entry):
    """Extract context string from one JSONL entry."""
    return entry["Anchor"]["Context"]

def extract_retrieved_context(entry):
    """Extract retrieved context string from one JSONL entry."""
    contexts = [doc["Context"] for doc in entry["Retrieved"]]
    return contexts

def extract_retrieved_decision(entry):
    """Extract retrieved decision string from one JSONL entry."""
    decisions = [doc["Decision"] for doc in entry["Retrieved"]]
    return decisions

def extract_primary_key(entry):
    """Extract primary key from one JSONL entry."""
    return entry["Anchor"]["PrimaryKey"]

def context_formator(retrieved_contexts, retrieved_decisions, context):
    messages = [
        {"role": "system", "content": "You are an expert software architect responsible for maintaining and thoroughly documenting all architectural decisions. You are writing an Architectural Decision Record for a software. Below are a few examples of Context and the corresponding Decision. Following the examples, provide only the ## Decision for the final ## Context provided by the user. Provide only the Decision in about 2-400 words. Do not add any explanations, introductions, or additional responses."},
        {"role": "user", "content": f"## Context: {retrieved_contexts[0]}"},
        {"role": "assistant", "content": f"## Decision: {retrieved_decisions[0]}"},
        {"role": "user", "content": f"## Context: {retrieved_contexts[1]}"},
        {"role": "assistant", "content": f"## Decision: {retrieved_decisions[1]}"},
        {"role": "user", "content": f"## Context: {context}"},
    ]
    return messages


"""
Load test data and run inference
"""
input_file = "Retrieval/CDtest.jsonl"
output_file = "DRAFT/Results/gemma-3-4b-it-CDtest-results.jsonl"
entries = load_jsonl(input_file)

results = []

# Iterate over entries
for i, entry in enumerate(entries): # limit to first 3 for demo
    primary_key = extract_primary_key(entry)
    context = extract_context(entry)
    retrieved_contexts = extract_retrieved_context(entry)
    retrieved_decisions = extract_retrieved_decision(entry)
    messages = context_formator(retrieved_contexts, retrieved_decisions, context)

    start_time = time.time()
    response, gen_tokens = generate_response(model, tokenizer, messages, device)
    elapsed = time.time() - start_time


    result = {
    "PrimaryKey": primary_key,
    "Decision": response,
    "GeneratedTokens": gen_tokens,
    "Time": elapsed
    }
    results.append(result)

# Save all results to output JSONL
save_jsonl(results, output_file)
print(f"Results saved to {output_file}")