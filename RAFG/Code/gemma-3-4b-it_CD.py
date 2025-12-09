from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
import os
import torch
import json
import time

load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_name = "google/gemma-3-4b-it"
# Adjust cache_dir as needed per environment
cache_dir = "/research/ug/ug2k21dual/csd/adyansh.kakran/DRAFT/cache" 
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, token=HUGGINGFACE_TOKEN)
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, token=HUGGINGFACE_TOKEN, device_map="auto")

model.to(device)
model.eval()

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

def generate_response(model, tokenizer, messages, device, max_new_tokens=500):
    formatted_chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted_chat, return_tensors="pt").to(device)
    input_length = inputs["input_ids"].shape[1]
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids = outputs[0][input_length:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response, generated_ids.shape[0]

def extract_context(entry):
    return entry["Anchor"]["Context"]

def extract_retrieved_context(entry):
    return [doc["Context"] for doc in entry["Retrieved"]]

def extract_retrieved_decision(entry):
    return [doc["Decision"] for doc in entry["Retrieved"]]

def extract_primary_key(entry):
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

input_file = "Retrieval/qwen3-embedding-8B/CDtest.jsonl"
output_file = "RAFG/Results/gemma-3-4b-it_CDtest.jsonl"
entries = load_jsonl(input_file)
results = []

for i, entry in enumerate(entries):
    primary_key = extract_primary_key(entry)
    context = extract_context(entry)
    retrieved_contexts = extract_retrieved_context(entry)
    retrieved_decisions = extract_retrieved_decision(entry)
    
    messages = context_formator(retrieved_contexts, retrieved_decisions, context)
    
    start_time = time.time()
    response, gen_tokens = generate_response(model, tokenizer, messages, device)
    elapsed = time.time() - start_time
    
    results.append({
        "PrimaryKey": primary_key,
        "Decision": response,
        "GeneratedTokens": gen_tokens,
        "Time": elapsed
    })
    
    if (i + 1) % 100 == 0: print(f"Processed {i + 1}")

save_jsonl(results, output_file)
print(f"Results saved to {output_file}")