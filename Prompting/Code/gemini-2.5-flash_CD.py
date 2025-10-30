from dotenv import load_dotenv
import os

load_dotenv()
VERTEXAI_API_KEY = os.getenv("VERTEXAI_API_KEY")

"""
Define helper functions
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
        
    

def generate_response(model, tokenizer, messages, device, max_new_tokens=500):
    """Generate model response given messages."""
    formatted_chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Tokenize
    inputs = tokenizer(formatted_chat, return_tensors="pt").to(device)
    input_length = inputs["input_ids"].shape[1]
    # Generate
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids = outputs[0][input_length:]  # slice only new tokens
    response = tokenizer.decode(generated_ids, skip_special_tokens=True) # Decode only the generated text
    generated_tokens = generated_ids.shape[0] # Number of generated tokens
    return response, generated_tokens


def extract_context(entry):
    """Extract context string from one JSONL entry."""
    return entry["Anchor"]["Context"]

def extract_primary_key(entry):
    """Extract primary key from one JSONL entry."""
    return entry["Anchor"]["PrimaryKey"]

def context_formator(context):
    messages = [
        {
            "role": "system", 
            "content": "You are an expert software architect responsible for maintaining and thoroughly documenting all architectural decisions. You are writing an Architectural Decision Record for a software. Give a ## Decision corresponding to the ## Context provided by the User. Provide only the Decision in about 2-400 words. Do not add any explanations, introductions, or additional responses."
        },
        {
            "role": "user",
            "content": f"## Context: {context}"
        }
    ]
    return messages