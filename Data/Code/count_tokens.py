import json

data = ["Data/ADR-data/test.jsonl", "Data/ADR-data/train.jsonl", "Data/ADR-data/val.jsonl"]

all_tokens = 0
for file_path in data:
    # read jsonl file
    with open(file_path, 'r') as f:
        contents = [json.loads(line) for line in f]
    
    tokens = 0
    for item in contents:
        tokens += item['tokenContext'] + item['tokenDecision'] + item['tokenBody']
        
    all_tokens += tokens
    print(f"{file_path}: {tokens} tokens")

print(f"Total tokens across all files: {all_tokens} tokens")