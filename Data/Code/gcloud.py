import json
import os

# RUN FROM DRAFT DIRECTORY
new_directory = os.getcwd()
print(f"Working directory: {new_directory}")


def load_jsonl(file_path):
    """Load a JSONL file and return list of dicts."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data, file_path, overwrite=False):
    """Save list of dicts to a JSONL file.
    If overwrite is True, replace the file. Otherwise append to it.
    """
    # ensure target directory exists
    dirpath = os.path.dirname(file_path)
    if dirpath and not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)

    mode = "w" if overwrite else "a"
    with open(file_path, mode, encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# if ADR-data/google is not a directory, create it
if not os.path.exists("Retrieval/google"):
    os.makedirs("Retrieval/google")


def format_example_cd(example, test=False):
    context = example["Anchor"]["Context"]
    decision = example["Anchor"]["Decision"]

    messages = {
        "system_instruction": {
            "role": "model",
            "parts": [
                {
                    "text": "You are an expert software architect responsible for maintaining and thoroughly documenting all architectural decisions. You are writing an Architectural Decision Record for a software. Give a ## Decision corresponding to the ## Context provided by the User. Provide only the Decision in about 2-400 words. Do not add any explanations, introductions, or additional responses."
                }
            ],
        },
        "contents": [
            {"role": "user", "parts": [{"text": f"## Context: {context}"}]},
            {"role": "model", "parts": [{"text": f"## Decision: {decision}"}]},
        ],
    }
    if test:
        return {"request": messages}
    return messages


def format_example_tb(example, test=False):
    title = example["Anchor"]["Title"]
    body = example["Anchor"]["Body"]

    messages = {
        "system_instruction": {
            "role": "system",
            "parts": [
                {
                    "text": "You are an expert software architect responsible for maintaining and thoroughly documenting all architectural decisions. You are writing an Architectural Decision Record for a software. Write the ADR corresponding to the ADR Title provided by the User. Provide only the ADR content in about 10-800 words. Do not add any additional responses—only the ADR content."
                }
            ],
        },
        "contents": [
            {"role": "user", "parts": [{"text": f"# Title: {title}"}]},
            {"role": "model", "parts": [{"text": body}]},
        ],
    }
    if test:
        return {"request": messages}
    return messages


def format_example_cd_draft(example, test=False):
    retrieved_contexts = [doc["Context"] for doc in example["Retrieved"]]
    retrieved_decisions = [doc["Decision"] for doc in example["Retrieved"]]
    context = example["Anchor"]["Context"]
    decision = example["Anchor"]["Decision"]

    messages = {
        "system_instruction": {
            "role": "system",
            "parts": [
                {
                    "text": "You are an expert software architect responsible for maintaining and thoroughly documenting all architectural decisions. You are writing an Architectural Decision Record for a software. Below are a few examples of Context and the corresponding Decision. Following the examples, provide only the ## Decision for the final ## Context provided by the user. Provide only the Decision in about 2-400 words. Do not add any explanations, introductions, or additional responses."
                }
            ],
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": f"## Context: {retrieved_contexts[0]}"}],
            },
            {
                "role": "model",
                "parts": [{"text": f"## Decision: {retrieved_decisions[0]}"}],
            },
            {
                "role": "user",
                "parts": [{"text": f"## Context: {retrieved_contexts[1]}"}],
            },
            {
                "role": "model",
                "parts": [{"text": f"## Decision: {retrieved_decisions[1]}"}],
            },
            {"role": "user", "parts": [{"text": f"## Context: {context}"}]},
            {"role": "model", "parts": [{"text": f"## Decision: {decision}"}]},
        ],
    }
    if test:
        return {"request": messages}
    return messages


def format_example_tb_draft(example, test=False):
    retrieved_title = [doc["Title"] for doc in example["Retrieved"]]
    retrieved_body = [doc["Body"] for doc in example["Retrieved"]]
    title = example["Anchor"]["Title"]
    body = example["Anchor"]["Body"]

    title = example["Anchor"]["Title"]
    body = example["Anchor"]["Body"]

    messages = {
        "system_instruction": {
            "role": "system",
            "parts": [
                {
                    "text": "You are an expert software architect responsible for maintaining and thoroughly documenting all architectural decisions. You are writing an Architectural Decision Record for a software. Write the ADR corresponding to the ADR Title provided by the User. Provide only the ADR content in about 10-800 words. Do not add any additional responses—only the ADR content."
                }
            ],
        },
        "contents": [
            {"role": "user", "parts": [{"text": f"# Title: {retrieved_title[0]}"}]},
            {"role": "assistant", "parts": [{"text": retrieved_body[0]}]},
            {"role": "user", "parts": [{"text": f"# Title: {retrieved_title[1]}"}]},
            {"role": "assistant", "parts": [{"text": retrieved_body[1]}]},
            {"role": "user", "parts": [{"text": f"# Title: {title}"}]},
            {"role": "assistant", "parts": [{"text": body}]},
        ],
    }
    if test:
        return {"request": messages}
    return messages


cds = [
    "Retrieval/gemini/CDtrain.jsonl",
    "Retrieval/gemini/CDval.jsonl",
    "Retrieval/gemini/CDtest.jsonl",
]
tbs = [
    "Retrieval/gemini/TBtrain.jsonl",
    "Retrieval/gemini/TBval.jsonl",
    "Retrieval/gemini/TBtest.jsonl",
]

for cd_file in cds:
    data = load_jsonl(cd_file)
    formatted_data = [
        format_example_cd(example, test="test" in cd_file) for example in data
    ]
    output_file = os.path.join("Retrieval/gcp-ft", os.path.basename(cd_file))
    save_jsonl(formatted_data, output_file, overwrite=True)
    
    formatted_data = [
        format_example_cd_draft(example, test="test" in cd_file) for example in data
    ]
    output_file = os.path.join("Retrieval/gcp", os.path.basename(cd_file))
    save_jsonl(formatted_data, output_file, overwrite=True)

for tb_file in tbs:
    data = load_jsonl(tb_file)
    
    formatted_data = [
        format_example_tb(example, test="test" in tb_file) for example in data
    ]
    output_file = os.path.join("Retrieval/gcp-ft", os.path.basename(tb_file))
    save_jsonl(formatted_data, output_file, overwrite=True)
    
    formatted_data = [
        format_example_tb_draft(example, test="test" in tb_file) for example in data
    ]
    output_file = os.path.join("Retrieval/gcp", os.path.basename(tb_file))
    save_jsonl(formatted_data, output_file, overwrite=True)