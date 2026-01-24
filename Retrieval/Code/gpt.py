import json
import faiss
import numpy as np
import os
import time
from tqdm import tqdm
import sys
from dotenv import load_dotenv
from openai import OpenAI

# ------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-large"
BATCH_SIZE = 10

# ------------------------------------------------------------------------------
# Client Initialization
# ------------------------------------------------------------------------------

def initialize_client():
    """Initializes and returns the OpenAI client."""
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        print("✅ Initialized OpenAI client.")
        return client
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        sys.exit(1)

# ------------------------------------------------------------------------------
# JSONL Helpers
# ------------------------------------------------------------------------------

def load_jsonl(filepath):
    """Load JSONL into dict keyed by PrimaryKey."""
    data_map = {}
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            data_map[record["PrimaryKey"]] = record
            records.append(record)
    return data_map, records

# ------------------------------------------------------------------------------
# Embeddings
# ------------------------------------------------------------------------------

def encode_texts(client, texts, batch_size=BATCH_SIZE):
    """Generates batched embeddings using OpenAI."""
    embeddings = []

    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

    for batch in tqdm(batches, desc="🔄 Generating embeddings (OpenAI)"):
        try:
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=batch
            )
            embeddings.extend([e.embedding for e in response.data])
        except Exception as e:
            print(f"\nError during embedding batch: {e}")
            raise

    embeddings_np = np.array(embeddings, dtype="float32")

    # Normalize for cosine similarity with IndexFlatIP
    faiss.normalize_L2(embeddings_np)

    return embeddings_np

# ------------------------------------------------------------------------------
# FAISS Query
# ------------------------------------------------------------------------------

def query_faiss(client, q_text, index, top_k=3):
    """Encodes query text and searches FAISS index."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[q_text]
    )

    q_emb = np.array([response.data[0].embedding], dtype="float32")
    faiss.normalize_L2(q_emb)

    scores, ids = index.search(q_emb, k=top_k)

    return [
        {"PrimaryKey": int(pk), "cosine_similarity": float(score)}
        for pk, score in zip(ids[0], scores[0])
    ]

# ------------------------------------------------------------------------------
# Output Helpers
# ------------------------------------------------------------------------------

def save_few_shots_to_jsonl(triplets, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for triplet in triplets:
            f.write(json.dumps(triplet, ensure_ascii=False) + "\n")

# ------------------------------------------------------------------------------
# Retrieval Logic
# ------------------------------------------------------------------------------

def retrieve_few_shots(client, records, data_map, x, y, index):
    """For each datapoint: anchor + 2 retrieved neighbors."""
    fewShots = []

    for rec in tqdm(records, desc="Retrieving few shots"):
        anchor_pk = rec["PrimaryKey"]
        anchor_context = rec[x]
        anchor_decision = rec[y]

        start_time = time.time()

        retrieved = query_faiss(client, anchor_context, index, top_k=3)

        filtered = [r for r in retrieved if r["PrimaryKey"] != anchor_pk]
        filtered = filtered[:2]

        neighbors = []
        for f in filtered:
            pk = f["PrimaryKey"]
            if pk in data_map:
                neighbors.append({
                    "PrimaryKey": pk,
                    x: data_map[pk][x],
                    y: data_map[pk][y]
                })

        retrieval_time = time.time() - start_time

        fewShots.append({
            "Anchor": {
                "PrimaryKey": anchor_pk,
                x: anchor_context,
                y: anchor_decision
            },
            "Retrieved": neighbors,
            "Time": retrieval_time
        })

    return fewShots

# ------------------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------------------

def run_retrieval_pipeline(client, input_file_path, output_file_path, x_field, y_field):
    print(f"\n--- Processing: {os.path.basename(input_file_path)} ({x_field} -> {y_field}) ---")

    data_map, records = load_jsonl(input_file_path)

    texts = [r[x_field] for r in records]
    ids = [r["PrimaryKey"] for r in records]

    embeddings = encode_texts(client, texts)

    dim = embeddings.shape[1]
    index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
    index.add_with_ids(embeddings, np.array(ids, dtype="int64"))

    print(f"✅ FAISS index built ({dim} dims, {len(records)} vectors).")

    fewShots = retrieve_few_shots(
        client, records, data_map, x=x_field, y=y_field, index=index
    )

    save_few_shots_to_jsonl(fewShots, output_file_path)
    print(f"✅ Saved {len(fewShots)} triplets to {output_file_path}")

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    client = initialize_client()

    print("\n## Context Decision Retrieval (Context -> Decision)")
    # run_retrieval_pipeline(client, "Data/ADR-data/val.jsonl",   "Retrieval/openai/CDval.jsonl",   "Context", "Decision")
    run_retrieval_pipeline(client, "Data/ADR-data/test.jsonl",  "Retrieval/openai/CDtest.jsonl",  "Context", "Decision")
    # run_retrieval_pipeline(client, "Data/ADR-data/train.jsonl", "Retrieval/openai/CDtrain.jsonl", "Context", "Decision")

    print("\n## Title Body Retrieval (Title -> Body)")
    # run_retrieval_pipeline(client, "Data/ADR-data/val.jsonl",   "Retrieval/openai/TBval.jsonl",   "Title", "Body")
    run_retrieval_pipeline(client, "Data/ADR-data/test.jsonl",  "Retrieval/openai/TBtest.jsonl",  "Title", "Body")
    # run_retrieval_pipeline(client, "Data/ADR-data/train.jsonl", "Retrieval/openai/TBtrain.jsonl", "Title", "Body")

    print("\n🎉 Retrieval process complete using text-embedding-3-large.")
