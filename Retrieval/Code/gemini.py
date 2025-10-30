import json
import faiss
import numpy as np
import os
import time
from tqdm import tqdm
from google import genai
from google.genai import types
import sys
from dotenv import load_dotenv

load_dotenv()

GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
DOCUMENT_TASK_TYPE = "RETRIEVAL_DOCUMENT"
QUERY_TASK_TYPE = "RETRIEVAL_QUERY"
BATCH_SIZE = 10 


def initialize_client():
    """Initializes and returns the genai.Client for Vertex AI."""
    project_id = os.environ.get("GCP_PROJECT_ID")
    location = os.environ.get("LOCATION")

    try:
        client = genai.Client(
            vertexai=True,
            project=project_id,
            location=location
        )
        print(f"✅ Initialized genai.Client for Vertex AI project '{project_id}' in '{location}'.")
        return client
    except Exception as e:
        print(f"Error initializing Vertex AI client. Check your authentication/credentials: {e}")
        sys.exit(1)


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


def encode_texts(client, texts, batch_size=BATCH_SIZE, task_type=DOCUMENT_TASK_TYPE):
    """Generates batched embeddings for a list of texts using the Gemini API."""
    embeddings_list = []
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    
    config = types.EmbedContentConfig(task_type=task_type)
    
    for batch in tqdm(batches, desc="🔄 Generating embeddings (Gemini API)"):
        try:
            response = client.models.embed_content(
                model=GEMINI_EMBEDDING_MODEL,
                contents=batch,
                config=config
            )
            # Extend the list with the 'values' (the embedding vector) from each object
            embeddings_list.extend([e.values for e in response.embeddings])
            
        except Exception as e:
            print(f"\nError during embedding for a batch: {e}")
            raise

    embeddings_np = np.array(embeddings_list, dtype="float32")
    return embeddings_np


def query_faiss(client, q_text, index, top_k=3):
    """Encodes the query text and searches the pre-built FAISS index."""
    
    query_config = types.EmbedContentConfig(task_type=QUERY_TASK_TYPE)
    
    q_emb_response = client.models.embed_content(
        model=GEMINI_EMBEDDING_MODEL,
        contents=[q_text], # Pass as a list, even for a single text
        config=query_config 
    )
    
    q_emb = np.array([q_emb_response.embeddings[0].values], dtype="float32")
    
    scores, ids = index.search(q_emb, k=top_k)

    results = [
        {"PrimaryKey": int(pk), "cosine_similarity": float(score)}
        for pk, score in zip(ids[0], scores[0])
    ]
    return results


def save_few_shots_to_jsonl(triplets, filepath):
    """Save triplets list to JSONL file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for triplet in triplets:
            f.write(json.dumps(triplet, ensure_ascii=False) + "\n")


def retrieve_few_shots(client, records, data_map, x, y, index):
    """For each datapoint: anchor + 2 retrieved neighbors."""
    fewShots = []

    for rec in tqdm(records, desc="Retrieving few shots"):
        anchor_pk = rec["PrimaryKey"]
        anchor_context = rec[x]
        anchor_decision = rec[y]

        start_time = time.time()

        # Retrieve top-3
        retrieved = query_faiss(client, anchor_context, index, top_k=3)

        # Drop the anchor if present, otherwise drop the least similar
        filtered = [r for r in retrieved if r['PrimaryKey'] != anchor_pk]
        if len(filtered) > 2:
            filtered = filtered[:2]

        # Map back to full records
        neighbors = []
        for f in filtered:
            pk = f['PrimaryKey']
            if pk in data_map:
                neighbors.append({
                    "PrimaryKey": pk,
                    x: data_map[pk][x],
                    y: data_map[pk][y]
                })
            
        retrieval_time = time.time() - start_time

        # Build final structure
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


# --- Main Orchestration ---

def run_retrieval_pipeline(client, input_file_path, output_file_path, x_field, y_field):
    """Runs the full embedding, indexing, and retrieval pipeline."""
    
    print(f"\n--- Processing: {os.path.basename(input_file_path)} ({x_field} -> {y_field}) ---")
    
    # 1. Load data
    data_map, records = load_jsonl(input_file_path)

    # 2. Extract texts and IDs
    contexts = [r[x_field] for r in records]
    ids = [r["PrimaryKey"] for r in records]

    # 3. Generate embeddings (using DOCUMENT_TASK_TYPE for the corpus)
    embeddings = encode_texts(client, contexts, task_type=DOCUMENT_TASK_TYPE)

    # 4. Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
    # Use int64 for FAISS IDs
    index.add_with_ids(embeddings, np.array(ids, dtype="int64"))
    print(f"✅ FAISS Index built with dimension {dim} and {len(records)} vectors.")

    # 5. Retrieve few-shots
    fewShots = retrieve_few_shots(client, records, data_map, x=x_field, y=y_field, index=index)

    # 6. Save results
    save_few_shots_to_jsonl(fewShots, output_file_path)
    print(f"✅ Saved {len(fewShots)} triplets to {output_file_path}")


if __name__ == "__main__":
    
    # Initialize the Vertex AI Client once
    client = initialize_client()
    
    # --- Context Decision Retrieval ---
    print("\n## Context Decision Retrieval (Context -> Decision)")
    
    run_retrieval_pipeline(client, "Data/ADR-data/val.jsonl", "Retrieval/gemini/CDval.jsonl", "Context", "Decision")
    run_retrieval_pipeline(client, "Data/ADR-data/test.jsonl", "Retrieval/gemini/CDtest.jsonl", "Context", "Decision")
    run_retrieval_pipeline(client, "Data/ADR-data/train.jsonl", "Retrieval/gemini/CDtrain.jsonl", "Context", "Decision")

    # --- Title Body Retrieval ---
    print("\n## Title Body Retrieval (Title -> Body)")
    
    run_retrieval_pipeline(client, "Data/ADR-data/val.jsonl", "Retrieval/gemini/TBval.jsonl", "Title", "Body")
    run_retrieval_pipeline(client, "Data/ADR-data/test.jsonl", "Retrieval/gemini/TBtest.jsonl", "Title", "Body")
    run_retrieval_pipeline(client, "Data/ADR-data/train.jsonl", "Retrieval/gemini/TBtrain.jsonl", "Title", "Body")
    
    print("\n🎉 Retrieval process complete using gemini-embedding-001 via Vertex AI.")