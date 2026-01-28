import os
import sys
import glob
import json
import argparse
import pandas as pd
from pathlib import Path

BASE_DIR = os.getcwd()
OUTPUT_DIR = os.path.join(BASE_DIR, "Results")
GT_CONFIGS = {
    "CDtest": {
        "filename": "CDtest.jsonl",
        "true_col": "Decision",
        "pred_col": "Decision"
    },
    "TBtest": {
        "filename": "TBtest.jsonl",
        "true_col": "Body",
        "pred_col": "Body"
    }
}

def infer_metadata(filepath):
    """
    Infers Approach, Model, and Dataset from the file path.
    Assumes structure like: .../Approach/Results/Model[-_]Dataset[-_]results.jsonl
    """
    path_obj = Path(filepath)
    filename = path_obj.name
    
    # 1. Determine Dataset
    if "CDtest" in filename:
        dataset = "CDtest"
    elif "TBtest" in filename:
        dataset = "TBtest"
    else:
        return None, None, None 

    # 2. Determine Approach
    parts = path_obj.parts
    if "Results" in parts:
        idx = parts.index("Results")
        approach = parts[idx - 1]
    else:
        approach = path_obj.parent.name

    # 3. Determine Model
    stem = path_obj.stem 
    stem = stem.replace("-results", "").replace("_results", "")
    
    if f"-{dataset}" in stem:
        model = stem.split(f"-{dataset}")[0]
    elif f"_{dataset}" in stem:
        model = stem.split(f"_{dataset}")[0]
    else:
        model = stem.replace(dataset, "")

    return approach, model, dataset


def get_retrieval_path(base_dir, dataset, model_name):
    """
    Determines the correct Ground Truth path based on the model name.
    """
    # Logic for embedding model folder selection
    model_lower = model_name.lower()
    
    if "gemini" in model_lower:
        embedding_subdir = "gemini"
    elif "gpt" in model_lower:
        embedding_subdir = "openai"
    elif "qwen" in model_lower or "gemma" in model_lower:
        embedding_subdir = "qwen3-embedding-8B"
    else:
        # Default fallback if model name doesn't match known patterns
        print(f"  [WARN] Unknown model type '{model_name}', defaulting to 'qwen3-embedding-8B' path.")
        embedding_subdir = "qwen3-embedding-8B"

    filename = GT_CONFIGS[dataset]['filename']
    return os.path.join(base_dir, "Retrieval", embedding_subdir, filename)


def process_file(efficiency, filepath):
    print(f"\nProcessing: {filepath}")
    
    approach, model, dataset = infer_metadata(filepath)
    
    if not dataset:
        print(f"  [SKIP] Could not identify CDtest or TBtest in filename: {filepath}")
        return

    print(f"  -> Detected: Approach='{approach}', Model='{model}', Dataset='{dataset}'")    
    try:    
        pred_df = pd.read_json(filepath, lines=True)
    except Exception as e:
        print(f"  [ERROR] Failed to read files: {e}")
        return
    
    gt_config = GT_CONFIGS[dataset]
    
    # remove all rows in pred_df where the prediction is null or empty
    pred_df = pred_df[pred_df[gt_config['pred_col']].notnull() & (pred_df[gt_config['pred_col']].astype(str).str.strip() != "")]

    try:
        eff = {
            "approach": approach,
            "model": model,
            "dataset": dataset
        }
        if 'Time' in pred_df.columns:
            eff['time'] = {
                "mean": pred_df['Time'].mean(),
                "std": pred_df['Time'].std(),
                "median": pred_df['Time'].median(),
                "min": pred_df['Time'].min(),
                "max": pred_df['Time'].max()
            }
        if 'GeneratedTokens' in pred_df.columns:
            eff['GeneratedTokens'] = {
                "mean": pred_df['GeneratedTokens'].mean(),
                "std": pred_df['GeneratedTokens'].std(),
                "median": pred_df['GeneratedTokens'].median(),
                "min": int(pred_df['GeneratedTokens'].min()),
                "max": int(pred_df['GeneratedTokens'].max())
            }
        efficiency_key = f"{approach}_{model}_{dataset}"
        efficiency[efficiency_key] = eff
        print(f"  -> Efficiency metrics recorded for {efficiency_key}")
    except Exception as e:
        print(f"  [ERROR] Scoring failed: {e}")
        return

def main():
    parser = argparse.ArgumentParser(description="Evaluate multiple JSONL prediction files.")
    parser.add_argument('files', nargs='+', help='Path(s) to prediction files. Supports wildcards (e.g., "DRAFT/*/Results/*.jsonl")')
    args = parser.parse_args()

    all_files = []
    for f in args.files:
        expanded = glob.glob(f)
        if not expanded:
            print(f"Warning: No files matched pattern '{f}'")
        all_files.extend(expanded)

    if not all_files:
        print("No files found to process.")
        return

    print(f"Found {len(all_files)} files to process.")
    
    efficiency = {}
    
    for filepath in all_files:
        process_file(efficiency, os.path.abspath(filepath))
        
    output_path = os.path.join(OUTPUT_DIR, "efficiency_results.json")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(efficiency, f, indent=2)

if __name__ == "__main__":
    main()