import os
import sys
import glob
import json
import argparse
import pandas as pd
import nltk
from evaluate import load
from pathlib import Path

from sympy import true

BASE_DIR = os.getcwd()
CACHE_DIR = "../cache"
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

def load_metrics():
    """Loads all evaluation metrics once."""
    print("Loading metrics (this may take a moment)...")
    metrics = {}
    metrics['rouge'] = load('rouge', cache_dir=CACHE_DIR)
    metrics['bleu'] = load('bleu', cache_dir=CACHE_DIR)
    metrics['meteor'] = load('meteor', cache_dir=CACHE_DIR)
    metrics['bertscore'] = load("bertscore", cache_dir=CACHE_DIR)
    return metrics

def calculate_scores(metrics, true_df, pred_df, true_col, pred_col):
    """Calculates Rouge, Bleu, Meteor, and BertScore."""
    results = {}
    
    # Ensure comparison lists are strings
    refs = true_df[true_col].astype(str).tolist()
    preds = pred_df[pred_col].astype(str).tolist()

    print(f"  - Computing Rouge...")
    results['rouge'] = metrics['rouge'].compute(predictions=preds, references=refs)
    
    print(f"  - Computing Bleu...")
    results['bleu'] = metrics['bleu'].compute(predictions=preds, references=refs)
    
    print(f"  - Computing Meteor...")
    results['meteor'] = metrics['meteor'].compute(predictions=preds, references=refs)
    
    print(f"  - Computing BertScore...")
    bs_res = metrics['bertscore'].compute(predictions=preds, references=refs, lang='en', batch_size=64)
    
    results['bertscore'] = {}
    for key in ['precision', 'recall', 'f1']:
        results['bertscore'][key] = pd.Series(bs_res[key]).mean()
    
    if 'Time' in pred_df.columns:
        results['AverageTime'] = pred_df['Time'].mean()
    if 'GeneratedTokens' in pred_df.columns:
        results['AverageGeneratedTokens'] = pred_df['GeneratedTokens'].mean()
        
    return results

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
    elif "qwen" in model_lower or "gemma" in model_lower:
        embedding_subdir = "qwen3-embedding-8B"
    else:
        # Default fallback if model name doesn't match known patterns
        print(f"  [WARN] Unknown model type '{model_name}', defaulting to 'qwen3-embedding-8B' path.")
        embedding_subdir = "qwen3-embedding-8B"

    filename = GT_CONFIGS[dataset]['filename']
    return os.path.join(base_dir, "Retrieval", embedding_subdir, filename)

def process_file(filepath, metrics):
    print(f"\nProcessing: {filepath}")
    
    approach, model, dataset = infer_metadata(filepath)
    
    if not dataset:
        print(f"  [SKIP] Could not identify CDtest or TBtest in filename: {filepath}")
        return

    print(f"  -> Detected: Approach='{approach}', Model='{model}', Dataset='{dataset}'")
    output_filename = f"{approach}_{model}_{dataset}.json"
    if os.path.exists(os.path.join(OUTPUT_DIR, output_filename)):
        print(f"  [SKIP] Output already exists: {output_filename}")
        return

    # Determine dynamic GT path based on model
    gt_path = get_retrieval_path(BASE_DIR, dataset, model)
    gt_config = GT_CONFIGS[dataset]
    
    print(f"  -> Ground Truth Path: {gt_path}")

    # Load Data
    try:
        # Load Ground Truth
        if not os.path.exists(gt_path):
            print(f"  [ERROR] Ground truth file not found at: {gt_path}")
            return

        true_df = pd.read_json(gt_path, lines=True)
        
        # Flatten Anchor if necessary
        if "Anchor" in true_df.columns:
            true_df = true_df["Anchor"].apply(pd.Series)
            
        # Load Prediction
        pred_df = pd.read_json(filepath, lines=True)
        
    except Exception as e:
        print(f"  [ERROR] Failed to read files: {e}")
        return

    # Calculate Scores
    try:
        print(true_df.head())
        print(pred_df.head())
        results = calculate_scores(
            metrics, 
            true_df, 
            pred_df, 
            true_col=gt_config['true_col'], 
            pred_col=gt_config['pred_col']
        )
    except Exception as e:
        print(f"  [ERROR] Scoring failed: {e}")
        return

    # Save Results
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    # Ensure output dir exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"  [SUCCESS] Saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate multiple JSONL prediction files.")
    parser.add_argument('files', nargs='+', help='Path(s) to prediction files. Supports wildcards (e.g., "DRAFT/*/Results/*.jsonl")')
    args = parser.parse_args()

    metrics = load_metrics()

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
    
    for filepath in all_files:
        process_file(os.path.abspath(filepath), metrics)

if __name__ == "__main__":
    main()