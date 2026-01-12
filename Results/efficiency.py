import os
import sys
import glob
import json
import argparse
import pandas as pd
from pathlib import Path

BASE_DIR = os.getcwd()
OUTPUT_DIR = os.path.join(BASE_DIR, "Results")

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