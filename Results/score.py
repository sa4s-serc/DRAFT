import os
import glob
import json
import argparse
import pandas as pd
from evaluate import load
from pathlib import Path

BASE_DIR = os.getcwd()
CACHE_DIR = "../cache"
OUTPUT_DIR = os.path.join(BASE_DIR, "Results/scores")
PER_SAMPLE_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "per_sample_scores")

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

SAMPLE_ID_CANDIDATES = ["PrimaryKey", "IssueKey", "sample_id", "id", "ID"]


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

    print("  - Computing Rouge...")
    results['rouge'] = metrics['rouge'].compute(predictions=preds, references=refs)

    print("  - Computing Bleu...")
    results['bleu'] = metrics['bleu'].compute(predictions=preds, references=refs)

    print("  - Computing Meteor...")
    results['meteor'] = metrics['meteor'].compute(predictions=preds, references=refs)

    print("  - Computing BertScore...")
    bs_res = metrics['bertscore'].compute(predictions=preds, references=refs, lang='en', batch_size=64)

    results['bertscore'] = {}
    for key in ['precision', 'recall', 'f1']:
        results['bertscore'][key] = pd.Series(bs_res[key]).mean()

    if 'Time' in pred_df.columns:
        results['AverageTime'] = pred_df['Time'].mean()
    if 'GeneratedTokens' in pred_df.columns:
        results['AverageGeneratedTokens'] = pred_df['GeneratedTokens'].mean()

    return results


def calculate_scores_per_sample(metrics, true_df, pred_df, true_col, pred_col):
    """Calculates per-sample Rouge, Bleu, Meteor, and BertScore."""
    refs = true_df[true_col].astype(str).tolist()
    preds = pred_df[pred_col].astype(str).tolist()

    rouge_res = metrics['rouge'].compute(
        predictions=preds, references=refs, use_aggregator=False)
    bleu_res = []
    meteor_res = []

    for pred, ref in zip(preds, refs):
        bleu = metrics['bleu'].compute(predictions=[pred], references=[ref])
        meteor = metrics['meteor'].compute(
            predictions=[pred], references=[ref])
        bleu_res.append(bleu['bleu'])
        meteor_res.append(meteor['meteor'])

    bert_res = metrics['bertscore'].compute(
        predictions=preds, references=refs, lang='en', batch_size=64)

    per_sample = pred_df.copy().reset_index(drop=True)
    per_sample['rouge1'] = rouge_res['rouge1']
    per_sample['rouge2'] = rouge_res['rouge2']
    per_sample['rougeL'] = rouge_res['rougeL']
    per_sample['bleu'] = bleu_res
    per_sample['meteor'] = meteor_res
    per_sample['bert_precision'] = bert_res['precision']
    per_sample['bert_recall'] = bert_res['recall']
    per_sample['bert_f1'] = bert_res['f1']

    return per_sample


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


def process_file(filepath, metrics):
    print(f"\nProcessing: {filepath}")

    approach, model, dataset = infer_metadata(filepath)

    if not dataset:
        print(f"  [SKIP] Could not identify CDtest or TBtest in filename: {filepath}")
        return

    print(f"  -> Detected: Approach='{approach}', Model='{model}', Dataset='{dataset}'")
    output_filename = f"{approach}_{model}_{dataset}.json"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    per_sample_output_path = os.path.join(
        PER_SAMPLE_OUTPUT_DIR, f"{approach}_{model}_{dataset}_per_sample.jsonl")
    aggregate_exists = os.path.exists(output_path)
    per_sample_exists = os.path.exists(per_sample_output_path)
    if aggregate_exists and per_sample_exists:
        print(f"  [SKIP] Outputs already exist: {output_filename} and per-sample JSONL")
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

    except (OSError, ValueError, RuntimeError, KeyError) as e:
        print(f"  [ERROR] Failed to read files: {e}")
        return

    # remove all rows in pred_df and true_df where the prediction is null or empty
    pred_df = pred_df[pred_df[gt_config['pred_col']].notnull() & (pred_df[gt_config['pred_col']].astype(str).str.strip() != "")]
    true_df = true_df.loc[pred_df.index]
    sample_id_col = next((col for col in SAMPLE_ID_CANDIDATES if col in pred_df.columns), None)

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
    except (ValueError, RuntimeError, KeyError) as e:
        print(f"  [ERROR] Scoring failed: {e}")
        return

    try:
        per_sample_df = calculate_scores_per_sample(
            metrics,
            true_df,
            pred_df,
            true_col=gt_config['true_col'],
            pred_col=gt_config['pred_col']
        )
        per_sample_df['approach'] = approach
        per_sample_df['model'] = model
        per_sample_df['dataset'] = dataset
        per_sample_df['sample_id'] = pred_df[sample_id_col].tolist() if sample_id_col else pred_df.index.tolist()
        per_sample_df['reference'] = true_df[gt_config['true_col']].astype(str).tolist()
        per_sample_df['prediction'] = pred_df[gt_config['pred_col']].astype(str).tolist()
    except (ValueError, RuntimeError, KeyError) as e:
        print(f"  [ERROR] Per-sample scoring failed: {e}")
        return

    # Save Results
    # Ensure output dir exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not aggregate_exists:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)

    os.makedirs(PER_SAMPLE_OUTPUT_DIR, exist_ok=True)
    per_sample_columns = [
        'sample_id',
        'rouge1',
        'rouge2',
        'rougeL',
        'bleu',
        'meteor',
        'bert_precision',
        'bert_recall',
        'bert_f1',
    ]
    if not per_sample_exists:
        per_sample_df[per_sample_columns].to_json(
            per_sample_output_path, orient='records', lines=True)

    if not aggregate_exists:
        print(f"  [SUCCESS] Saved to: {output_path}")
    else:
        print(f"  [SKIP] Aggregate output already exists: {output_filename}")
    if not per_sample_exists:
        print(f"  [SUCCESS] Saved per-sample scores to: {per_sample_output_path}")
    else:
        print(f"  [SKIP] Per-sample output already exists: {os.path.basename(per_sample_output_path)}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate multiple JSONL prediction files.")
    parser.add_argument(
        'files', nargs='+', help='Path(s) to prediction files. Supports wildcards (e.g., "DRAFT/*/Results/*.jsonl")')
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
