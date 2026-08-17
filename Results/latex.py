import pandas as pd

# -------------------------
# Configuration
# -------------------------
CSV_FILE = "Results/stat_tests/significance_tests.csv"

DATASET = "TBtest"
METRICS = {"rouge1": "ROUGE-1", "rougeL": "ROUGE-L", "bleu": "BLEU", "meteor": "METEOR",
           "bert_precision": "BERT-Precision", "bert_recall": "BERT-Recall", "bert_f1": "BERT-F1"}

MODELS = {"gemini-2.5-flash": "Gemini-2.5",
          "gemma-3-4b-it": "Gemma-3-4b",
          "qwen3-30b-a3b-instruct-2507": "Qwen-3-30b",
        #   "gpt-5": "GPT-5",
          }

APPROACH_ORDER = [
    "DRAFT",
    "Finetune",
    "Prompting",
    "RAFG"
]

SIG_THRESHOLDS = [
    (0.001, "***"),
    (0.01, "**"),
    (0.05, "*"),
]

# -------------------------
# Helpers
# -------------------------


def sig_symbol(p):
    if pd.isna(p):
        return ""
    for thresh, sym in SIG_THRESHOLDS:
        if p < thresh:
            return sym
    return "ns"


def format_effect_with_sig(effect, p):
    if pd.isna(effect):
        return "--"
    marker = sig_symbol(p)
    if marker:
        return f"{effect:.3f} {marker}"
    return f"{effect:.3f}"


def lookup_comparison(posthoc, left, right):
    direct = posthoc[posthoc["comparison"] == f"{left} vs {right}"]
    if not direct.empty:
        row = direct.iloc[0]
        return format_effect_with_sig(row["effect_size"], row["p_holm"])

    reverse = posthoc[posthoc["comparison"] == f"{right} vs {left}"]
    if not reverse.empty:
        row = reverse.iloc[0]
        return format_effect_with_sig(-row["effect_size"], row["p_holm"])

    return "--"


def create_dataset_table(dataset):
    df = pd.read_csv(CSV_FILE)
    df = df[
        (df["dataset"] == dataset) &
        (df["family"] == "approach-within-model")
    ]

    available_metrics = [
        metric for metric in METRICS if not df[df["metric"] == metric].empty]
    pairs = [
        ("DRAFT", "Finetune"),
        ("DRAFT", "Prompting"),
        ("DRAFT", "RAFG"),
        ("Finetune", "Prompting"),
        ("RAFG", "Prompting"),
    ]

    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\small")
    print(r"\label{tab:significance}")
    print(r"\resizebox{\linewidth}{!}{")
    print(r"\begin{tabular}{llcccccc}")
    print(r"\toprule")
    print(r"Metric & Model & Friedman & DRAFT-Finetune & DRAFT-Prompting & DRAFT-RAFG & Finetune-Prompting & RAFG-Prompting \\")
    print(r"\midrule")
    print("\n")

    for metric_index, metric in enumerate(available_metrics):
        metric_df = df[df["metric"] == metric]
        models = sorted(metric_df["model"].dropna().unique())
        if not models:
            continue

        for model_index, model in enumerate(models):
            if model not in MODELS:
                continue
            model_df = metric_df[metric_df["model"] == model]

            friedman_rows = model_df[model_df["test"] == "friedman"]
            if friedman_rows.empty:
                friedman_cell = "--"
            else:
                row = friedman_rows.iloc[0]
                friedman_cell = format_effect_with_sig(
                    row["effect_size"], row["p_value"])

            posthoc = model_df[model_df["test"] == "wilcoxon_posthoc"]
            vals = [lookup_comparison(posthoc, left, right)
                    for left, right in pairs]

            metric_cell = rf"\multirow{{{len(MODELS)}}}{{*}}{{{METRICS.get(metric, metric).replace('_', '\\_')}}}" if model_index == 0 else ""
            print(
                f"{metric_cell} & {MODELS.get(model, model)} & {friedman_cell} & "
                + " & ".join(vals)
                + r" \\")

        if metric_index < len(available_metrics) - 1:
            print(r"\midrule"+"\n")

    print(r"\bottomrule")
    print(r"\end{tabular}}")
    print(rf"\caption{{Statistical significance of approach comparisons for {dataset}. "
          r"Friedman omnibus test followed by Wilcoxon signed-rank tests with Holm correction. "
          r"Significance levels: *** ($p<0.001$), ** ($p<0.01$), * ($p<0.05$), ns = not significant.}"
          )
    print(r"\end{table}")


create_dataset_table(DATASET)
