import argparse
import glob
import json
import math
import os
import re
from itertools import combinations

import pandas as pd


BASE_DIR = os.getcwd()
INPUT_DIR = os.path.join(BASE_DIR, "Results/scores/per_sample_scores")
OUTPUT_DIR = os.path.join(BASE_DIR, "Results/stat_tests")
DEFAULT_METRICS = ["bert_f1", "rouge1", "rougeL",
                   "bleu", "meteor", "bert_precision", "bert_recall"]


FILE_RE = re.compile(
    r"^(?P<approach>.+?)_(?P<model>.+)_(?P<dataset>CDtest|TBtest)_per_sample\.jsonl$")


def canonical_model_name(model: str) -> str:
    model_lower = model.lower()

    if "gpt_5" in model_lower or "gpt-5" in model_lower:
        return "gpt-5"
    if "gemini-2.5-flash" in model_lower:
        return "gemini-2.5-flash"
    if "gemma-3-4b-it" in model_lower:
        return "gemma-3-4b-it"
    if "qwen3-30b" in model_lower:
        return "qwen3-30b-a3b-instruct-2507"

    return model_lower


def rankdata_average(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(indexed):
        tie_end = position + 1
        tie_value = indexed[position][1]
        while tie_end < len(indexed) and indexed[tie_end][1] == tie_value:
            tie_end += 1
        average_rank = (position + 1 + tie_end) / 2.0
        for tie_index in range(position, tie_end):
            original_index = indexed[tie_index][0]
            ranks[original_index] = average_rank
        position = tie_end
    return ranks


def regularized_gamma_q(a: float, x: float) -> float:
    if a <= 0.0 or x < 0.0:
        return float("nan")
    if x == 0.0:
        return 1.0

    eps = 1e-14
    fpmin = 1e-300
    gln = math.lgamma(a)

    if x < a + 1.0:
        ap = a
        summation = 1.0 / a
        delta = summation
        while True:
            ap += 1.0
            delta *= x / ap
            summation += delta
            if abs(delta) < abs(summation) * eps:
                break
        p = summation * math.exp(-x + a * math.log(x) - gln)
        return max(0.0, min(1.0, 1.0 - p))

    b = x + 1.0 - a
    c = 1.0 / fpmin
    d = 1.0 / b
    h = d
    i = 1
    while True:
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < fpmin:
            d = fpmin
        c = b + an / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
        i += 1
    q = math.exp(-x + a * math.log(x) - gln) * h
    return max(0.0, min(1.0, q))


def chi_square_sf(statistic: float, degrees_of_freedom: int) -> float:
    return regularized_gamma_q(degrees_of_freedom / 2.0, statistic / 2.0)


def holm_adjust(p_values: list[float]) -> list[float]:
    if not p_values:
        return []

    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, (original_index, p_value) in enumerate(indexed):
        adjusted_value = min(1.0, (m - rank) * p_value)
        running_max = max(running_max, adjusted_value)
        adjusted[original_index] = running_max
    return adjusted


def load_per_sample_scores(input_dir: str) -> pd.DataFrame:
    rows = []
    for file_path in glob.glob(os.path.join(input_dir, "*.jsonl")):
        match = FILE_RE.match(os.path.basename(file_path))
        if not match:
            continue

        meta = match.groupdict()
        df = pd.read_json(file_path, lines=True)
        if "sample_id" not in df.columns:
            raise ValueError(f"sample_id column missing in {file_path}")

        metric_columns = [
            column for column in DEFAULT_METRICS if column in df.columns]
        if not metric_columns:
            raise ValueError(f"No known metric columns found in {file_path}")

        df = df.assign(
            approach=meta["approach"],
            model=canonical_model_name(meta["model"]),
            dataset=meta["dataset"],
            source_file=os.path.basename(file_path),
        )

        rows.append(df[["sample_id", "approach", "model",
                    "dataset", *metric_columns, "source_file"]])

    if not rows:
        raise FileNotFoundError(
            f"No per-sample score files found in {input_dir}")

    return pd.concat(rows, ignore_index=True)


def kendalls_w(friedman_statistic: float, n_blocks: int, n_groups: int) -> float:
    if n_blocks <= 0 or n_groups <= 1:
        return float("nan")
    return friedman_statistic / (n_blocks * (n_groups - 1))


def rank_biserial_from_paired(x: pd.Series, y: pd.Series) -> float:
    diff = (x - y).dropna()
    diff = diff[diff != 0]
    if diff.empty:
        return float("nan")

    diff_values = diff.tolist()
    ranks = rankdata_average([abs(value) for value in diff_values])
    positive = sum(rank for rank, value in zip(
        ranks, diff_values) if value > 0)
    negative = sum(rank for rank, value in zip(
        ranks, diff_values) if value < 0)
    total = sum(ranks)
    return float((positive - negative) / total)


def wilcoxon_signed_rank(x: pd.Series, y: pd.Series) -> tuple[float, float, float]:
    diff = (x - y).dropna()
    diff = diff[diff != 0]
    if diff.empty:
        raise ValueError("No non-zero paired differences")

    abs_values = [abs(value) for value in diff.tolist()]
    ranks = rankdata_average(abs_values)
    positive_rank_sum = sum(rank for rank, value in zip(
        ranks, diff.tolist()) if value > 0)
    negative_rank_sum = sum(rank for rank, value in zip(
        ranks, diff.tolist()) if value < 0)
    statistic = min(positive_rank_sum, negative_rank_sum)

    n = len(diff)
    mean = n * (n + 1) / 4.0
    variance = n * (n + 1) * (2 * n + 1) / 24.0
    if variance == 0.0:
        raise ValueError("Zero variance in Wilcoxon test")

    continuity = 0.5
    z = (abs(statistic - mean) - continuity) / math.sqrt(variance)
    p_value = math.erfc(abs(z) / math.sqrt(2.0))
    effect_size = (positive_rank_sum - negative_rank_sum) / (n * (n + 1) / 2.0)
    return float(statistic), float(p_value), float(effect_size)


def friedman_test(arrays: list[pd.Series]) -> tuple[float, float]:
    n_blocks = len(arrays[0])
    n_groups = len(arrays)
    ranks_per_block = []
    for row in zip(*[series.tolist() for series in arrays]):
        ranks_per_block.append(rankdata_average(list(row)))

    rank_sums = [sum(block[group_index] for block in ranks_per_block)
                 for group_index in range(n_groups)]
    chi_square = (12.0 / (n_blocks * n_groups * (n_groups + 1))) * sum(
        rank_sum ** 2 for rank_sum in rank_sums
    ) - 3.0 * n_blocks * (n_groups + 1)
    p_value = chi_square_sf(chi_square, n_groups - 1)
    return float(chi_square), float(p_value)


def paired_frame(df: pd.DataFrame, group_col: str, groups: list[str], metric: str) -> pd.DataFrame:
    wide = df[df[group_col].isin(groups)].pivot_table(
        index="sample_id",
        columns=group_col,
        values=metric,
        aggfunc="mean",
    )
    return wide.dropna(subset=groups, how="any") if all(group in wide.columns for group in groups) else wide


def run_family_tests(wide: pd.DataFrame, groups: list[str], family_label: str, scope: dict, metric: str) -> list[dict]:
    results = []
    wide = wide.dropna(subset=groups)
    if wide.empty or len(groups) < 2:
        return results
    n_blocks = len(wide)

    if len(groups) >= 3:
        stat, p_value = friedman_test([wide[group] for group in groups])
        results.append(
            {
                **scope,
                "metric": metric,
                "family": family_label,
                "test": "friedman",
                "comparison": "omnibus",
                "n_blocks": n_blocks,
                "n_groups": len(groups),
                "statistic": float(stat),
                "p_value": float(p_value),
                "effect_size": kendalls_w(float(stat), n_blocks, len(groups)),
            }
        )
    elif len(groups) == 2:
        try:
            stat, p_value, effect_size = wilcoxon_signed_rank(
                wide[groups[0]], wide[groups[1]])
        except ValueError:
            return results
        results.append(
            {
                **scope,
                "metric": metric,
                "family": family_label,
                "test": "wilcoxon",
                "comparison": f"{groups[0]} vs {groups[1]}",
                "n_blocks": n_blocks,
                "n_groups": len(groups),
                "statistic": float(stat),
                "p_value": float(p_value),
                "effect_size": effect_size,
            }
        )

    pairwise_rows = []
    pairwise_pairs = []
    for left, right in combinations(groups, 2):
        pair_df = wide[[left, right]].dropna()
        if pair_df.empty:
            continue
        try:
            stat, p_value, effect_size = wilcoxon_signed_rank(
                pair_df[left], pair_df[right])
        except ValueError:
            continue

        pairwise_rows.append(
            {
                **scope,
                "metric": metric,
                "family": family_label,
                "test": "wilcoxon_posthoc",
                "comparison": f"{left} vs {right}",
                "n_blocks": len(pair_df),
                "n_groups": 2,
                "statistic": float(stat),
                "p_value": float(p_value),
                "effect_size": effect_size,
            }
        )
        pairwise_pairs.append((left, right))

    if pairwise_rows:
        adjusted = holm_adjust([row["p_value"] for row in pairwise_rows])
        for row, adjusted_p in zip(pairwise_rows, adjusted):
            row["p_holm"] = float(adjusted_p)
        results.extend(pairwise_rows)

    return results


def compare_approaches_within_models(df: pd.DataFrame, metric: str) -> list[dict]:
    all_rows = []
    for dataset, dataset_subset in df.groupby("dataset"):
        for model, subset in dataset_subset.groupby("model"):
            available_groups = sorted(
                subset["approach"].dropna().unique().tolist())
            if len(available_groups) < 2:
                continue

            wide = subset.pivot_table(
                index="sample_id",
                columns="approach",
                values=metric,
                aggfunc="mean",
            )
            groups = [
                group for group in available_groups if group in wide.columns]
            rows = run_family_tests(
                wide, groups, "approach-within-model", {"dataset": dataset, "model": model}, metric)
            all_rows.extend(rows)

    return all_rows


def compare_models_within_approaches(df: pd.DataFrame, metric: str) -> list[dict]:
    all_rows = []
    for dataset, dataset_subset in df.groupby("dataset"):
        for approach, subset in dataset_subset.groupby("approach"):
            available_groups = sorted(
                subset["model"].dropna().unique().tolist())
            if len(available_groups) < 2:
                continue

            wide = subset.pivot_table(
                index="sample_id",
                columns="model",
                values=metric,
                aggfunc="mean",
            )
            groups = [
                group for group in available_groups if group in wide.columns]
            rows = run_family_tests(wide, groups, "model-within-approach",
                                    {"dataset": dataset, "approach": approach}, metric)
            all_rows.extend(rows)

    return all_rows


def main():
    parser = argparse.ArgumentParser(
        description="Run paired significance tests over per-sample score files.")
    parser.add_argument("--input-dir", default=INPUT_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--metrics", nargs="*", default=DEFAULT_METRICS,
                        help="Metrics to test. Default: all available metrics.")
    args = parser.parse_args()

    df = load_per_sample_scores(args.input_dir)
    requested_metrics = [
        metric for metric in args.metrics if metric in df.columns]
    if not requested_metrics:
        raise ValueError(
            f"None of the requested metrics are available. Requested={args.metrics}")

    results = []
    for metric in requested_metrics:
        metric_df = df[["sample_id", "approach", "model",
                        "dataset", metric]].dropna(subset=[metric])
        results.extend(compare_approaches_within_models(metric_df, metric))
        results.extend(compare_models_within_approaches(metric_df, metric))

    out_df = pd.DataFrame(results)
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "significance_tests.csv")
    out_df.to_csv(csv_path, index=False)

    if out_df.empty:
        print("No comparable groups found.")
        return

    display_cols = ["dataset", "family", "test", "comparison",
                    "n_blocks", "statistic", "p_value", "p_holm", "effect_size"]
    print(out_df[display_cols].to_string(index=False))
    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
