from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import krippendorff
from openpyxl import load_workbook
from scipy.stats import rankdata

APPROACH_ORDER = ["Prompting", "RAFG", "Finetuning", "DRAFT"]
APPROACH_POSITIONS = ["Model_A", "Model_B", "Model_C", "Model_D"]
METRICS = ["closeness", "correctness"]


def load_evaluator_rows(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Load evaluation rows from sheets containing a task identifier column."""
    sheet_rows: dict[str, list[dict[str, Any]]] = {}

    workbook = load_workbook(
        path,
        read_only=True,
        data_only=True,
    )

    try:
        for worksheet in workbook.worksheets:
            rows = list(worksheet.iter_rows(values_only=True))

            if not rows:
                continue

            headers = [
                str(cell).strip() if cell is not None else ""
                for cell in rows[0]
            ]

            identifier_header = next(
                (
                    header
                    for header in ("Decision_ID", "Task_ID")
                    if header in headers
                ),
                None,
            )

            if identifier_header is None:
                continue

            parsed_rows = []

            for row in rows[1:]:
                if not has_content(row):
                    continue

                parsed_row = {
                    header: row[index] if index < len(row) else None
                    for index, header in enumerate(headers)
                }
                parsed_row["Decision_ID"] = safe_float(parsed_row[identifier_header])
                parsed_rows.append(parsed_row)

            sheet_rows[worksheet.title] = parsed_rows

    finally:
        workbook.close()

    return sheet_rows


def has_content(row: tuple[Any, ...]) -> bool:
    """Return True when a row contains at least one non-empty value."""
    return any(
        cell is not None and str(cell).strip()
        for cell in row
    )


def safe_float(value: Any) -> float:
    """Convert a value to float, treating missing/empty values as NaN."""
    if value is None or str(value).strip() == "":
        return np.nan
    try:
        return float(value)
    except ValueError:
        return np.nan


def get_approach_score(
    row: dict[str, Any],
    position: str,
) -> tuple[str, float, float] | None:
    """Return (approach, closeness, correctness) for an approach position."""
    approach = row.get(position)

    if approach is None or not str(approach).strip():
        return None

    approach = str(approach).strip()

    closeness = safe_float(row.get(f"{position}_Closeness"))
    correctness = safe_float(row.get(f"{position}_Correctness"))

    return approach, closeness, correctness


def get_approach_scores(row: dict[str, Any]) -> list[tuple[str, float, float]]:
    """Return scores for all populated approach positions in a row."""
    scores = []

    for position in APPROACH_POSITIONS:
        score = get_approach_score(row, position)
        if score is not None:
            scores.append(score)

    return scores


def build_reliability_matrices(
    evaluators: dict[str, list[dict[str, Any]]]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Aligns scores by Decision_ID and Approach, converting raw scores into 
    rankings for each sample to calculate agreement on relative preferences.
    """
    evaluator_keys = list(evaluators.keys())
    num_evaluators = len(evaluator_keys)

    closeness_data = defaultdict(lambda: [np.nan] * num_evaluators)
    correctness_data = defaultdict(lambda: [np.nan] * num_evaluators)

    for i, eval_key in enumerate(evaluator_keys):
        for row in evaluators[eval_key]:
            decision_id = row.get("Decision_ID")
            if not decision_id:
                continue

            scores = get_approach_scores(row)
            if not scores:
                continue

            # Separate the components for ranking
            approaches = [s[0] for s in scores]
            closeness_vals = np.array([s[1] for s in scores])
            correctness_vals = np.array([s[2] for s in scores])

            def get_ranks(vals: np.ndarray) -> np.ndarray:
                ranks = np.full(len(vals), np.nan)
                valid_mask = ~np.isnan(vals)
                if np.any(valid_mask):
                    valid_vals = vals[valid_mask]
                    # We rank negative values so the highest score (e.g., 5) gets rank 1
                    valid_ranks = rankdata(
                        [-v for v in valid_vals], method='average')
                    ranks[valid_mask] = valid_ranks
                return ranks

            closeness_ranks = get_ranks(closeness_vals)
            correctness_ranks = get_ranks(correctness_vals)

            for approach, c_rank, corr_rank in zip(approaches, closeness_ranks, correctness_ranks):
                key = (decision_id, approach)
                closeness_data[key][i] = c_rank
                correctness_data[key][i] = corr_rank

    closeness_matrix = np.array(list(closeness_data.values())).T
    correctness_matrix = np.array(list(correctness_data.values())).T

    return closeness_matrix, correctness_matrix


def calculate_krippendorff(matrix: np.ndarray) -> float:
    """Calculate interval Krippendorff's Alpha on rank-transformed data."""
    try:
        if matrix.size == 0 or matrix.ndim < 2 or matrix.shape[1] == 0 or np.isnan(matrix).all():
            return np.nan

        # We switch to "interval" because fractional ranks (like 1.5 for ties)
        # represent equidistant, continuous intervals, not strict ordinal categories.
        return krippendorff.alpha(
            reliability_data=matrix,
            level_of_measurement="interval"
        )
    except Exception as e:
        print(f"Warning: Could not calculate Krippendorff's alpha: {e}")
        return np.nan


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate approach means and rankings, ignoring NaN values."""
    approach_stats = defaultdict(
        lambda: {
            "closeness": [],
            "correctness": [],
        }
    )

    for row in rows:
        scores = get_approach_scores(row)
        if not scores:
            continue

        for approach, closeness, correctness in scores:
            if not math.isnan(closeness):
                approach_stats[approach]["closeness"].append(closeness)
            if not math.isnan(correctness):
                approach_stats[approach]["correctness"].append(correctness)

    mean_per_approach = {}

    for approach in APPROACH_ORDER:
        if approach not in approach_stats:
            continue

        c_vals = approach_stats[approach]["closeness"]
        corr_vals = approach_stats[approach]["correctness"]

        mean_per_approach[approach] = {
            "closeness": sum(c_vals) / len(c_vals) if c_vals else np.nan,
            "correctness": sum(corr_vals) / len(corr_vals) if corr_vals else np.nan,
        }

    return {
        "mean_per_approach": {
            approach: {
                "closeness": round(stats["closeness"], 3) if not math.isnan(stats["closeness"]) else "N/A",
                "correctness": round(stats["correctness"], 3) if not math.isnan(stats["correctness"]) else "N/A",
            }
            for approach, stats in mean_per_approach.items()
        }
    }


def combined_mean(
    rows_list: list[dict[str, Any]],
) -> dict[str, dict[str, float | str]]:
    """Calculate approach means across multiple evaluator datasets."""
    scores = defaultdict(
        lambda: {
            "closeness": [],
            "correctness": [],
        }
    )

    for row in rows_list:
        for approach, closeness, correctness in get_approach_scores(row):
            if not math.isnan(closeness):
                scores[approach]["closeness"].append(closeness)
            if not math.isnan(correctness):
                scores[approach]["correctness"].append(correctness)

    combined = {}

    for approach in APPROACH_ORDER:
        if approach not in scores:
            continue

        closeness_values = scores[approach]["closeness"]
        correctness_values = scores[approach]["correctness"]

        combined[approach] = {
            "closeness": round(sum(closeness_values) / len(closeness_values), 3) if closeness_values else "N/A",
            "correctness": round(sum(correctness_values) / len(correctness_values), 3) if correctness_values else "N/A",
        }

    return combined


def find_evaluator_sheets(
    base_dir: Path,
    task: str
) -> dict[str, list[dict[str, Any]]]:
    """Load the two author evaluators and the named expert evaluator."""
    authors = load_evaluator_rows(base_dir / f"{task} Authors.xlsx")
    expert = load_evaluator_rows(base_dir / f"{task} Expert.xlsx")

    author_rows = list(authors.values())
    expert_rows = list(expert.values())

    if len(author_rows) < 2:
        raise ValueError(
            "Need at least two evaluator sheets in the Authors workbook.")
    if len(expert_rows) != 1:
        raise ValueError(
            "Need exactly one evaluator sheet in the Expert workbook.")

    return {
        f"Evaluator 1 ({list(authors.keys())[0]})": author_rows[0],
        f"Evaluator 2 ({list(authors.keys())[1]})": author_rows[1],
        f"Expert ({list(expert.keys())[0]})": expert_rows[0],
    }


def format_results(
    evaluators: dict[str, list[dict[str, Any]]],
) -> str:
    """Generate the complete evaluation report, including inter-rater agreement."""
    summaries = {
        name: summarize_rows(rows)["mean_per_approach"]
        for name, rows in evaluators.items()
    }

    all_rows = [row for rows in evaluators.values() for row in rows]
    summaries["Combined"] = combined_mean(all_rows)

    lines = [
        "## Performance Means",
        "| Evaluator | Approach | Closeness | Correctness |",
        "|---|---|---:|---:|"
    ]

    for evaluator, summary in summaries.items():
        for approach in APPROACH_ORDER:
            if approach in summary:
                lines.append(
                    f"| {evaluator} | {approach} | "
                    f"{summary[approach]['closeness']} | "
                    f"{summary[approach]['correctness']} |"
                )

    lines.append("\n## Inter-Rater Agreement (Krippendorff's Alpha - Interval)")

    closeness_matrix, correctness_matrix = build_reliability_matrices(
        evaluators)

    alpha_closeness = calculate_krippendorff(closeness_matrix)
    alpha_correctness = calculate_krippendorff(correctness_matrix)

    lines.append(f"* **Closeness:** {alpha_closeness:.3f}")
    lines.append(f"* **Correctness:** {alpha_correctness:.3f}")
    lines.append("\n*(Note: Alpha > 0.66 is acceptable, > 0.80 is reliable)*")

    return "\n".join(lines)


def visualize_raw_matrix(
    evaluators: dict[str, list[dict[str, Any]]], 
    metric: str = "closeness"
) -> str:
    """
    Creates a formatted text table of raw scores for all evaluators,
    labeled by Decision_ID and Approach.
    """
    evaluator_keys = list(evaluators.keys())
    
    # Store data as: (Decision_ID, Approach) -> [eval_1_score, eval_2_score, eval_3_score]
    matrix_data = defaultdict(lambda: ["NaN"] * len(evaluator_keys))
    
    for i, eval_key in enumerate(evaluator_keys):
        for row in evaluators[eval_key]:
            decision_id = row.get("Decision_ID")
            if not decision_id:
                continue
            
            for approach, closeness, correctness in get_approach_scores(row):
                key = (str(decision_id), approach)
                
                # Select the score based on the requested metric
                val = closeness if metric == "closeness" else correctness
                
                if math.isnan(val):
                    matrix_data[key][i] = "NaN"
                else:
                    matrix_data[key][i] = f"{val:.1f}"

    # Sort the rows by Decision_ID, then by your Approach order
    def sort_key(k):
        dec_id, app = k
        app_idx = APPROACH_ORDER.index(app) if app in APPROACH_ORDER else 999
        return (dec_id, app_idx)
        
    sorted_keys = sorted(matrix_data.keys(), key=sort_key)
    
    # Build the table layout
    headers = ["Decision_ID", "Approach"] + evaluator_keys
    
    # Dynamically calculate column widths for clean alignment
    col_widths = [len(h) for h in headers]
    for k in sorted_keys:
        col_widths[0] = max(col_widths[0], len(k[0]))
        col_widths[1] = max(col_widths[1], len(k[1]))
    
    # Add a little padding to the widths
    col_widths = [w + 2 for w in col_widths]
    
    lines = []
    lines.append(f"### Raw Scores Matrix: {metric.capitalize()}")
    
    header_line = "".join(h.ljust(w) for h, w in zip(headers, col_widths))
    lines.append(header_line)
    lines.append("-" * len(header_line))
    
    for k in sorted_keys:
        row_data = [k[0], k[1]] + matrix_data[k]
        row_line = "".join(str(item).ljust(w) for item, w in zip(row_data, col_widths))
        lines.append(row_line)
        
    return "\n".join(lines)

def main(task: str) -> None:
    base_dir = Path(__file__).resolve().parent
    output_file = base_dir / f"{task}_results.txt"

    try:
        evaluators = find_evaluator_sheets(base_dir, task)
        
        print(f"\n{'='*60}")
        print(f"VISUALIZING DATA FOR TASK: {task}")
        print(f"{'='*60}")
        print(visualize_raw_matrix(evaluators, metric="closeness"))
        print("\n")
        print(visualize_raw_matrix(evaluators, metric="correctness"))
        print(f"{'='*60}\n")
        
        report = format_results(evaluators)

        output_file.write_text(
            report,
            encoding="utf-8",
        )
        print(f"[{task}] Results written to: {output_file}")

    except Exception as e:
        print(f"[{task}] Failed to process: {e}")


if __name__ == "__main__":
    main("CD")
    main("TB")
