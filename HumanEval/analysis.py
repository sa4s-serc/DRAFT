from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from openpyxl import load_workbook


MODEL_ORDER = ["Finetuning", "RAFG", "DRAFT", "Prompting"]
MODEL_POSITIONS = ["Model_A", "Model_B", "Model_C", "Model_D"]
METRICS = ["closeness", "correctness"]


def load_evaluator_rows(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Load evaluation rows from all sheets containing a Decision_ID column."""
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

            if "Decision_ID" not in headers:
                continue

            parsed_rows = []

            for row in rows[1:]:
                if not has_content(row):
                    continue

                parsed_rows.append(
                    {
                        header: row[index] if index < len(row) else None
                        for index, header in enumerate(headers)
                    }
                )

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


def safe_int(value: Any) -> int:
    """Convert a value to int, treating missing/empty values as zero."""
    if value is None or str(value).strip() == "":
        return 0

    return int(value)


def get_model_score(
    row: dict[str, Any],
    position: str,
) -> tuple[str, int, int] | None:
    """Return (model, closeness, correctness) for a model position."""
    model = row.get(position)

    if model is None or not str(model).strip():
        return None

    model = str(model).strip()

    closeness = safe_int(row.get(f"{position}_Closeness"))
    correctness = safe_int(row.get(f"{position}_Correctness"))

    return model, closeness, correctness


def get_model_scores(row: dict[str, Any]) -> list[tuple[str, int, int]]:
    """Return scores for all populated model positions in a row."""
    scores = []

    for position in MODEL_POSITIONS:
        score = get_model_score(row, position)

        if score is not None:
            scores.append(score)

    return scores


def winner_by_metric(
    row: dict[str, Any],
    metric: str,
) -> str | None:
    """Return the model with the highest score for a given metric."""
    scores = get_model_scores(row)

    if not scores:
        return None

    score_index = 1 if metric == "closeness" else 2

    return max(scores, key=lambda score: score[score_index])[0]


def cohen_kappa(
    labels_a: list[Any],
    labels_b: list[Any],
) -> float:
    """Calculate Cohen's kappa between two sets of categorical labels."""
    if len(labels_a) != len(labels_b):
        raise ValueError("Label lists must have the same length.")

    n = len(labels_a)

    if n == 0:
        return 0.0

    categories = sorted(set(labels_a) | set(labels_b), key=str)

    counts = {
        category_a: {
            category_b: 0
            for category_b in categories
        }
        for category_a in categories
    }

    for label_a, label_b in zip(labels_a, labels_b):
        counts[label_a][label_b] += 1

    observed_agreement = (
        sum(counts[category][category] for category in categories) / n
    )

    row_totals = {
        category: sum(counts[category].values())
        for category in categories
    }

    column_totals = {
        category: sum(
            counts[row_category][category]
            for row_category in categories
        )
        for category in categories
    }

    expected_agreement = sum(
        (row_totals[category] / n)
        * (column_totals[category] / n)
        for category in categories
    )

    denominator = 1 - expected_agreement

    if denominator == 0:
        return 0.0

    return (observed_agreement - expected_agreement) / denominator


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate model means, rankings, and row-level wins."""
    model_stats = defaultdict(
        lambda: {
            "closeness": [],
            "correctness": [],
        }
    )

    row_wins = {
        metric: defaultdict(int)
        for metric in METRICS
    }

    for row in rows:
        scores = get_model_scores(row)

        if not scores:
            continue

        for model, closeness, correctness in scores:
            model_stats[model]["closeness"].append(closeness)
            model_stats[model]["correctness"].append(correctness)

        best_closeness = max(scores, key=lambda score: score[1])[0]
        best_correctness = max(scores, key=lambda score: score[2])[0]

        row_wins["closeness"][best_closeness] += 1
        row_wins["correctness"][best_correctness] += 1

    mean_per_model = {}

    for model in MODEL_ORDER:
        if model not in model_stats:
            continue

        closeness_values = model_stats[model]["closeness"]
        correctness_values = model_stats[model]["correctness"]

        mean_per_model[model] = {
            "closeness": sum(closeness_values) / len(closeness_values),
            "correctness": sum(correctness_values) / len(correctness_values),
        }

    closeness_rank = sorted(
        mean_per_model,
        key=lambda model: mean_per_model[model]["closeness"],
        reverse=True,
    )

    correctness_rank = sorted(
        mean_per_model,
        key=lambda model: mean_per_model[model]["correctness"],
        reverse=True,
    )

    overall = {
        model: (
            stats["closeness"] + stats["correctness"]
        ) / 2
        for model, stats in mean_per_model.items()
    }

    combined_rank = sorted(
        overall,
        key=overall.get,
        reverse=True,
    )

    return {
        "mean_per_model": {
            model: {
                "closeness": round(stats["closeness"], 3),
                "correctness": round(stats["correctness"], 3),
            }
            for model, stats in mean_per_model.items()
        },
        "closeness_rank": closeness_rank,
        "correctness_rank": correctness_rank,
        "combined_rank": combined_rank,
        "row_wins": {
            metric: dict(sorted(row_wins[metric].items()))
            for metric in METRICS
        },
    }


def rows_by_decision_id(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Index rows by Decision_ID."""
    result = {}

    for row in rows:
        decision_id = row.get("Decision_ID")

        if decision_id is None:
            continue

        decision_id = str(decision_id).strip()

        if decision_id:
            result[decision_id] = row

    return result


def compare_evaluators(
    evaluator_a: list[dict[str, Any]],
    evaluator_b: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare evaluator model winners using common Decision_IDs."""
    evaluator_a_by_id = rows_by_decision_id(evaluator_a)
    evaluator_b_by_id = rows_by_decision_id(evaluator_b)

    common_ids = sorted(
        set(evaluator_a_by_id) & set(evaluator_b_by_id)
    )

    labels = {
        metric: []
        for metric in METRICS
    }

    for decision_id in common_ids:
        row_a = evaluator_a_by_id[decision_id]
        row_b = evaluator_b_by_id[decision_id]

        for metric in METRICS:
            labels[metric].append(
                (
                    winner_by_metric(row_a, metric),
                    winner_by_metric(row_b, metric),
                )
            )

    agreement = {
        metric: {
            "agree": 0,
            "disagree": 0,
        }
        for metric in METRICS
    }

    kappas = {}

    for metric in METRICS:
        evaluator_a_labels = [
            pair[0]
            for pair in labels[metric]
        ]

        evaluator_b_labels = [
            pair[1]
            for pair in labels[metric]
        ]

        for label_a, label_b in zip(
            evaluator_a_labels,
            evaluator_b_labels,
        ):
            result = "agree" if label_a == label_b else "disagree"
            agreement[metric][result] += 1

        kappas[metric] = cohen_kappa(
            evaluator_a_labels,
            evaluator_b_labels,
        )

    return {
        "common_ids": len(common_ids),
        "agreement": agreement,
        "kappa": kappas,
    }


def combined_mean(
    rows_list: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """Calculate model means across multiple evaluator datasets."""
    scores = defaultdict(
        lambda: {
            "closeness": [],
            "correctness": [],
        }
    )

    for row in rows_list:
        for model, closeness, correctness in get_model_scores(row):
            scores[model]["closeness"].append(closeness)
            scores[model]["correctness"].append(correctness)

    combined = {}

    for model in MODEL_ORDER:
        if model not in scores:
            continue

        closeness_values = scores[model]["closeness"]
        correctness_values = scores[model]["correctness"]

        combined[model] = {
            "closeness": round(
                sum(closeness_values) / len(closeness_values),
                3,
            ),
            "correctness": round(
                sum(correctness_values) / len(correctness_values),
                3,
            ),
        }

    if not combined:
        return {}, {
            "closeness": 0.0,
            "correctness": 0.0,
        }

    overall = {
        metric: round(
            sum(stats[metric] for stats in combined.values())
            / len(combined),
            3,
        )
        for metric in METRICS
    }

    return combined, overall


def format_evaluator_summary(
    evaluator_number: int,
    rows: list[dict[str, Any]],
) -> str:
    """Format summary statistics for one evaluator as text."""
    summary = summarize_rows(rows)
    lines = [f"Evaluator {evaluator_number} mean:"]

    for model in MODEL_ORDER:
        stats = summary["mean_per_model"].get(model)

        if stats is None:
            continue

        lines.append(
            f"  {model}: "
            f"closeness={stats['closeness']}, "
            f"correctness={stats['correctness']}"
        )

    return "\n".join(lines)


def find_evaluator_sheets(
    base_dir: Path,
) -> list[list[dict[str, Any]]]:
    """Load evaluator sheets from matching Excel files."""
    files = sorted(base_dir.glob("CD Authors.xlsx"))

    if not files:
        raise FileNotFoundError(
            f"No Excel files found in {base_dir} "
            "matching pattern 'CD Authors.xlsx'"
        )

    evaluator_sheets = []

    for file_path in files:
        sheets = load_evaluator_rows(file_path)
        evaluator_sheets.extend(sheets.values())

    if len(evaluator_sheets) < 2:
        raise ValueError("Need at least 2 evaluator sheets.")

    return evaluator_sheets


def format_results(
    evaluator_a: list[dict[str, Any]],
    evaluator_b: list[dict[str, Any]],
) -> str:
    """Generate the complete evaluation report."""
    sections = [
        format_evaluator_summary(1, evaluator_a),
        format_evaluator_summary(2, evaluator_b),
    ]

    combined, overall = combined_mean(evaluator_a + evaluator_b)

    combined_lines = ["Combined mean (both evaluators):"]

    for model in MODEL_ORDER:
        if model not in combined:
            continue

        stats = combined[model]

        combined_lines.append(
            f"  {model}: "
            f"closeness={stats['closeness']}, "
            f"correctness={stats['correctness']}"
        )

    combined_lines.append(
        f"Overall combined mean: "
        f"closeness={overall['closeness']}, "
        f"correctness={overall['correctness']}"
    )

    sections.append("\n".join(combined_lines))

    comparison = compare_evaluators(
        evaluator_a,
        evaluator_b,
    )

    comparison_lines = [
        f"Common Decision_IDs: {comparison['common_ids']}",
        (
            "Cohen's kappa (closeness): "
            f"{comparison['kappa']['closeness']:.3f}"
        ),
        (
            "Cohen's kappa (correctness): "
            f"{comparison['kappa']['correctness']:.3f}"
        ),
    ]

    sections.append("\n".join(comparison_lines))

    return "\n\n".join(sections)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    output_file = base_dir / "results.txt"

    evaluator_sheets = find_evaluator_sheets(base_dir)

    evaluator_a = evaluator_sheets[0]
    evaluator_b = evaluator_sheets[1]

    report = format_results(
        evaluator_a,
        evaluator_b,
    )

    output_file.write_text(
        report,
        encoding="utf-8",
    )

    print(f"Results written to: {output_file}")
    
if __name__ == "__main__":
    main()