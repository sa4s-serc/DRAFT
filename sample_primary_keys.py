import argparse
import json
import random
from pathlib import Path


def extract_primary_keys(path: Path):
    primary_keys = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            for key_name in ("PrimaryKey", "primary_key", "primaryKey", "id", "ID", "pk"):
                if key_name in row:
                    primary_keys.append(row[key_name])
                    break
            else:
                raise KeyError(f"No primary key field found in {path} at line {line_number}")

    return primary_keys


def main():
    parser = argparse.ArgumentParser(description="Read a JSONL file, collect primary keys, and sample 64 of them")
    parser.add_argument("input_file", nargs="?", help="Path to a JSONL file")
    args = parser.parse_args()

    if args.input_file:
        input_path = Path(args.input_file).expanduser().resolve()
    else:
        default_dir = Path(__file__).resolve().parent / "DRAFT" / "Results"
        input_path = next(sorted(default_dir.glob("*.jsonl")), None)
        if input_path is None:
            raise FileNotFoundError(f"No JSONL files found in {default_dir}")

    primary_keys = extract_primary_keys(input_path)
    random.seed(42)
    sampled_keys = random.sample(primary_keys, 64)

    print(f"File: {input_path}")
    print(f"Rows: {len(primary_keys)}")
    print("Primary keys:")
    print(primary_keys)
    print("\nSampled 64 primary keys:")
    print(sampled_keys)


if __name__ == "__main__":
    main()
