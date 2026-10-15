
import json
from pathlib import Path

EXPECTED_SIZES = {"train": 9999, "valid": 999, "test": 999}
SPLITS = ("train", "valid", "test")


def load_dataset(file_path):

    data = []

    with Path(file_path).open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                record = json.loads(line)
                data.append(record)

    return data

VALID_LABELS = {"smile", "ok", "cry"}

def validate_labels(data):
    for i, record in enumerate(data):
        if "sentence" not in record or "label" not in record:
            raise ValueError(f"Record {i} is missing a required field")

        if not record["sentence"] or not record["sentence"].strip():
            raise ValueError(f"Record {i} has an empty sentence")

        if record["label"] not in VALID_LABELS:
            raise ValueError(
                f"Record {i} has an invalid label: {record['label']}"
            )

def validate_split_size(split_name, data):
    expected = EXPECTED_SIZES[split_name]
    actual = len(data)

    if actual != expected:
        raise ValueError(
            f"{split_name} has {actual} records; expected {expected}"
        )

if __name__ == "__main__":
    print("Starting data loader...")

    for split in SPLITS:
        file_path = f"{split}.jsonl"
        print(f"Loading {file_path}...")

        data = load_dataset(file_path)
        validate_labels(data)
        validate_split_size(split, data)

        print(f"{split}: {len(data)} records — OK")

    print("Data loading complete!")