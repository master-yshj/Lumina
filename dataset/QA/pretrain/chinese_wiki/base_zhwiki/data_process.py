import argparse
import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = DATA_DIR / "base_zhwiki_original.jsonl"
DEFAULT_TRAIN_OUTPUT = DATA_DIR / "base_zhwiki_train.jsonl"
DEFAULT_VAL_OUTPUT = DATA_DIR / "base_zhwiki_val.jsonl"
DEFAULT_TEST_OUTPUT = DATA_DIR / "base_zhwiki_test.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"找不到原始数据集：{path}")

    records = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"第 {line_number} 行不是合法 JSON：{error}") from error
            if not isinstance(record, dict):
                raise ValueError(f"第 {line_number} 行必须是 JSON 对象")
            records.append(record)

    if not records:
        raise ValueError(f"原始数据集为空：{path}")
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        temporary_path.replace(path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def split_dataset(
    records: list[dict], seed: int
) -> tuple[list[dict], list[dict], list[dict]]:
    shuffled_records = records.copy()
    random.Random(seed).shuffle(shuffled_records)

    total = len(shuffled_records)
    train_end = total * 8 // 10
    val_end = train_end + total // 10
    return (
        shuffled_records[:train_end],
        shuffled_records[train_end:val_end],
        shuffled_records[val_end:],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按 8:1:1 划分 Base 中文维基数据集")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--train-output", type=Path, default=DEFAULT_TRAIN_OUTPUT)
    parser.add_argument("--val-output", type=Path, default=DEFAULT_VAL_OUTPUT)
    parser.add_argument("--test-output", type=Path, default=DEFAULT_TEST_OUTPUT)
    parser.add_argument("--seed", type=int, default=42, help="随机划分种子")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_jsonl(args.input)
    train_records, val_records, test_records = split_dataset(records, args.seed)

    write_jsonl(args.train_output, train_records)
    write_jsonl(args.val_output, val_records)
    write_jsonl(args.test_output, test_records)

    print(
        f"划分完成：总计 {len(records):,} 条，"
        f"训练集 {len(train_records):,} 条，"
        f"验证集 {len(val_records):,} 条，"
        f"测试集 {len(test_records):,} 条"
    )
    print(f"训练集：{args.train_output.resolve()}")
    print(f"验证集：{args.val_output.resolve()}")
    print(f"测试集：{args.test_output.resolve()}")


if __name__ == "__main__":
    main()
