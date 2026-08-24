import argparse
import html
import json
import random
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator

DEFAULT_INPUT = Path(__file__).resolve().parent / (
    "zhwiki-latest-pages-articles.xml/zhwiki-latest-pages-articles.xml"
)
DEFAULT_FULL_OUTPUT = Path(__file__).resolve().parent / (""
                                                         "full_zhwiki_original.jsonl")
DEFAULT_BASE_OUTPUT = Path(__file__).resolve().parent / "base_zhwiki_original.jsonl"
DEFAULT_TINY_OUTPUT = Path(__file__).resolve().parent / "tiny_zhwiki.jsonl"

COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
REF_RE = re.compile(r"<ref\b[^>]*?/\s*>|<ref\b[^>]*>.*?</ref\s*>", re.I | re.DOTALL)
TABLE_RE = re.compile(r"\{\|.*?\|\}", re.DOTALL)
TEMPLATE_RE = re.compile(r"\{\{[^{}]*\}\}")
FILE_RE = re.compile(r"\[\[(?:File|Image|文件|图像):.*?\]\]", re.I | re.DOTALL)
CATEGORY_RE = re.compile(r"\[\[(?:Category|分类):.*?\]\]", re.I)
INTERNAL_LINK_RE = re.compile(r"\[\[(?:[^\]|]*\|)?([^\]]+)\]\]")
EXTERNAL_LINK_RE = re.compile(r"\[(?:https?://|//)\S+(?:\s+([^\]]+))?\]")
TAG_RE = re.compile(r"<[^>]+>")
HEADING_RE = re.compile(r"^\s*=+\s*(.*?)\s*=+\s*$", re.MULTILINE)
FORMAT_MARK_RE = re.compile(r"'{2,5}")
SPACE_RE = re.compile(r"[ \t\f\v]+")
BLANK_LINES_RE = re.compile(r"\n{3,}")


def remove_balanced_templates(text: str) -> str:
    previous = None
    while previous != text:
        previous = text
        text = TEMPLATE_RE.sub("", text)
    return text


def clean_wikitext(text: str) -> str:
    text = COMMENT_RE.sub("", text)
    text = REF_RE.sub("", text)
    text = TABLE_RE.sub("", text)
    text = remove_balanced_templates(text)
    text = FILE_RE.sub("", text)
    text = CATEGORY_RE.sub("", text)
    text = INTERNAL_LINK_RE.sub(r"\1", text)
    text = EXTERNAL_LINK_RE.sub(lambda match: match.group(1) or "", text)
    text = HEADING_RE.sub(r"\n\1\n", text)
    text = TAG_RE.sub("", text)
    text = FORMAT_MARK_RE.sub("", text)
    text = html.unescape(text).replace("\r\n", "\n").replace("\r", "\n")

    lines = []
    for line in text.splitlines():
        line = SPACE_RE.sub(" ", line).strip()
        if line and not line.startswith(("{|", "|}", "|-", "!", "|")):
            lines.append(line)
    return BLANK_LINES_RE.sub("\n\n", "\n".join(lines)).strip()


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def child_text(element: ET.Element, name: str) -> str | None:
    for child in element:
        if local_name(child.tag) == name:
            return child.text
    return None


def revision_text(page: ET.Element) -> str:
    for child in page:
        if local_name(child.tag) != "revision":
            continue
        for revision_child in child:
            if local_name(revision_child.tag) == "text":
                return revision_child.text or ""
    return ""


def iter_articles(path: Path, min_chars: int) -> Iterator[dict[str, str]]:
    context = ET.iterparse(path, events=("end",))
    for _, element in context:
        if local_name(element.tag) != "page":
            continue

        title = child_text(element, "title") or ""
        namespace = child_text(element, "ns")
        page_id = child_text(element, "id") or ""
        raw_text = revision_text(element)

        if namespace == "0" and raw_text and not raw_text.lstrip().lower().startswith("#redirect"):
            cleaned_text = clean_wikitext(raw_text)
            if len(cleaned_text) >= min_chars:
                yield {
                    "id": f"zhwiki-{page_id}",
                    "title": title,
                    "text": cleaned_text,
                    "source": "zhwiki",
                }
        element.clear()


def write_jsonl_record(file, record: dict[str, str]) -> None:
    file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def build_datasets(
    input_path: Path,
    full_output: Path,
    base_output: Path,
    tiny_output: Path,
    base_size: int,
    tiny_size: int,
    min_chars: int,
    seed: int,
) -> tuple[int, int, int]:
    if not input_path.is_file():
        raise FileNotFoundError(f"找不到解压后的维基百科 XML：{input_path}")
    if not 0 < tiny_size <= base_size or min_chars <= 0:
        raise ValueError("必须满足 0 < tiny-size <= base-size，且 min-chars 为正整数")

    for output in (full_output, base_output, tiny_output):
        output.parent.mkdir(parents=True, exist_ok=True)
    full_temporary = full_output.with_suffix(full_output.suffix + ".tmp")
    base_temporary = base_output.with_suffix(base_output.suffix + ".tmp")
    tiny_temporary = tiny_output.with_suffix(tiny_output.suffix + ".tmp")
    random_generator = random.Random(seed)
    reservoir: list[dict[str, str]] = []
    count = 0

    try:
        with full_temporary.open("w", encoding="utf-8") as full_file:
            for count, record in enumerate(iter_articles(input_path, min_chars), start=1):
                write_jsonl_record(full_file, record)
                if len(reservoir) < base_size:
                    reservoir.append(record)
                else:
                    replacement = random_generator.randrange(count)
                    if replacement < base_size:
                        reservoir[replacement] = record
                if count % 10_000 == 0:
                    print(f"\r已处理 {count:,} 篇有效文章", end="", flush=True)

        if count < base_size:
            raise RuntimeError(f"有效文章仅 {count:,} 篇，少于 base-size={base_size:,}")

        reservoir.sort(key=lambda item: item["id"])
        tiny_ids = {item["id"] for item in random_generator.sample(reservoir, tiny_size)}
        with (
            base_temporary.open("w", encoding="utf-8") as base_file,
            tiny_temporary.open("w", encoding="utf-8") as tiny_file,
        ):
            for record in reservoir:
                write_jsonl_record(base_file, record)
                if record["id"] in tiny_ids:
                    write_jsonl_record(tiny_file, record)

        full_temporary.replace(full_output)
        base_temporary.replace(base_output)
        tiny_temporary.replace(tiny_output)
    except Exception:
        full_temporary.unlink(missing_ok=True)
        base_temporary.unlink(missing_ok=True)
        tiny_temporary.unlink(missing_ok=True)
        raise

    return count, len(reservoir), len(tiny_ids)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="清洗中文维基百科并生成全量与 Tiny MLM 语料")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="解压后的 XML 文件")
    parser.add_argument("--full-output", type=Path, default=DEFAULT_FULL_OUTPUT)
    parser.add_argument("--base-output", type=Path, default=DEFAULT_BASE_OUTPUT)
    parser.add_argument("--tiny-output", type=Path, default=DEFAULT_TINY_OUTPUT)
    parser.add_argument("--base-size", type=int, default=50_000, help="正式教学预训练集文章数")
    parser.add_argument("--tiny-size", type=int, default=5_000, help="Tiny 调试集文章数")
    parser.add_argument("--min-chars", type=int, default=100, help="保留文章的最少清洗后字符数")
    parser.add_argument("--seed", type=int, default=42, help="分层数据集的随机采样种子")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    full_count, base_count, tiny_count = build_datasets(
        input_path=args.input,
        full_output=args.full_output,
        base_output=args.base_output,
        tiny_output=args.tiny_output,
        base_size=args.base_size,
        tiny_size=args.tiny_size,
        min_chars=args.min_chars,
        seed=args.seed,
    )
    print(
        f"\n完成：full_zhwiki 共 {full_count:,} 篇，"
        f"base_zhwiki 共 {base_count:,} 篇，tiny_zhwiki 共 {tiny_count:,} 篇"
    )
    print(f"全量数据：{args.full_output.resolve()}")
    print(f"教学数据：{args.base_output.resolve()}")
    print(f"Tiny 数据：{args.tiny_output.resolve()}")


if __name__ == "__main__":
    main()
