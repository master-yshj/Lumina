from pathlib import Path
import hashlib
import shutil
import sys

# ===== 按需修改这里 =====
SOURCE_ROOT = Path(r"/home/hzy/NLP")
DEST_ROOT = Path(r"/home/hzy/lumina_for_freshmen")

# 排除的目录名（整棵目录跳过）
EXCLUDE_DIR_NAMES = {".git"}

# 排除的文件名（任何层级命中都跳过）
EXCLUDE_FILE_NAMES = {
    ".gitignore",
    ".gitattributes",
    ".gitmodules",
    ".gitkeep",
}

CHUNK_SIZE = 1024 * 1024


def is_excluded_dir(path: Path) -> bool:
    return path.name in EXCLUDE_DIR_NAMES


def is_excluded_file(path: Path) -> bool:
    name = path.name
    return (
        name in EXCLUDE_FILE_NAMES
        or name == "Zone.Identifier"
        or ":Zone.Identifier" in name
        or name.endswith("Zone.Identifier")
    )


def iter_source_files(src_root: Path):
    """按确定顺序产出源树中所有需要复制的文件（相对路径）。"""
    for path in sorted(src_root.rglob("*")):
        if path.is_dir():
            continue
        # 跳过位于被排除目录下的文件
        if any(is_excluded_dir(p) for p in path.parents):
            continue
        if is_excluded_file(path):
            continue
        yield path.relative_to(src_root), path


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def copy_all(src_root: Path, dst_root: Path) -> None:
    copied = 0
    for rel, src in iter_source_files(src_root):
        dst = dst_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
        if copied % 500 == 0:
            print(f"[进度] 已复制 {copied} 个文件...")
    print(f"[复制完成] 共复制 {copied} 个文件")


def verify_all(src_root: Path, dst_root: Path) -> int:
    src_map = {rel: src for rel, src in iter_source_files(src_root)}

    errors = []

    # 1. 文件集合必须一致
    dst_files = {
        p.relative_to(dst_root)
        for p in dst_root.rglob("*")
        if p.is_file() and not is_excluded_file(p)
    }
    missing = set(src_map) - dst_files
    extra = dst_files - set(src_map)
    if missing:
        errors.append(f"目标缺少 {len(missing)} 个文件，例如: {sorted(missing)[:5]}")
    if extra:
        errors.append(f"目标多出 {len(extra)} 个文件，例如: {sorted(extra)[:5]}")

    # 2. 逐文件校验大小 + SHA-256
    checked = 0
    for rel, src in src_map.items():
        dst = dst_root / rel
        if not dst.exists():
            continue
        if src.stat().st_size != dst.stat().st_size:
            errors.append(f"大小不一致: {rel}")
            continue
        if sha256_of(src) != sha256_of(dst):
            errors.append(f"内容不一致(SHA-256): {rel}")
            continue
        checked += 1
        if checked % 500 == 0:
            print(f"[校验进度] 已校验 {checked} 个文件...")

    # 3. 确认目标没有 Zone.Identifier
    zone = [p for p in dst_root.rglob("*") if is_excluded_file(p)]
    if zone:
        errors.append(f"目标存在 Zone.Identifier 文件 {len(zone)} 个")

    print(f"[校验完成] 成功校验 {checked}/{len(src_map)} 个文件")
    if errors:
        print("校验失败：")
        for e in errors:
            print(f"  - {e}")
        return 1

    print("校验通过：文件集合一致、大小一致、SHA-256 全部一致、无 Zone.Identifier")
    return 0


def main() -> int:
    if not SOURCE_ROOT.is_dir():
        print(f"源目录不存在: {SOURCE_ROOT}")
        return 1
    if DEST_ROOT.exists() and any(DEST_ROOT.iterdir()):
        print(f"目标目录非空，为安全起见请先清空或换一个目标: {DEST_ROOT}")
        return 1
    DEST_ROOT.mkdir(parents=True, exist_ok=True)

    copy_all(SOURCE_ROOT, DEST_ROOT)
    return verify_all(SOURCE_ROOT, DEST_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())