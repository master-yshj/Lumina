from pathlib import Path
from urllib.request import urlopen

VOCAB_URL = (
    "https://huggingface.co/google-bert/"
    "bert-base-chinese/resolve/main/vocab.txt"
)


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".tmp")

    print(f"正在下载: {url}")
    print(f"下载到目录: {destination}")

    try:
        with urlopen(url, timeout=30) as response:
            if response.status != 200:
                raise RuntimeError(f"HTTP 请求失效: {response.status}")

            with temporary_path.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)

        if temporary_path.stat().st_size == 0:
            raise RuntimeError("下载文件为空")

        temporary_path.replace(destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def main() -> None:
    current_directory = Path(__file__).resolve().parent
    destination = current_directory / "vocab.txt"

    if destination.exists() and destination.stat().st_size > 0:
        print(f"已存在: {destination}")
        return

    download_file(VOCAB_URL, destination)
    print("下载完成")


if __name__ == "__main__":
    main()