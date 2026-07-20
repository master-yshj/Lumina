import json
from pathlib import Path
from urllib.request import Request, urlopen

BASE_URL = "https://raw.githubusercontent.com/ymcui/cmrc2018/master/squad-style-data"
FILES = {
    "train": "cmrc2018_train.json",
    "dev": "cmrc2018_dev.json",
    "trial": "cmrc2018_trial.json",
}
CHUNK_SIZE = 1024 * 1024


def validate_dataset(path: Path) -> None:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ValueError(f"{path.name} 不是有效的 SQuAD 格式数据集")
    if not payload["data"]:
        raise ValueError(f"{path.name} 不包含任何样本")


def download_file(url: str, destination: Path) -> None:
    temporary_path = destination.with_suffix(destination.suffix + ".tmp")
    request = Request(url, headers={"User-Agent": "NLP-Course-Dataset-Downloader/1.0"})

    print(f"正在下载: {url}")
    try:
        with urlopen(request, timeout=60) as response, temporary_path.open("wb") as output:
            if getattr(response, "status", response.getcode()) != 200:
                raise RuntimeError(f"HTTP 请求失败: {response.status}")
            while chunk := response.read(CHUNK_SIZE):
                output.write(chunk)

        if temporary_path.stat().st_size == 0:
            raise RuntimeError("下载文件为空")
        validate_dataset(temporary_path)
        temporary_path.replace(destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def main() -> None:
    output_directory = Path(__file__).resolve().parent
    output_directory.mkdir(parents=True, exist_ok=True)

    for split, filename in FILES.items():
        destination = output_directory / filename
        if destination.exists() and destination.stat().st_size > 0:
            try:
                validate_dataset(destination)
                print(f"{split} 数据已存在且校验通过: {destination}")
                continue
            except (OSError, ValueError, json.JSONDecodeError):
                print(f"{destination} 无效，将重新下载")

        download_file(f"{BASE_URL}/{filename}", destination)
        print(f"{split} 下载并校验完成: {destination}")

    print("CMRC 2018 数据集下载完成。官方测试集不公开答案，因此这里只下载 train、dev 和 trial。")


if __name__ == "__main__":
    main()
