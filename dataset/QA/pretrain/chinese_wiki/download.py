from pathlib import Path
from urllib.request import Request, urlopen

DUMP_URL = "https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2"
CHUNK_SIZE = 1024 * 1024


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".tmp")
    downloaded_size = temporary_path.stat().st_size if temporary_path.exists() else 0

    request = Request(url, headers={"User-Agent": "NLP-Course-Dataset-Downloader/1.0"})
    if downloaded_size:
        request.add_header("Range", f"bytes={downloaded_size}-")

    print(f"正在下载: {url}")
    print(f"保存到: {destination}")
    if downloaded_size:
        print(f"尝试从 {downloaded_size} 字节处继续下载")

    try:
        with urlopen(request, timeout=60) as response:
            status = getattr(response, "status", response.getcode())
            resumed = downloaded_size > 0 and status == 206
            mode = "ab" if resumed else "wb"
            if downloaded_size and not resumed:
                print("服务器不支持断点续传，将重新下载")

            total_header = response.headers.get("Content-Length")
            response_size = int(total_header) if total_header else None
            expected_size = downloaded_size + response_size if resumed and response_size else response_size
            written_size = downloaded_size if resumed else 0

            with temporary_path.open(mode) as output:
                while chunk := response.read(CHUNK_SIZE):
                    output.write(chunk)
                    written_size += len(chunk)
                    if expected_size:
                        percent = written_size / expected_size * 100
                        print(
                            f"\r已下载 {written_size / 1024**2:.1f} MiB / "
                            f"{expected_size / 1024**2:.1f} MiB ({percent:.1f}%)",
                            end="",
                            flush=True,
                        )
                    else:
                        print(
                            f"\r已下载 {written_size / 1024**2:.1f} MiB",
                            end="",
                            flush=True,
                        )

        print()
        if temporary_path.stat().st_size == 0:
            raise RuntimeError("下载文件为空")
        if expected_size and temporary_path.stat().st_size != expected_size:
            raise RuntimeError(
                f"文件大小不完整：期望 {expected_size} 字节，实际 "
                f"{temporary_path.stat().st_size} 字节"
            )
        temporary_path.replace(destination)
    except Exception:
        print(f"下载中断，临时文件已保留，可再次运行脚本续传: {temporary_path}")
        raise


def main() -> None:
    destination = Path(__file__).resolve().parent / "zhwiki-latest-pages-articles.xml.bz2"
    if destination.exists() and destination.stat().st_size > 0:
        print(f"数据集已存在: {destination}")
        return

    download_file(DUMP_URL, destination)
    print("中文维基百科原始 Dump 下载完成。后续仍需解析、清洗并抽取训练子集。")


if __name__ == "__main__":
    main()
