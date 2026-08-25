import pathlib
import csv
import matplotlib.pyplot as plt

from math import isfinite

def read_csv_metrics(metrics_path):
    """
    读取csv metrics
    :param metrics_path: str或pathlib.Path
    :return:
    """
    if not isinstance(metrics_path, (pathlib.Path, str)):
        raise ValueError(f"metrics_path: {metrics_path} 必须为str或pathlib.Path类型的喵! 现在的type: {type(metrics_path)}")
    if isinstance(metrics_path, str):
        if not metrics_path.strip():
            raise ValueError(f"metrics_path 为字符串时不能为空喵!")
    metrics_path = pathlib.Path(metrics_path)
    if not metrics_path.suffix.casefold() == ".csv":
        raise ValueError(f"metrics_path: {metrics_path} 必须指向csv文件喵!")
    if not metrics_path.exists():
        raise ValueError(f"metrics_path: {metrics_path} 指向的路径不存在喵!")
    if not metrics_path.is_file():
        raise ValueError(f"metrics_path: {metrics_path} 必须指向一个文件喵!")

    with open(metrics_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        current_fieldnames = reader.fieldnames
        exist_fieldnames = []
        metrics = {}
        if not current_fieldnames:
            raise ValueError(f"csv文件表头不能为空喵!")
        if "global_step" not in current_fieldnames:
            raise ValueError(f"csv文件: {metrics_path} 表头缺少了字段 global_step 喵!")
        for current_fieldname in current_fieldnames:
            if not isinstance(current_fieldname, str):
                raise ValueError(f"csv文件表头: {current_fieldname} 必须为字符串类型的喵! 现在的type: {type(current_fieldname)}")
            if not current_fieldname.strip():
                raise ValueError(f"csv文件: {metrics_path} 表头都不能为空喵!")
            if current_fieldname in exist_fieldnames:
                raise ValueError(f"csv文件: {metrics_path} 表头: {current_fieldname} 出现重复了喵!")
            if current_fieldname not in exist_fieldnames:
                exist_fieldnames.append(current_fieldname)
            if current_fieldname not in metrics:
                metrics[current_fieldname] = []
        for row in reader:
            parsed_row = {}
            for key, value in row.items():
                if key is None:
                    raise ValueError(f"csv文件: {metrics_path} 第{reader.line_num}行出现了多余值: {value} 喵!")
                if value is None:
                    raise ValueError(f"csv文件: {metrics_path} 第{reader.line_num}行的字段: {key} 缺失了喵!")
                if isinstance(value, str) and not value.strip():
                    raise ValueError(f"csv文件: {metrics_path} 第{reader.line_num}行的字段: {key} 的单元格为空喵!")
            for fieldname in current_fieldnames:
                fieldname_value = row[fieldname]
                if fieldname == "global_step":
                    try:
                        int_value = int(fieldname_value)
                    except ValueError:
                        raise ValueError(f"global_step: {fieldname_value} 转换失败了喵! csv_path: {metrics_path} line: {reader.line_num}")
                    if int_value < 0:
                        raise ValueError(f"global_step: {int_value} 不能为负数喵!")
                    parsed_row[fieldname] = int_value
                else:
                    try:
                        float_value = float(fieldname_value)
                    except ValueError:
                        raise ValueError(f"{fieldname}: {fieldname_value} 转换失败了喵! csv_path: {metrics_path} line: {reader.line_num}")
                    if not isfinite(float_value):
                        raise ValueError(f"{fieldname}: {fieldname_value} 必须为有限数值喵! csv_path: {metrics_path} line: {reader.line_num}")
                    parsed_row[fieldname] = float_value
            for fieldname in current_fieldnames:
                metrics[fieldname].append(parsed_row[fieldname])
        return metrics

def plot_csv_metrics(metrics_path, figures_dir):
    if not isinstance(metrics_path, pathlib.Path):
        raise ValueError(f"metrics_path: {metrics_path} 必须为pathlib.Path类型的喵! 现在的type: {type(metrics_path)}")
    if not metrics_path.suffix.casefold() == ".csv":
        raise ValueError(f"metrics_path: {metrics_path} 必须指向一个csv文件喵!")
    if not metrics_path.exists():
        raise ValueError(f"metrics_path: {metrics_path} 指向的路径不存在喵!")
    if not metrics_path.is_file():
        raise ValueError(f"metrics_path: {metrics_path} 必须指向一个文件喵!")
    if not isinstance(figures_dir, pathlib.Path):
        raise ValueError(f"figures_dir: {figures_dir} 必须为pathlib.Path类型的喵! 现在的type: {type(figures_dir)}")
    if not figures_dir.exists():
        raise ValueError(f"figures_dir: {figures_dir} 指向的路径不存在喵!")
    if not figures_dir.is_dir():
        raise ValueError(f"figures_dir: {figures_dir} 必须指向一个目录喵!")
    metrics = read_csv_metrics(metrics_path)
    stem = metrics_path.stem
    x = metrics["global_step"]
    if not x:
        raise ValueError(f"metrics_path: {metrics_path} 该文件没有数据行喵!")
    save_dir = figures_dir / f"{stem}"
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
    for key, value in metrics.items():
        key = key.strip()
        if "/" in key:
            key = key.replace("/", "_")
        if key == "global_step":
            continue
        else:
            y = value
            plt.figure()
            plt.plot(x, y)
            plt.title(key)
            plt.xlabel("global_step")
            plt.ylabel(key)
            plt.savefig(f"{save_dir}/{key}.png")
            plt.close()
    return None


