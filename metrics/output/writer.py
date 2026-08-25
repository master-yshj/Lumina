import pathlib
import csv

class MetricWriter:
    def __init__(self, metrics_path, fieldnames):
        """

        :param metrics_path: 预期csv路径
        :param fieldnames: 特定场景下的csv表头
        """
        self.metrics_path = metrics_path
        self.fieldnames = fieldnames

        if not isinstance(self.metrics_path, (str, pathlib.Path)):
            raise ValueError(f"metrics_path: {metrics_path} 必须为字符串类型或pathlib.Path类型的喵! 现在的type: {type(metrics_path)}")
        if isinstance(self.metrics_path, str):
            if not self.metrics_path.strip():
                raise ValueError(f"metrics_path 为字符串类型时不能为空喵!")
        self.metrics_path = pathlib.Path(self.metrics_path)
        if not self.metrics_path.suffix.casefold() == ".csv":
            raise ValueError(f"metrics_path: {self.metrics_path} 的后缀名必须为csv喵!")
        if not self.metrics_path.parent.exists():
            raise ValueError(f"metrics_path: {self.metrics_path} 的父目录不存在喵!")
        if not self.metrics_path.parent.is_dir():
            raise ValueError(f"metrics_path: {self.metrics_path} 的父目录必须指向一个目录喵!")
        if self.metrics_path.exists() and not self.metrics_path.is_file():
            raise ValueError(f"metrics_path: {self.metrics_path} 存在时必须指向一个文件喵!")

        if not isinstance(self.fieldnames, (list, tuple)):
            raise ValueError(f"fieldnames: {self.fieldnames} 必须为list类型或者tuple类型喵! 现在的type: {type(self.fieldnames)}")
        if not self.fieldnames:
            raise ValueError(f"fieldnames 不能为空喵!")
        if not all(isinstance(elem, str) for elem in self.fieldnames):
            raise ValueError(f"fieldnames 中的元素必须为字符串类型的喵!")
        if not all(elem.strip() for elem in self.fieldnames):
            raise ValueError(f"fieldnames 中的元素不能为空喵!")
        exist_name = []
        for elem in self.fieldnames:
            if elem not in exist_name:
                exist_name.append(elem)
            else:
                raise ValueError(f"fieldnames 不能出现重复的元素 '{elem}' 喵!")
        self.fieldnames = list(self.fieldnames)
        self._ensure_header()

    def _ensure_header(self):
        if not self.metrics_path.exists() or self.metrics_path.stat().st_size == 0:
            with open(self.metrics_path, "w", encoding="utf-8", newline="") as f:
                csv.writer(f, delimiter=",").writerow(self.fieldnames)
        else:
            with open(self.metrics_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f, delimiter=",")
                first_row = next(reader)
                if not first_row == self.fieldnames:
                    raise ValueError(f"csv文件: {self.metrics_path} 的表头"
                                     f"{first_row}"
                                     f"与预期不一致喵! "
                                     f"{self.fieldnames}")

    def write(self, metrics):
        if not isinstance(metrics, dict):
            raise ValueError(f"metrics 必须为dict类型的喵! 现在的type: {type(metrics)}")
        if not metrics:
            raise ValueError(f"metrics 不能为空喵!")
        missing_fields = set(self.fieldnames) - metrics.keys()
        extra_fields = metrics.keys() - set(self.fieldnames)
        if missing_fields:
            raise ValueError(f"metrics 中有缺失字段喵! 缺失: {missing_fields}, 预期: {self.fieldnames}")
        if extra_fields:
            raise ValueError(f"metrics 中有多余字段喵! 多余: {extra_fields}, 预期: {self.fieldnames}")
        with open(self.metrics_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames, delimiter=",")
            writer.writerow(metrics)
