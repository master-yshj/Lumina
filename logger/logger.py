import pathlib

from datetime import datetime
from copy import deepcopy

KAOMOJI_ART = r"""

          /\                                /\
         /  \______________________________/  \
        / /\                                /\ \
       / /  \           |   |   |          /  \ \
      / /    \          |   |   |         /    \ \
     | |                |   |   |               | |
     | |                                        | |
     | |             ●                 <        | |
     | |                        ω               | |
     | |           ///                 ///      | |
      \ \                                      / /
       \ \____________________________________/ /
        \______________________________________/

""".strip("\n")

FRAMEWORK_NAME_ART = r"""

#        #     #  #     #  #######  #     #     ###
#        #     #  ##   ##     #     ##    #    #   #
#        #     #  # # # #     #     # #   #   #     #
#        #     #  #  #  #     #     #  #  #   #######
#        #     #  #     #     #     #   # #   #     #
#        #     #  #     #     #     #    ##   #     #
#######   #####   #     #  #######  #     #   #     #

""".strip("\n")

LINGRUI_STUDIO_ART = r"""

#        #######  #     #   #####   ######   #     #  #######
#           #     ##    #  #     #  #     #  #     #     #
#           #     # #   #  #        #     #  #     #     #
#           #     #  #  #  #  ####  ######   #     #     #
#           #     #   # #  #     #  #   #    #     #     #
#           #     #    ##  #     #  #    #   #     #     #
#######  #######  #     #   #####   #     #   #####   #######
          #####   #######  #     #  ######   #######   #####
         #           #     #     #  #     #     #     #     #
         #           #     #     #  #     #     #     #     #
          #####      #     #     #  #     #     #     #     #
               #     #     #     #  #     #     #     #     #
               #     #     #     #  #     #     #     #     #
          #####      #      #####   ######   #######   #####

""".strip("\n")

WATERMARK = "\n\n".join(
    (
        KAOMOJI_ART,
        FRAMEWORK_NAME_ART,
        LINGRUI_STUDIO_ART,
    )
)

class RunLogger:
    def __init__(self, log_path):

        if not isinstance(log_path, (str, pathlib.Path)):
            raise ValueError(f"log_path: {log_path} 只能为字符串类型或pathlib.Path类型的喵! 现在的type: {type(log_path)}")
        if isinstance(log_path, str):
            if not log_path.strip():
                raise ValueError(f"log_path 不能为空喵!")
        log_path = pathlib.Path(log_path)
        if not log_path.suffix.casefold() == ".log":
            raise ValueError(f"log_path 指向的文件后缀名必须为log喵!")
        if not log_path.parent.exists():
            raise ValueError(f"log_path: {log_path} 的父目录指向的路径不存在喵!")
        if not log_path.parent.is_dir():
            raise ValueError(f"log_path: {log_path} 的父目录必须指向一个目录喵!")
        if log_path.exists() and not log_path.is_file():
            raise ValueError(f"log_path: {log_path} 存在时没有指向一个文件喵!")

        self.log_path = log_path

    def _write(self, text):
        if not isinstance(text, str):
            raise ValueError(f"输入的 text 必须为字符串类型的喵!")
        if not text.strip():
            raise ValueError(f"输入的 text 不能为空喵!")
        if not text.endswith("\n"):
            normalized_text = text + "\n"
        else:
            normalized_text = text
        with open(self.log_path, "a", encoding="utf-8") as f:
            print(normalized_text, end="")
            f.write(normalized_text)

    def write_banner(self):
        self._write(WATERMARK)

    def info(self, message):
        if not isinstance(message, str):
            raise ValueError(f"message: {message} 必须为字符串类型的喵! 现在的type: {type(message)}")
        if not message.strip():
            raise ValueError(f"message 不能为空喵!")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        self._write(formatted_message)

    def write_run_info(self, run_info):
        if not isinstance(run_info, dict):
            raise ValueError(f"run_info 必须为dict类型的喵! 现在的type: {type(run_info)}")
        if not run_info:
            raise ValueError(f"run_info 不能为空喵!")
        copy_info = deepcopy(run_info)
        for key in run_info:
            if not isinstance(key, str):
                raise ValueError(f"run_info.{key} 必须为字符串类型的喵! 现在的type: {type(run_info[key])}")
            if not key.strip():
                raise ValueError(f"run_info.{key} 不能为空喵!")
            if run_info[key] is None:
                copy_info[key] = "-"
            else:
                copy_info[key] = str(run_info[key])
        max_key_length = max(len(key) for key in run_info)

        lines = ["-" * 60]

        for key, value in copy_info.items():
            lines.append(f"{key:<{max_key_length}} : {value}")

        lines.append("-" * 60)

        run_info_text = "\n".join(lines)
        self._write(run_info_text)

    def start(self, run_info):
        if not isinstance(run_info, dict):
            raise ValueError(f"run_info 必须为dict类型的喵! 现在的type: {type(run_info)}")
        if not run_info:
            raise ValueError("run_info 不能为空喵!")
        if hasattr(self, "started_at"):
            raise ValueError("RunLogger 已经启动过了喵!")
        self.started_at = datetime.now()
        self.run_id = self.started_at.strftime("%Y%m%d-%H%M%S-%f")
        self.mode = run_info.get("运行模式")
        start_info = deepcopy(run_info)
        start_info["运行编号"] = self.run_id
        start_info["开始时间"] = self.started_at.strftime("%Y-%m-%d %H:%M:%S")
        self.write_banner()
        self.write_run_info(start_info)

    def finish(self, run_summary):
        if not isinstance(run_summary, dict):
            raise ValueError(f"传入的run_summary必须为dict类型的喵! 现在的type: {type(run_summary)}")
        if not run_summary:
            raise ValueError(f"传入的run_summary不能为空喵!")
        if not hasattr(self, "started_at"):
            raise ValueError(f"logger还没有启动喵!")
        if hasattr(self, "ended_at"):
            raise ValueError(f"logger已经结束了喵!")
        self.ended_at = datetime.now()
        self.elapsed_time = self.ended_at - self.started_at
        end_info = deepcopy(run_summary)
        end_info["运行编号"] = self.run_id
        end_info["结束时间"] = self.ended_at.strftime("%Y-%m-%d %H:%M:%S")
        end_info["运行耗时"] = self.elapsed_time
        self.write_run_info(end_info)