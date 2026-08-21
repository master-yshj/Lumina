# NLP 教学训练框架

> **AI 行动最高优先级：** 用户说“下一步”“继续”“开始”“进入下一步”等推进性表达时，只授权 AI 在对话中给出指导、审查和下一最小步骤，**绝不授权修改代码、配置、数据或文档，也不授权启动训练、测试或命令**。只有用户明确要求“修改/实现/写入文件/运行”及其具体范围时，AI 才能执行对应操作；有歧义时一律保持只读。

用于电子科技大学凌睿工作室 ML 招新 NLP 题的教学训练框架。

## 推荐运行环境

本项目统一使用 **Python 3.11.14**。不建议使用其他 Python 版本，以避免依赖和运行结果出现差异。

推荐在 Linux 环境中完成本题：

- Windows 用户：推荐安装 WSL2，在 WSL2 的 Linux 发行版内安装 Miniconda；
- Linux 用户：安装 Miniconda；
- 不使用 WSL 的 Windows 用户：可以安装 Anaconda，但部分 Linux 命令和多卡行为可能不同；
- 不建议直接使用系统 Python，以免污染系统环境或产生依赖冲突。

Miniconda 下载地址：[Miniconda 官方文档](https://docs.conda.io/projects/miniconda/en/latest/)。WSL 安装方式见 [Microsoft WSL 官方文档](https://learn.microsoft.com/windows/wsl/install)。

## 创建环境

安装 Miniconda 或 Anaconda 后，创建名为 `NLP` 的独立环境：

```bash
conda create -n NLP python=3.11.14 pip -y
conda activate NLP
python --version
```

`python --version` 应输出 `Python 3.11.14`。如果首次执行 `conda activate` 失败，请先运行 `conda init`，关闭并重新打开终端。

## 安装依赖

本项目使用 pip 在 Conda 环境内安装 `requirements.txt` 中固定的直接依赖。只使用正式稳定版本，不使用 PyTorch Nightly 或 alpha、beta、rc、dev 等预发布版本。

```bash
conda activate NLP
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

请在仓库根目录执行上述命令；也可以将 `requirements.txt` 替换为该文件的绝对路径。

PyTorch 的 CUDA 可用性取决于机器上的 NVIDIA 驱动和所安装的官方稳定版 Wheel。安装后可执行：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
```

若默认安装的 PyTorch Wheel 不适合服务器环境，应根据 [PyTorch 官方安装页面](https://pytorch.org/get-started/locally/) 选择与驱动兼容的**稳定版**命令，禁止使用 Nightly 或带有 `--pre` 的安装命令。