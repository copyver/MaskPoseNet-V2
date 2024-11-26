import yaml
from pathlib import Path
from easydict import EasyDict as edict
from loguru import logger
from colorama import Fore, Style

def yaml_model_load(path):
    """Load a model config from a YAML file."""
    d = yaml_load(path)  # model dict
    d.yaml_file = str(path)
    return d


def yaml_load(file="base.yaml"):
    """
    Load YAML data from a file.

    Args:
        file (str, optional): File name. Default is 'base.yaml'.

    Returns:
        (dict): YAML data and file name.
    """
    assert Path(file).suffix in {".yaml", ".yml"}, f"Attempting to load non-YAML file {file} with yaml_load()"
    with open(file, errors="ignore", encoding="utf-8") as f:
        s = f.read()  # string
        data = yaml.safe_load(s) or {}  # always return a dict (yaml.safe_load() may return None for empty files)
        return edict(data)

def yaml_save(file="base.yaml", data=None, header=""):
    """
    Save YAML data to a file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        data (dict): Data to save in YAML format.
        header (str, optional): YAML header to add.

    Returns:
        (None): Data is saved to the specified file.
    """
    if data is None:
        data = {}
    file = Path(file)
    if not file.parent.exists():
        # Create parent directories if they don't exist
        file.parent.mkdir(parents=True, exist_ok=True)

    # Convert Path objects to strings
    valid_types = int, float, str, bool, list, tuple, dict, type(None)
    for k, v in data.items():
        if not isinstance(v, valid_types):
            data[k] = str(v)

    # Dump data to file in YAML format
    with open(file, "w", errors="ignore", encoding="utf-8") as f:
        if header:
            f.write(header)
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def yaml_print(configs):
    """
    打印 YAML 格式配置字典，使用 loguru 并为不同的键赋予不同颜色。

    Args:
        configs (dict): 从 YAML 文件读取的配置字典。
    """
    def _print_dict(d, level=0):
        """递归打印字典内容，支持缩进和不同颜色显示"""
        indent = "  " * level
        for key, value in d.items():
            colored_key = f"{Fore.GREEN}{key}{Style.RESET_ALL}"  # 绿色显示键
            if isinstance(value, dict):
                logger.info(f"{indent}{colored_key}:")
                _print_dict(value, level + 1)  # 递归处理子字典
            else:
                if isinstance(value, int):
                    colored_value = f"{Fore.BLUE}{value}{Style.RESET_ALL}"  # 蓝色显示整数
                elif isinstance(value, float):
                    colored_value = f"{Fore.CYAN}{value}{Style.RESET_ALL}"  # 青色显示浮点数
                elif isinstance(value, str):
                    colored_value = f"{Fore.YELLOW}{value}{Style.RESET_ALL}"  # 黄色显示字符串
                else:
                    colored_value = f"{Fore.MAGENTA}{value}{Style.RESET_ALL}"  # 紫色显示其他类型
                logger.info(f"{indent}{colored_key}: {colored_value}")

    logger.info(f"{Fore.WHITE}YAML Configuration:{Style.RESET_ALL}")
    _print_dict(configs)


if __name__ == "__main__":
    path = "../cfg/base.yaml"
    d = yaml_model_load(path)
    print(d)
