import yaml
from pathlib import Path
from easydict import EasyDict as edict
from loguru import logger
from colorama import Fore, Style


def colorstr(*input):
    r"""
    Colors a string based on the provided color and style arguments. Utilizes ANSI escape codes.
    See https://en.wikipedia.org/wiki/ANSI_escape_code for more details.

    This function can be called in two ways:
        - colorstr('color', 'style', 'your string')
        - colorstr('your string')

    In the second form, 'blue' and 'bold' will be applied by default.

    Args:
        *input (str | Path): A sequence of strings where the first n-1 strings are color and style arguments,
                      and the last string is the one to be colored.

    Supported Colors and Styles:
        Basic Colors: 'black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'
        Bright Colors: 'bright_black', 'bright_red', 'bright_green', 'bright_yellow',
                       'bright_blue', 'bright_magenta', 'bright_cyan', 'bright_white'
        Misc: 'end', 'bold', 'underline'

    Returns:
        (str): The input string wrapped with ANSI escape codes for the specified color and style.

    Examples:
        >>> colorstr("blue", "bold", "hello world")
        >>> "\033[34m\033[1mhello world\033[0m"
    """
    *args, string = input if len(input) > 1 else ("blue", "bold", input[0])  # color arguments, string
    colors = {
        "black": "\033[30m",  # basic colors
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "bright_black": "\033[90m",  # bright colors
        "bright_red": "\033[91m",
        "bright_green": "\033[92m",
        "bright_yellow": "\033[93m",
        "bright_blue": "\033[94m",
        "bright_magenta": "\033[95m",
        "bright_cyan": "\033[96m",
        "bright_white": "\033[97m",
        "end": "\033[0m",  # misc
        "bold": "\033[1m",
        "underline": "\033[4m",
    }
    return "".join(colors[x] for x in args) + f"{string}" + colors["end"]

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
