import yaml
from pathlib import Path
from easydict import EasyDict as edict
from loguru import logger
import re
import platform
import os
import inspect
from types import SimpleNamespace


__version__ = "1.3.0"
MACOS, LINUX, WINDOWS = (platform.system() == x for x in ["Darwin", "Linux", "Windows"])
PIN_MEMORY = str(os.getenv("PIN_MEMORY", True)).lower() == "true"
NUM_THREADS = min(8, max(1, os.cpu_count() - 1))
RANK = int(os.getenv("RANK", -1))
LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))


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


def remove_colors(message):
    """
    移除日志消息中的 ANSI 转义字符，防止颜色转义码被写入日志文件。

    Args:
        message (str): 日志消息。

    Returns:
        str: 去除颜色转义字符后的日志消息。
    """
    # 匹配 ANSI 转义字符的正则表达式
    ansi_escape = re.compile(r'\033\[[0-9;]*m')
    return ansi_escape.sub('', message)


def logger_format_function(record):
    """
`   格式化函数，移除 ANSI 转义字符。

    Args:
        record (dict): Loguru 的日志记录。

    Returns:
        str: 格式化后的日志字符串。
    """
    record["message"] = remove_colors(record["message"])  # 移除颜色转义字符
    return "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}\n".format(**record)


def yaml_model_load(path):
    """
    Load a model config from a YAML file.
    """
    dict_ = yaml_load(path)  # model dict
    dict_.YAML_FILE = str(path)
    return dict_


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


def serialize_data(data):
    """
    Recursively convert unsupported data types into serializable formats.
    """
    if isinstance(data, (int, float, str, bool, type(None))):
        return data
    elif isinstance(data, (list, tuple)):
        return [serialize_data(item) for item in data]
    elif isinstance(data, dict):
        return {key: serialize_data(value) for key, value in data.items()}
    else:
        # Convert unsupported types to strings
        return str(data)


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
    # valid_types = int, float, str, bool, list, tuple, dict, type(None)
    # for k, v in data.items():
    #     if not isinstance(v, valid_types):
    #         data[k] = str(v)
    data = serialize_data(data)

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
            colored_key = colorstr("green", "bold", key)  # 绿色加粗显示键
            if isinstance(value, dict):
                logger.info(f"{indent}{colored_key}:")
                _print_dict(value, level + 1)  # 递归处理子字典
            else:
                if isinstance(value, bool):
                    colored_value = colorstr("red", str(value))  # 红色显示布尔值
                elif isinstance(value, int):
                    colored_value = colorstr("blue", str(value))  # 蓝色显示整数
                elif isinstance(value, float):
                    colored_value = colorstr("cyan", str(value))  # 青色显示浮点数
                elif isinstance(value, str):
                    colored_value = colorstr("yellow", value)  # 黄色显示字符串
                else:
                    colored_value = colorstr("magenta", str(value))  # 紫色显示其他类型
                logger.info(f"{indent}{colored_key}: {colored_value}")

    logger.info(colorstr("white", "bold", "YAML Configuration:"))
    _print_dict(configs)


def get_default_args(func):
    """
    Returns a dictionary of default arguments for a function.

    Args:
        func (callable): The function to inspect.

    Returns:
        (dict): A dictionary where each key is a parameter name, and each value is the default value of that parameter.
    """
    signature = inspect.signature(func)
    return {k: v.default for k, v in signature.parameters.items() if v.default is not inspect.Parameter.empty}


class SimpleClass:
    """
    A simple base class for creating objects with string representations of their attributes.

    This class provides a foundation for creating objects that can be easily printed or represented as strings,
    showing all their non-callable attributes. It's useful for debugging and introspection of object states.

    Methods:
        __str__: Returns a human-readable string representation of the object.
        __repr__: Returns a machine-readable string representation of the object.
        __getattr__: Provides a custom attribute access error message with helpful information.

    Examples:
        >>> class MyClass(SimpleClass):
        ...     def __init__(self):
        ...         self.x = 10
        ...         self.y = "hello"
        >>> obj = MyClass()
        >>> print(obj)
        __main__.MyClass object with attributes:

        x: 10
        y: 'hello'

    Notes:
        - This class is designed to be subclassed. It provides a convenient way to inspect object attributes.
        - The string representation includes the module and class name of the object.
        - Callable attributes and attributes starting with an underscore are excluded from the string representation.
    """

    def __str__(self):
        """Return a human-readable string representation of the object."""
        attr = []
        for a in dir(self):
            v = getattr(self, a)
            if not callable(v) and not a.startswith("_"):
                if isinstance(v, SimpleClass):
                    # Display only the module and class name for subclasses
                    s = f"{a}: {v.__module__}.{v.__class__.__name__} object"
                else:
                    s = f"{a}: {repr(v)}"
                attr.append(s)
        return f"{self.__module__}.{self.__class__.__name__} object with attributes:\n\n" + "\n".join(attr)

    def __repr__(self):
        """Return a machine-readable string representation of the object."""
        return self.__str__()

    def __getattr__(self, attr):
        """Custom attribute access error message with helpful information."""
        name = self.__class__.__name__
        raise AttributeError(f"'{name}' object has no attribute '{attr}'. See valid attributes below.\n{self.__doc__}")


class IterableSimpleNamespace(SimpleNamespace):
    """
    An iterable SimpleNamespace class that provides enhanced functionality for attribute access and iteration.

    This class extends the SimpleNamespace class with additional methods for iteration, string representation,
    and attribute access. It is designed to be used as a convenient container for storing and accessing
    configuration parameters.

    Methods:
        __iter__: Returns an iterator of key-value pairs from the namespace's attributes.
        __str__: Returns a human-readable string representation of the object.
        __getattr__: Provides a custom attribute access error message with helpful information.
        get: Retrieves the value of a specified key, or a default value if the key doesn't exist.

    Examples:
        >>> cfg = IterableSimpleNamespace(a=1, b=2, c=3)
        >>> for k, v in cfg:
        ...     print(f"{k}: {v}")
        a: 1
        b: 2
        c: 3
        >>> print(cfg)
        a=1
        b=2
        c=3
        >>> cfg.get("b")
        2
        >>> cfg.get("d", "default")
        'default'

    Notes:
        This class is particularly useful for storing configuration parameters in a more accessible
        and iterable format compared to a standard dictionary.
    """

    def __iter__(self):
        """Return an iterator of key-value pairs from the namespace's attributes."""
        return iter(vars(self).items())

    def __str__(self):
        """Return a human-readable string representation of the object."""
        return "\n".join(f"{k}={v}" for k, v in vars(self).items())

    def __getattr__(self, attr):
        """Custom attribute access error message with helpful information."""
        name = self.__class__.__name__
        raise AttributeError(
            f"""
            '{name}' object has no attribute '{attr}'. This may be caused by a modified or out of date ultralytics
            'default.yaml' file.
            """
        )

    def get(self, key, default=None):
        """Return the value of the specified key if it exists; otherwise, return the default value."""
        return getattr(self, key, default)


def get_override_cfg(cfg=None, **kwargs):
    """
    合并 cfg 和 kwargs，检查 cfg 中的键是否存在，若存在则覆盖，不存在则新增。

    Args:
        cfg (EasyDict, optional): 原始配置字典。默认 None。
        **kwargs: 需要添加或覆盖的配置参数。

    Returns:
        EasyDict: 合并后的配置字典。
    """
    # 初始化 cfg 为 EasyDict
    if cfg is None:
        cfg = edict()
    elif not isinstance(cfg, edict):
        raise TypeError("cfg must be an EasyDict object")

    for key, value in kwargs.items():
        cfg.key = value  # 添加或更新 cfg

    return cfg


def is_ubuntu() -> bool:
    """
    Check if the OS is Ubuntu.

    Returns:
        (bool): True if OS is Ubuntu, False otherwise.
    """
    try:
        with open("/etc/os-release") as f:
            return "ID=ubuntu" in f.read()
    except FileNotFoundError:
        return False


def get_ubuntu_version():
    """
    Retrieve the Ubuntu version if the OS is Ubuntu.

    Returns:
        (str): Ubuntu version or None if not an Ubuntu OS.
    """
    if is_ubuntu():
        try:
            with open("/etc/os-release") as f:
                return re.search(r'VERSION_ID="(\d+\.\d+)"', f.read())[1]
        except (FileNotFoundError, AttributeError):
            return None


if __name__ == "__main__":
    path = "../cfg/indus.yaml"
    d = yaml_model_load(path)
    print(d)
