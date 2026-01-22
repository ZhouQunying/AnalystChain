"""路径工具函数"""

from pathlib import Path


def get_project_root() -> Path:
    """获取项目根目录

    通过查找setup.py文件定位项目根目录

    Returns:
        项目根目录Path对象

    Raises:
        RuntimeError: 未找到项目根目录时
    """
    current = Path(__file__).resolve().parent
    max_depth = 10

    for _ in range(max_depth):
        if (current / "setup.py").exists():
            return current
        if current == current.parent:
            break
        current = current.parent

    raise RuntimeError("未找到项目根目录")
