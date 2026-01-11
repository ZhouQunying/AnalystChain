"""
静态变量
"""

import os
from pathlib import Path

# Embedding 模型，从环境变量读取，默认 Qwen/Qwen3-Embedding-0.6B
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_PATH", "Qwen/Qwen3-Embedding-0.6B")

# 路径配置
MACRO_ECONOMY_KNOWLEDGE_BASE_DIR = Path("../../../data/raw/knowledge_base/macro_economy")
MACRO_ECONOMY_PROCESSED_DIR = Path("../../../data/processed")
MACRO_ECONOMY_VECTOR_DB_DIR = MACRO_ECONOMY_PROCESSED_DIR / "knowledge/macro_economy_vector_db"
MACRO_ECONOMY_STRUCTURED_JSON_DIR = MACRO_ECONOMY_PROCESSED_DIR / "knowledge/macro_economy_structured_json"


# tmp
# src/analyst_chain/utils/path_utils.py
"""路径工具函数"""
from pathlib import Path
from typing import Final

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

# src/analyst_chain/knowledge/constants.py
"""知识库常量配置模块"""
import os
from pathlib import Path
from typing import Final
from ..utils.path_utils import get_project_root

_PROJECT_ROOT: Final[Path] = get_project_root()

# 模型配置
EMBEDDING_MODEL: Final[str] = os.getenv(
    "EMBEDDING_MODEL_PATH",
    "Qwen/Qwen3-Embedding-0.6B"
)

# 基础路径配置
DATA_DIR: Final[Path] = _PROJECT_ROOT / "data"
RAW_DATA_DIR: Final[Path] = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Final[Path] = DATA_DIR / "processed"
KNOWLEDGE_BASE_DIR: Final[Path] = RAW_DATA_DIR / "knowledge_base"
STRUCTURED_JSON_DIR: Final[Path] = PROCESSED_DATA_DIR / "knowledge" / "structured"
VECTOR_DB_DIR: Final[Path] = PROCESSED_DATA_DIR / "knowledge" / "vector_db"

# 领域特定路径函数
def get_knowledge_base_dir(domain: str = "macro_economy") -> Path:
    """获取指定领域的知识库目录"""
    return KNOWLEDGE_BASE_DIR / domain

def get_structured_json_dir(domain: str = "macro_economy") -> Path:
    """获取指定领域的结构化JSON目录"""
    return STRUCTURED_JSON_DIR / domain

def get_vector_db_dir(domain: str = "macro_economy") -> Path:
    """获取指定领域的向量库目录"""
    return VECTOR_DB_DIR / domain

# 向后兼容
MACRO_ECONOMY_KNOWLEDGE_BASE_DIR: Final[Path] = get_knowledge_base_dir("macro_economy")
MACRO_ECONOMY_PROCESSED_DIR: Final[Path] = PROCESSED_DATA_DIR
MACRO_ECONOMY_VECTOR_DB_DIR: Final[Path] = get_vector_db_dir("macro_economy")
MACRO_ECONOMY_STRUCTURED_JSON_DIR: Final[Path] = get_structured_json_dir("macro_economy")