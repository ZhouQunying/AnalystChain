"""
知识库常量配置
"""

import os
from typing import Final
from ..utils.path_utils import get_project_root

_PROJECT_ROOT: Final[str] = get_project_root()

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
