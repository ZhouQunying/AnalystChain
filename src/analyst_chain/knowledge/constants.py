"""
知识库常量配置
"""

import os
from typing import Final
from pathlib import Path
from ..utils.path_utils import get_project_root


def get_domain_dir(base_dir: Path, domain: str) -> Path:
    """获取指定领域在特定基础目录下的路径

    Args:
        base_dir: 基础目录
        domain: 领域名称

    Returns:
        领域特定路径
    """
    return base_dir / domain

# 根目录
_PROJECT_ROOT: Final[Path] = get_project_root()

# 模型配置
EMBEDDING_MODEL: Final[str] = os.getenv(
    'EMBEDDING_MODEL_PATH',
    'Qwen/Qwen3-Embedding-0.6B',
)

# 基础路径配置
DATA_DIR: Final[Path] = _PROJECT_ROOT / 'data'
KNOWLEDGE_BASE_DIR: Final[Path] = DATA_DIR / 'raw' / 'knowledge_base'
PROCESSED_DATA_DIR: Final[Path] = DATA_DIR / 'processed'
STRUCTURED_JSON_DIR: Final[Path] = PROCESSED_DATA_DIR / 'knowledge' / 'structured'
VECTOR_DB_DIR: Final[Path] = PROCESSED_DATA_DIR / 'knowledge' / 'vector_db'

MACRO_ECONOMY_KNOWLEDGE_BASE_DIR: Final[Path] = get_domain_dir(KNOWLEDGE_BASE_DIR, 'macro_economy')
MACRO_ECONOMY_PROCESSED_DATA_DIR: Final[Path] = get_domain_dir(PROCESSED_DATA_DIR, 'macro_economy')
MACRO_ECONOMY_STRUCTURED_JSON_DIR: Final[Path] = get_domain_dir(STRUCTURED_JSON_DIR, 'macro_economy')
MACRO_ECONOMY_VECTOR_DB_DIR: Final[Path] = get_domain_dir(VECTOR_DB_DIR, 'macro_economy')
