"""Embedding模型工具"""

from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings


def get_embeddings(model_name: str, **args) -> HuggingFaceEmbeddings:
    """获取Embedding模型（优先使用本地缓存）

    Args:
        model_name: 模型名称，如 "Qwen/Qwen3-Embedding-0.6B"

    Returns:
        HuggingFaceEmbeddings 实例
    """
    cache_path = Path.home() / ".cache/huggingface/hub" / f"models--{model_name.replace('/', '--')}"
    if cache_path.exists():
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"local_files_only": True},
            **args,
        )
    return HuggingFaceEmbeddings(
        model_name=model_name,
        **args,
    )
