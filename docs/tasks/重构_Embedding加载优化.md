# 重构方案：Embedding加载优化

> **目标**：抽离 `HuggingFaceEmbeddings` 加载逻辑为公共工具函数，统一管理

---

## 问题

**现象**：每次初始化 `HuggingFaceEmbeddings` 都很慢（5-10秒）

**原因**：
1. 未使用 `local_files_only=True`，导致每次都进行网络验证
2. 相同逻辑在多处重复，维护困难

---

## 方案：创建公共工具函数

### 步骤1：新增工具文件

创建 `src/analyst_chain/utils/embedding_utils.py`

```python
"""Embedding模型加载工具"""

from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings


def get_embeddings(model_name: str) -> HuggingFaceEmbeddings:
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
            model_kwargs={"local_files_only": True}
        )
    return HuggingFaceEmbeddings(model_name=model_name)
```

### 步骤2：更新 `__init__.py` 导出

在 `src/analyst_chain/utils/__init__.py` 中添加：

```python
from .embedding_utils import get_embeddings

__all__ = ["get_embeddings"]
```

### 步骤3：修改所有使用处

#### 3.1 `src/analyst_chain/tools/knowledge_retriever.py`

```python
# 修改前
from langchain_huggingface import HuggingFaceEmbeddings

# ...

self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
```

```python
# 修改后
from ..utils.embedding_utils import get_embeddings

# ...

self.embeddings = get_embeddings(embedding_model)
```

#### 3.2 `notebooks/stage1_macro_knowledge_pipeline.ipynb` VectorStoreManager

```python
# 修改前（在 VectorStoreManager.__init__ 中）
cache_path = Path.home() / ".cache/huggingface/hub" / f"models--{embedding_model.replace('/', '--')}"
start = time.time()
if cache_path.exists():
    print(f"[进度] 从缓存加载Embedding模型：{embedding_model}", flush=True)
    self.embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"local_files_only": True}
    )
else:
    print(f"[进度] 首次运行，下载Embedding模型：{embedding_model}...", flush=True)
    self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
elapsed = time.time() - start
print(f"[进度] Embedding模型加载完成（耗时{elapsed:.1f}s）", flush=True)
```

```python
# 修改后
from analyst_chain.utils.embedding_utils import get_embeddings

# ...

start = time.time()
self.embeddings = get_embeddings(embedding_model)
elapsed = time.time() - start
print(f"[进度] Embedding模型加载完成（耗时{elapsed:.1f}s）", flush=True)
```

#### 3.3 `notebooks/stage2_macro_agent.ipynb` KnowledgeRetriever

```python
# 修改前
self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
```

```python
# 修改后
from analyst_chain.utils.embedding_utils import get_embeddings

# ...

self.embeddings = get_embeddings(embedding_model)
```

---

## 影响范围汇总

| 文件 | 修改内容 |
|------|----------|
| `src/analyst_chain/utils/embedding_utils.py` | 新增 |
| `src/analyst_chain/utils/__init__.py` | 添加导出 |
| `src/analyst_chain/tools/knowledge_retriever.py` | 替换为 `get_embeddings()` |
| `notebooks/stage1_macro_knowledge_pipeline.ipynb` | 替换为 `get_embeddings()` |
| `notebooks/stage2_macro_agent.ipynb` | 替换为 `get_embeddings()` |

---

## 验证清单

- [ ] `embedding_utils.py` 创建完成
- [ ] `knowledge_retriever.py` 修改完成
- [ ] `stage1_*.ipynb` 修改完成
- [ ] `stage2_*.ipynb` 修改完成
- [ ] `python tests/test_stage1_knowledge_retrieval.py` 通过
- [ ] 首次查询延迟 < 3秒

---

**完成后删除此文档**
