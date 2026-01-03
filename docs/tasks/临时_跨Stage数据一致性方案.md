# 临时任务：跨Stage数据一致性共享模块

> **完成后删除此文档**

## 问题

Stage1 notebook 中定义的数据结构、常量、JSON结构需要在所有模块中保持一致地解析和使用。

**涉及模块**：
- **notebooks**：Stage1/Stage2+ notebook（数据结构、常量、JSON结构）
- **tools**：`knowledge_retrieval.py` 等工具（路径配置、JSON结构）
- **tests**：`test_knowledge_retrieval.py` 等测试（路径配置、JSON结构验证）
- **agents**：未来 Agent 代码（数据结构、常量、JSON结构）

**现状问题**：
- Stage1：不拆分 notebook，数据结构/常量直接定义
- Stage2+：既保留 notebook 也进行拆分，存在代码重复
- tools：硬编码路径配置，无法统一管理
- tests：依赖 tools 的默认路径，无法灵活测试多领域

## 方案

创建共享模块 `src/analyst_chain/knowledge/`，统一管理跨 stage 的数据结构、常量、JSON结构。

### 目录结构

```
src/analyst_chain/knowledge/
├── __init__.py
├── models.py          # 数据结构（FilePriority、FileInfo、KnowledgeGroup）
├── constants.py       # 常量（路径、模型、配置）
└── schemas.py         # JSON结构定义（TypedDict）
```

### 文件内容

**models.py**：
- `FilePriority`（IntEnum）
- `FileInfo`（@dataclass）
- `KnowledgeGroup`（@dataclass）

**constants.py**：
- 路径配置（基础路径，不带domain）：
  - `KNOWLEDGE_BASE_DIR`：原始知识库根目录（`data/raw/knowledge_base/`）
  - `OUTPUT_DIR`：产物根目录（`data/processed/`）
  - `VECTOR_DB_DIR`：向量库基础目录（`data/processed/knowledge/vector_db/`）
  - `STRUCTURED_JSON_DIR`：JSON基础目录（`data/processed/knowledge/structured/`）
  - 注意：domain（如`macro_economy`）通过参数动态传递，路径拼接为`VECTOR_DB_DIR / domain`
- 模型配置（EMBEDDING_MODEL、LLM_MODEL、LLM_TEMPERATURE）
- 文本分割配置（CHUNK_SIZE、CHUNK_OVERLAP）

**schemas.py**：
- `KeyConcept`（TypedDict）
- `Indicator`（TypedDict）
- `AnalysisMethod`（TypedDict）
- `KnowledgeJSON`（TypedDict）

## 实现步骤

1. 创建 `src/analyst_chain/knowledge/` 目录
2. 创建 `__init__.py`（导出主要类和常量）
3. 创建 `models.py`（从 stage1 notebook 迁移数据结构）
4. 创建 `constants.py`（从 stage1 notebook 迁移常量）
5. 创建 `schemas.py`（定义 JSON 结构）
6. 更新 stage1 notebook（可选：导入使用或保持现状）
7. 更新 stage2+ notebook/代码（统一从 src 导入）
8. 更新 tools（如 `knowledge_retrieval.py`）：
   - 从 `constants.py` 导入路径配置
   - 支持 `domain` 参数，动态拼接路径
   - 从 `schemas.py` 导入 JSON 结构定义（用于类型提示）
9. 更新 tests（如 `test_knowledge_retrieval.py`）：
   - 从 `constants.py` 导入路径配置
   - 支持测试不同 domain
   - 使用 `schemas.py` 验证 JSON 结构

## 使用方式

**Stage1 Notebook（不拆分）**：
```python
# 方式1：导入使用（推荐）
from src.analyst_chain.knowledge.models import FilePriority, FileInfo, KnowledgeGroup
from src.analyst_chain.knowledge.constants import CHUNK_SIZE, VECTOR_DB_DIR
from src.analyst_chain.knowledge.schemas import KnowledgeJSON

# 方式2：直接定义（向后兼容，但建议迁移到方式1）
# 保持现有代码不变
```

**Stage2+ Notebook/代码（会拆分）**：
```python
# 统一从src导入
from src.analyst_chain.knowledge.models import FilePriority
from src.analyst_chain.knowledge.constants import VECTOR_DB_DIR
from src.analyst_chain.knowledge.schemas import KnowledgeJSON
```

**Tools（如 knowledge_retrieval.py）**：
```python
from src.analyst_chain.knowledge.constants import VECTOR_DB_DIR, STRUCTURED_JSON_DIR
from src.analyst_chain.knowledge.schemas import KnowledgeJSON

# 使用基础路径 + domain 动态拼接
def __init__(self, domain: str = "macro_economy"):
    vector_db_path = VECTOR_DB_DIR / domain
    json_dir_path = STRUCTURED_JSON_DIR / domain
    # ...
```

**Tests（如 test_knowledge_retrieval.py）**：
```python
from src.analyst_chain.knowledge.constants import VECTOR_DB_DIR, STRUCTURED_JSON_DIR
from src.analyst_chain.tools.knowledge_retrieval import KnowledgeRetriever

# 测试时可以指定不同 domain
retriever = KnowledgeRetriever(domain="macro_economy")
# 或测试其他领域
retriever_policy = KnowledgeRetriever(domain="policy")
```

## 迁移策略

**阶段1：创建共享模块**（立即执行）
- 创建目录和文件
- 迁移数据结构、常量、JSON结构定义

**阶段2：更新 Stage1 Notebook**（可选）
- 在 notebook 中导入共享模块
- 或保持现状（向后兼容）

**阶段3：更新后续 Stage**（后续执行）
- Stage2+ 的 notebook 和拆分代码统一从 src 导入

## 优势

1. **单一数据源**：所有模块（notebooks、tools、tests、agents）从 src 导入，保证一致性
2. **不拆分友好**：Stage1 notebook 可导入可定义（向后兼容）
3. **拆分友好**：后续 stage 的 notebook 和代码统一导入
4. **类型安全**：使用 TypedDict 定义 JSON 结构，便于验证
5. **多领域支持**：tools 和 tests 通过 `domain` 参数支持多领域（macro_economy、policy、stock 等）
6. **统一管理**：路径配置、模型配置、JSON结构统一管理，修改一处全局生效

## 注意事项

- Stage1 notebook 保持向后兼容（可导入可定义）
- 后续 stage 必须统一从 src 导入
- tools 必须支持 `domain` 参数，不能硬编码路径
- tests 必须支持测试不同 domain，不能依赖硬编码路径
- 路径配置是基础路径（不带 domain），domain 通过参数动态传递
- 完成后删除此临时文档

