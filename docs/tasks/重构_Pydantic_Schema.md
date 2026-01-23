# 重构计划：TypedDict → Pydantic

> 状态：待执行

---

## 背景

当前 `schemas.py` 使用 TypedDict 定义知识库结构，存在以下问题：

| 问题 | 影响 |
|------|------|
| 字段名硬编码两处 | schema 定义一处，retriever 读取一处 |
| 无运行时验证 | JSON 格式错误不会报错 |
| IDE 补全受限 | dict 访问无法完整补全 |
| 重构风险高 | 改字段名容易遗漏 |

---

## 目标

改为 Pydantic BaseModel：
- 单一来源：属性访问，无需字符串字段名
- 运行时验证：JSON 格式错误会抛异常
- 完整 IDE 支持：属性补全、类型检查
- 重构安全：IDE 全局重命名

---

## 改动文件

| 文件 | 改动内容 | 影响 |
|------|---------|------|
| `src/analyst_chain/knowledge/schemas.py` | TypedDict → BaseModel | 核心 |
| `src/analyst_chain/tools/knowledge_retriever.py` | dict 访问 → 属性访问 | 读取侧 |
| `notebooks/stage1_macro_knowledge_pipeline.ipynb` | 写入用 `.model_dump_json()` | 写入侧 |

---

## 详细改动

### 1. `src/analyst_chain/knowledge/schemas.py`

#### 改动前

```python
"""知识库Schema定义与示例"""

from typing import TypedDict, List


class KeyConcept(TypedDict):
    """关键概念（单个）"""
    name: str
    definition: str
    importance: str


class Indicator(TypedDict):
    """指标（单个）"""
    name: str
    calculation: str
    interpretation: str


class AnalysisMethod(TypedDict):
    """分析方法（单个）"""
    name: str
    steps: str
    application: str


class KnowledgeJSON(TypedDict):
    """知识库JSON结构"""
    topic: str
    key_concepts: List[KeyConcept]
    indicators: List[Indicator]
    analysis_methods: List[AnalysisMethod]
    summary: str


# 示例（dict 格式）
EXAMPLE_KNOWLEDGE: KnowledgeJSON = {
    "topic": "01第一节 中国经济的三驾马车",
    "key_concepts": [
        {
            "name": "三驾马车概念",
            "definition": "GDP = 消费 + 投资 + 净出口",
            "importance": "宏观经济分析基础框架"
        }
    ],
    "indicators": [
        {
            "name": "GDP",
            "calculation": "支出法核算",
            "interpretation": "经济总量指标"
        }
    ],
    "analysis_methods": [
        {
            "name": "数据解读",
            "steps": "一看占比，二看贡献率",
            "application": "判断经济成熟度"
        }
    ],
    "summary": "三驾马车是拉动经济增长的核心力量"
}
```

#### 改动后

```python
"""知识库Schema定义与示例"""

from pydantic import BaseModel
from typing import List


class KeyConcept(BaseModel):
    """关键概念（单个）"""
    name: str
    definition: str
    importance: str


class Indicator(BaseModel):
    """指标（单个）"""
    name: str
    calculation: str
    interpretation: str


class AnalysisMethod(BaseModel):
    """分析方法（单个）"""
    name: str
    steps: str
    application: str


class KnowledgeJSON(BaseModel):
    """知识库JSON结构"""
    topic: str
    key_concepts: List[KeyConcept]
    indicators: List[Indicator]
    analysis_methods: List[AnalysisMethod]
    summary: str


# 示例（Pydantic 实例）
EXAMPLE_KNOWLEDGE = KnowledgeJSON(
    topic="01第一节 中国经济的三驾马车",
    key_concepts=[KeyConcept(
        name="三驾马车概念",
        definition="GDP = 消费 + 投资 + 净出口",
        importance="宏观经济分析基础框架"
    )],
    indicators=[Indicator(
        name="GDP",
        calculation="支出法核算",
        interpretation="经济总量指标"
    )],
    analysis_methods=[AnalysisMethod(
        name="数据解读",
        steps="一看占比，二看贡献率",
        application="判断经济成熟度"
    )],
    summary="三驾马车是拉动经济增长的核心力量"
)

# 供 Prompt 使用的 JSON 字符串
EXAMPLE_KNOWLEDGE_JSON = EXAMPLE_KNOWLEDGE.model_dump_json(indent=2, ensure_ascii=False)
```

---

### 2. `src/analyst_chain/tools/knowledge_retriever.py`

#### 改动点：get_topic_knowledge 方法

**改动前**：

```python
json_file = self.json_files[topic_number]
with open(json_file, "r", encoding="utf-8") as f:
    knowledge = json.load(f)

output = f"主题{topic_number}：{knowledge.get('topic', 'N/A')}\n\n"

if "key_concepts" in knowledge:
    output += "关键概念：\n"
    for concept in knowledge["key_concepts"]:
        name = concept.get("name", "N/A")
        definition = concept.get("definition", "N/A")
        output += f"  - {name}：{definition}\n"
    output += "\n"

if "indicators" in knowledge:
    output += "关键指标：\n"
    for indicator in knowledge["indicators"]:
        name = indicator.get("name", "N/A")
        interpretation = indicator.get("interpretation", "N/A")
        calculation = indicator.get("calculation", "N/A")
        output += f"  - {name}：{interpretation} | {calculation}\n"
    output += "\n"

if "summary" in knowledge:
    output += f"摘要：\n{knowledge['summary']}\n"
```

**改动后**：

```python
from ..knowledge.schemas import KnowledgeJSON

json_file = self.json_files[topic_number]
with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)
    knowledge = KnowledgeJSON.model_validate(data)

output = f"主题{topic_number}：{knowledge.topic}\n\n"

if knowledge.key_concepts:
    output += "关键概念：\n"
    for concept in knowledge.key_concepts:
        output += f"  - {concept.name}：{concept.definition}\n"
    output += "\n"

if knowledge.indicators:
    output += "关键指标：\n"
    for indicator in knowledge.indicators:
        output += f"  - {indicator.name}：{indicator.interpretation} | {indicator.calculation}\n"
    output += "\n"

if knowledge.summary:
    output += f"摘要：\n{knowledge.summary}\n"
```

---

### 3. `notebooks/stage1_macro_knowledge_pipeline.ipynb`

#### 改动点：save_to_memories 方法或 JSON 写入单元格

**改动前**：

```python
knowledge = {
    "topic": topic,
    "key_concepts": [...],
    "indicators": [...],
    "analysis_methods": [...],
    "summary": summary
}
with open(json_file, "w", encoding="utf-8") as f:
    json.dump(knowledge, f, ensure_ascii=False, indent=2)
```

**改动后**：

```python
from analyst_chain.knowledge.schemas import KnowledgeJSON, KeyConcept, Indicator, AnalysisMethod

knowledge = KnowledgeJSON(
    topic=topic,
    key_concepts=[KeyConcept(**c) for c in concepts],
    indicators=[Indicator(**i) for i in indicators],
    analysis_methods=[AnalysisMethod(**m) for m in methods],
    summary=summary
)
with open(json_file, "w", encoding="utf-8") as f:
    f.write(knowledge.model_dump_json(indent=2, ensure_ascii=False))
```

---

## 依赖

- `pydantic`：LangChain 依赖已包含，无需额外安装

---

## 测试验证

执行后需验证：

1. `python tests/test_stage1_knowledge_retrieval.py` 通过
2. `notebooks/stage1_macro_knowledge_pipeline.ipynb` 完整执行通过
3. JSON 文件格式不变（兼容现有数据）

---

## 改动量统计

| 文件 | 改动行数 |
|------|---------|
| `schemas.py` | ~40行 |
| `knowledge_retriever.py` | ~20行 |
| `stage1_*.ipynb` | ~10行 |
| **合计** | **~70行** |
