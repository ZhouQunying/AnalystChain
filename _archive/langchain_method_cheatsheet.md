# LangChain方法速查

> **文档结构**：LangChain运行时API（执行层） + Prompt工程（策略层）

---

# 第一部分：LangChain运行时API

## 整体架构（2×2矩阵）

```
同步/异步 × 完整/流式 = 4个核心方法

同步方法（阻塞）：
├── invoke()  → 完整结果
└── stream() → 流式输出

异步方法（非阻塞）：
├── ainvoke() → 完整结果
└── astream() → 流式输出
```

## 核心概念系统

### 1. 同步 vs 异步

| 特性 | 同步 | 异步 |
|------|------|------|
| 调用方式 | 直接调用 | `await` 或 `async for` |
| 执行特点 | 阻塞：等待结果返回 | 非阻塞：立即继续，结果可用时处理 |
| 适用场景 | 简单脚本、单任务 | Web应用、高并发、I/O密集型 |
| 返回类型 | 直接值或生成器 | Awaitable或异步生成器 |

**关键字**：
- `async`：声明异步函数（定义）
- `await`：等待异步操作完成（调用）
- `async for`：遍历异步生成器

### 2. 完整 vs 流式

| 特性 | 完整（invoke/ainvoke） | 流式（stream/astream） |
|------|----------------------|----------------------|
| 返回方式 | 一次性返回完整结果 | 逐步返回多个chunk |
| 调用方式 | 直接调用/await | `for`/`async for` 遍历 |
| 适用场景 | 需要完整结果 | 实时显示、逐步处理 |
| 类比 | 点外卖等完整送达 | 流水线逐步接收 |

## 4个核心方法

| 方法 | 类型 | 调用方式 | 返回值 | 示例 |
|------|------|---------|--------|------|
| `invoke()` | 同步完整 | `result = llm.invoke("问题")` | BaseMessage | `llm.invoke("问题")` |
| `stream()` | 同步流式 | `for chunk in llm.stream("问题"):` | BaseMessage迭代器 | `for chunk in llm.stream("问题"):` |
| `ainvoke()` | 异步完整 | `result = await llm.ainvoke("问题")` | Awaitable[BaseMessage] | `await llm.ainvoke("问题")` |
| `astream()` | 异步流式 | `async for chunk in llm.astream("问题"):` | 异步BaseMessage迭代器 | `async for chunk in llm.astream("问题"):` |

**记忆口诀**：同步单流，异步加a；完整invoke，流式stream

## 入参系统（LangGraph专用）

### message入参格式

**LangGraph的stream/invoke方法**使用字典格式输入：

```python
# 标准格式
{"messages": [{"role": "user", "content": "问题"}]}

# 实际使用
for message_chunk, metadata in agent.stream(
    {"messages": [{"role": "user", "content": question}]},
    stream_mode="messages"
):
    print(message_chunk.content)
```

**说明**：
- `messages`：消息列表（必需）
- `role`：角色（"user"/"assistant"/"system"）
- `content`：消息内容

### 返回值系统（LangGraph stream模式）

**stream()返回**：`(message_chunk, metadata)` 元组

```python
for message_chunk, metadata in agent.stream(
    {"messages": [...]},
    stream_mode="messages"
):
    # message_chunk是BaseMessage对象
    # 直接访问.content，没有.message属性
    print(message_chunk.content)  # 正确
    # print(message_chunk.message)  # 错误，不存在
```

**重要**：返回值是`message_chunk`（BaseMessage对象），**没有`.message`属性**，直接使用`.content`获取内容

## stream_mode参数系统（LangGraph专用）

### 整体分类（4种模式）

```
stream_mode可选值：
├── "values"   → 流式返回完整状态值
├── "messages" → 流式返回消息（最常用）
├── "updates"  → 流式返回状态更新
└── "debug"    → 调试模式（详细日志）
```

### 详细说明

#### 1. `stream_mode="values"`（默认）

**作用**：流式返回每个节点执行后的完整状态值

**返回值**：完整状态字典

**示例**：
```python
for state in agent.stream(
    {"messages": [{"role": "user", "content": "问题"}]},
    stream_mode="values"  # 默认值，可省略
):
    print(state)  # 完整状态字典
    # {"messages": [...], "other_keys": ...}
```

**适用场景**：需要完整状态信息、调试工作流

#### 2. `stream_mode="messages"`（最常用）

**作用**：流式返回消息对象（BaseMessage）

**返回值**：`(message_chunk, metadata)` 元组

**示例**：
```python
for message_chunk, metadata in agent.stream(
    {"messages": [{"role": "user", "content": "问题"}]},
    stream_mode="messages"
):
    if message_chunk.content:
        print(message_chunk.content, end="")  # Token级流式输出
```

**适用场景**：实时显示响应、Token级流式输出（ChatGPT效果）

**优势**：
- 实时显示，用户体验好
- 可以区分不同类型的消息（AIMessage/ToolMessage等）
- 适合交互式应用

#### 3. `stream_mode="updates"`

**作用**：流式返回状态更新（只包含变化的部分）

**返回值**：状态更新字典（只包含变化的键值对）

**示例**：
```python
for update in agent.stream(
    {"messages": [{"role": "user", "content": "问题"}]},
    stream_mode="updates"
):
    print(update)  # 只包含变化的键值对
    # {"messages": [新增的消息]}
```

**适用场景**：监控状态变化、增量更新

**优势**：
- 只返回变化部分，效率高
- 适合需要追踪状态变化的场景

#### 4. `stream_mode="debug"`

**作用**：调试模式，返回详细执行信息

**返回值**：包含执行步骤、节点信息、错误等详细日志

**示例**：
```python
for event in agent.stream(
    {"messages": [{"role": "user", "content": "问题"}]},
    stream_mode="debug"
):
    print(event)  # 详细调试信息
    # 包含：节点名、输入、输出、错误等
```

**适用场景**：调试工作流、排查问题

**优势**：
- 提供完整的执行轨迹
- 包含错误信息和中间状态

### 对比表

| stream_mode | 返回值类型 | 返回内容 | 适用场景 | 性能 |
|------------|----------|---------|---------|------|
| `"values"` | 完整状态字典 | 每个节点后的完整状态 | 需要完整状态、调试 | 中等 |
| `"messages"` | `(message, metadata)` | 消息对象 | 实时显示、流式输出 | 高 |
| `"updates"` | 状态更新字典 | 只包含变化的部分 | 监控变化、增量更新 | 最高 |
| `"debug"` | 调试信息 | 详细执行日志 | 调试、排查问题 | 最低 |

### 知识结构图

```
LangGraph stream_mode系统
├── "values"（默认）
│   └── 完整状态值（每个节点后）
├── "messages"（最常用）
│   └── 消息对象（Token级流式）
├── "updates"
│   └── 状态更新（只包含变化）
└── "debug"
    └── 调试信息（详细日志）
```

### 快速决策树

```
需要流式输出？
├── 需要实时显示响应？
│   └── stream_mode="messages"（最常用）
├── 需要完整状态信息？
│   └── stream_mode="values"（默认）
├── 需要监控状态变化？
│   └── stream_mode="updates"
└── 需要调试工作流？
    └── stream_mode="debug"
```

### 记忆口诀

- values：完整状态（默认）
- messages：消息流式（最常用）
- updates：增量更新（高效）
- debug：调试模式（详细）

## 返回值系统

### BaseMessage对象结构

所有方法返回的都是`BaseMessage`对象（或其迭代器）

| 属性 | 作用 | 示例 |
|------|------|------|
| `.content` | 文本内容（最常用） | `result.content` → `"这是回答"` |
| `.response_metadata` | 响应元数据 | `result.response_metadata` → `{'token_usage': {...}}` |
| `.tool_calls` | 工具调用信息 | `result.tool_calls` → `[{'name': 'search', ...}]` |
| `.id` | 消息唯一ID | `result.id` → `"run-abc123..."` |

**记忆口诀**：content拿内容，metadata看详情，tool_calls查工具

### 实际取值示例

**同步完整**：
```python
result = llm.invoke("问题")
print(result.content)  # "这是完整回答"
```

**同步流式**：
```python
for chunk in llm.stream("问题"):
    print(chunk.content, end="")  # 实时流式输出
```

**异步完整**：
```python
result = await llm.ainvoke("问题")
print(result.content)  # "这是完整回答"
```

**异步流式**：
```python
async for chunk in llm.astream("问题"):
    print(chunk.content, end="")  # 实时流式输出
```

## 输出解析器系统

### 整体分类（6种解析器）

| 解析器 | 作用 | 适用场景 |
|--------|------|----------|
| `JsonOutputParser` | JSON字符串 → Dict | 通用JSON输出 |
| `PydanticOutputParser` | JSON → Pydantic模型 | 需要类型验证 |
| `StrOutputParser` | 原样返回字符串 | 普通文本 |
| `ListOutputParser` | 逗号分隔 → List | 简单列表 |
| `XMLOutputParser` | XML → Dict | XML格式输出 |
| `OutputFixingParser` | 自动修复格式错误 | 包装其他Parser |

### 详细说明

#### 1. JsonOutputParser（最常用）

**一句话**：把LLM输出的JSON字符串转成Python字典

**代码**：
```python
from langchain_core.output_parsers import JsonOutputParser

chain = prompt | llm | JsonOutputParser()
result = chain.invoke({"content": "..."})  # 返回dict
```

**特点**：
- 自动解析JSON字符串
- 自动提取markdown代码块中的JSON
- 内置错误处理

#### 2. PydanticOutputParser（强校验）

**一句话**：把JSON解析为Pydantic模型，带类型验证

**代码**：
```python
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel

class Knowledge(BaseModel):
    topic: str
    summary: str

parser = PydanticOutputParser(pydantic_object=Knowledge)
chain = prompt | llm | parser
result = chain.invoke({"content": "..."})  # 返回Knowledge对象
```

**特点**：
- 自动校验字段类型
- 校验失败抛出异常
- 可用 `.get_format_instructions()` 获取格式说明插入Prompt

#### 3. OutputFixingParser（容错包装）

**一句话**：包装其他Parser，解析失败时用LLM自动修复

**代码**：
```python
from langchain.output_parsers import OutputFixingParser

base_parser = JsonOutputParser()
fixing_parser = OutputFixingParser.from_llm(parser=base_parser, llm=llm)

chain = prompt | llm | fixing_parser
```

**特点**：
- 解析失败时自动调用LLM修复
- 增加一次LLM调用（成本）
- 适合输出不稳定的场景

### 推荐组合

| 场景 | 推荐 | 理由 |
|------|------|------|
| 简单JSON提取 | `JsonOutputParser` | 最简单，够用 |
| 严格Schema | `PydanticOutputParser` | 类型安全 |
| 不可靠LLM | `OutputFixingParser` 包装 | 自动容错 |

### 快速决策树

```
需要解析LLM输出？
├── 返回JSON？
│   ├── 需要类型验证？
│   │   └── PydanticOutputParser
│   └── 不需要？
│       └── JsonOutputParser（最常用）
├── 返回文本？
│   └── StrOutputParser
├── 返回列表？
│   └── ListOutputParser
└── LLM输出不稳定？
    └── OutputFixingParser包装
```

### 记忆口诀

```
Json最常用，字符串转字典
Pydantic强校验，类型全检查
Fixing做包装，失败自动修
```

---

## 扩展方法

| 方法 | 与核心方法关系 | 适用场景 |
|------|--------------|---------|
| `batch()` | = 多个`invoke()` | 批量处理多个输入 |
| `abatch()` | = 多个`ainvoke()` | 异步批量处理 |
| `stream_log()` | = `stream()` + 详细日志 | 调试工作流 |
| `astream_log()` | = `astream()` + 详细日志 | 异步调试 |

## 快速决策树

```
需要调用LLM？
├── 需要完整结果？
│   ├── 同步场景 → invoke()
│   └── 异步场景 → ainvoke()
└── 需要流式输出？
    ├── 同步场景 → stream()
    └── 异步场景 → astream()
```

---

# 第二部分：Prompt工程

## Prompt示例系统（5种方案）

### 整体分类

```
Prompt示例方案（从简单到复杂）：
├── Few-shot Prompting        → 示例直接写死在Prompt
├── FewShotPromptTemplate     → 从列表中动态选示例
├── ExampleSelector           → 语义匹配选最佳示例
├── with_structured_output    → Pydantic Schema + 示例
└── Query Analysis Examples   → 查询解析专用示例
```

### 详细说明

#### 1. Few-shot Prompting（最简单）

**一句话**：示例直接写在Prompt字符串里

**代码**：
```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("""
提取JSON格式。

示例：
输入：消费是经济增长动力
输出：{{"topic": "消费", "summary": "经济增长动力"}}

输入：{content}
输出：
""")
```

**适用场景**：示例固定（1-3个）、结构简单

---

#### 2. FewShotPromptTemplate（动态选择）

**一句话**：从示例列表中动态选N个插入Prompt

**代码**：
```python
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

examples = [
    {"input": "消费文档", "output": '{"topic": "消费"}'},
    {"input": "投资文档", "output": '{"topic": "投资"}'}
]

prompt = FewShotPromptTemplate(
    examples=examples[:2],  # 选前2个
    example_prompt=PromptTemplate.from_template(
        "输入：{input}\n输出：{output}"
    ),
    prefix="提取JSON示例：",
    suffix="输入：{content}\n输出：",
    input_variables=["content"]
)
```

**适用场景**：多个示例需要轮换、动态控制示例数量

---

#### 3. ExampleSelector（语义匹配）

**一句话**：根据查询内容语义自动选最相似的示例

**代码**：
```python
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 向量化示例库
selector = SemanticSimilarityExampleSelector.from_examples(
    examples=[
        {"input": "消费相关文档", "output": "消费JSON"},
        {"input": "投资相关文档", "output": "投资JSON"},
        {"input": "出口相关文档", "output": "出口JSON"}
    ],
    embeddings=HuggingFaceEmbeddings(),
    vectorstore_cls=Chroma,
    k=2  # 选2个最相似的
)

# 使用selector替代固定examples
prompt = FewShotPromptTemplate(
    example_selector=selector,
    example_prompt=PromptTemplate.from_template("{input}\n{output}"),
    suffix="输入：{content}"
)
```

**适用场景**：大示例库（>10个）、需要智能匹配

---

#### 4. with_structured_output（强校验）

**一句话**：Pydantic模型定义结构，自动生成Schema示例

**代码**：
```python
from pydantic import BaseModel, Field

class KnowledgeJSON(BaseModel):
    topic: str = Field(
        description="主题名称",
        examples=["消费", "投资"]  # 示例
    )
    summary: str = Field(
        description="总结摘要",
        examples=["消费是经济增长动力"]
    )

# 方式1：直接传examples
llm_with_structure = llm.with_structured_output(
    KnowledgeJSON,
    examples=[
        {"topic": "消费", "summary": "..."}
    ]
)

# 方式2：在Field中配置（推荐）
chain = prompt | llm.with_structured_output(KnowledgeJSON)
```

**适用场景**：需要强校验、Pydantic项目

---

#### 5. Query Analysis Examples（查询解析）

**一句话**：教LLM如何把自然语言转成结构化查询

**代码**：
```python
examples = [
    {
        "input": "查询2023年的消费数据",
        "output": {"filter": {"year": 2023}, "query": "消费"}
    },
    {
        "input": "找投资相关的",
        "output": {"filter": {}, "query": "投资"}
    }
]

prompt = ChatPromptTemplate.from_messages([
    ("system", "解析查询为结构化格式"),
    ("human", "示例：\n{examples}\n\n查询：{query}")
])
```

**适用场景**：复杂查询解析、搜索系统

---

### 对比表

| 方案 | 代码量 | 灵活性 | 适用场景 |
|------|--------|--------|---------|
| Few-shot | 最少（3行） | 固定 | 示例固定、结构简单 |
| FewShotTemplate | 中（10行） | 中 | 多示例轮换 |
| ExampleSelector | 多（15行） | 高 | 大示例库（>10个） |
| with_structured_output | 少（5行） | 低 | Pydantic强校验 |
| Query Analysis | 多（20行） | 高 | 查询解析系统 |

### 快速决策树

```
需要给LLM示例？
├── 示例固定（1-3个）？
│   └── Few-shot Prompting（最简单）
├── 需要动态选示例？
│   ├── 示例多（>10个）？
│   │   └── ExampleSelector（语义匹配）
│   └── 示例少（<10个）？
│       └── FewShotPromptTemplate
├── 需要Pydantic校验？
│   └── with_structured_output
└── 查询解析场景？
    └── Query Analysis Examples
```

### 记忆口诀

```
Few-shot最简单，示例写死在里面
FewShot动态选，列表中挑几个
Selector语义配，向量库找最像
Structured强校验，Pydantic定规则
Query做解析，查询转结构
```

---

# 第三部分：RAG原理

## 向量检索机制

### 核心概念（4个）

| 概念 | 1句话解释 |
|------|---------|
| **存储单位** | chunk(文档片段,通常800字),不是完整文档 |
| **检索原理** | 问题和chunk都转为向量,计算余弦相似度(0-1) |
| **语义匹配** | 理解"意思"相近,不是关键词匹配 |
| **分块目的** | 提高检索精准度,避免长文本稀释相关性 |

### 检索流程

```
用户问题："投资时钟是什么？"
    ↓
问题向量化（Embedding）
    ↓ [0.123, -0.456, ...]
向量库中所有chunk向量
    ↓ 计算余弦相似度
Top-K结果（k=3）
    ↓
chunk_0 (0.892), chunk_1 (0.823), chunk_2 (0.765)
```

## RAG设计原则

### 核心权衡

| 权衡点 | 考量 |
|--------|------|
| **Chunk大小** | 过小→碎片化,过大→稀释相关性 |
| **返回数量(K)** | 过少→漏信息,过多→噪音+token消耗 |
| **完整vs截断** | 完整→信息全,截断→省token |

## 双存储架构

| 存储 | 角色 | 职责 |
|------|------|------|
| **向量库** | 搜索引擎 | 语义检索,快速定位 |
| **结构化库** | 数据库 | 完整知识,精确查询 |

```
向量检索定位 → 判断是否够用 → 不够？查结构化获取完整
```

## 记忆口诀

```
向量存chunk算相似,完整返回给LLM
双存分工互补充,省token减K不截断
```

---

# 整体知识结构图

```
LangChain知识体系
│
├── 【第一部分：运行时API】（执行层）
│   ├── 核心方法（4个）
│   │   ├── 同步：invoke/stream
│   │   └── 异步：ainvoke/astream
│   ├── 入参系统
│   │   ├── message格式
│   │   └── stream_mode（4种）
│   ├── 返回值系统
│   │   └── BaseMessage对象
│   ├── 输出解析器系统（6种）
│   │   ├── JsonOutputParser（最常用）
│   │   ├── PydanticOutputParser（强校验）
│   │   └── OutputFixingParser（容错包装）
│   └── 扩展方法（4个）
│       └── batch/stream_log等
│
├── 【第二部分：Prompt工程】（策略层）
│   └── 示例系统（5种方案）
│       ├── Few-shot（最简单）
│       ├── FewShotTemplate（动态选择）
│       ├── ExampleSelector（语义匹配）
│       ├── with_structured_output（强校验）
│       └── Query Analysis（查询解析）
│
└── 【第三部分：RAG原理】（检索层）
    ├── 向量检索机制（chunk存储/语义匹配）
    ├── 权衡设计（token经济/信息完整性）
    └── 双存储架构（向量定位/JSON完整）
```
