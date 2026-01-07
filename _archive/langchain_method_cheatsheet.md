# LangChain方法速查

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

## 扩展方法

| 方法 | 与核心方法关系 | 适用场景 |
|------|--------------|---------|
| `batch()` | = 多个`invoke()` | 批量处理多个输入 |
| `abatch()` | = 多个`ainvoke()` | 异步批量处理 |
| `stream_log()` | = `stream()` + 详细日志 | 调试工作流 |
| `astream_log()` | = `astream()` + 详细日志 | 异步调试 |

## 知识结构图

```
LangChain方法体系
├── 核心方法（4个）
│   ├── 同步
│   │   ├── invoke()（完整）
│   │   └── stream()（流式）
│   └── 异步
│       ├── ainvoke()（完整）
│       └── astream()（流式）
├── 返回值系统
│   └── BaseMessage对象（content/metadata/tool_calls/id）
└── 扩展方法（4个）
    ├── batch/abatch（批量）
    └── stream_log/astream_log（调试）
```

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
