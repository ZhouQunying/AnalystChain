# LangChain方法记忆速查卡

## 🧠 记忆口诀：**"同步单流，异步加a；批量带b，日志有log"**

### 🔄 同步 vs 异步：关键区别

| 特性 | 同步方法 | 异步方法 |
|------|---------|---------|
| **调用方式** | 直接调用 | 需`await`或`async for` |
| **执行特点** | 阻塞式：等待结果返回才继续 | 非阻塞：调用后立即继续执行后续代码，当结果可用时再处理 |
| **适用场景** | 简单脚本、单任务处理 | Web应用、高并发、I/O密集型任务 |
| **返回类型** | 直接值或生成器 | Awaitable对象或异步生成器 |
| **错误处理** | 直接try/except | 需在async函数中try/except |

### ⚙️ `async` vs `await`：关键字详解

| 关键字 | 作用 | 使用位置 | 示例 |
|--------|------|----------|------|
| **`async`** | 声明异步函数或异步生成器 | 函数定义前 | `async def my_func():` |
| **`await`** | 等待异步操作完成 | 异步函数内部 | `result = await llm.ainvoke("问题")` |
| **`async for`** | 遍历异步生成器 | 异步函数内部 | `async for chunk in llm.astream("问题"):` |

> 💡 **核心区别**：
> - **`async`** 是**定义**异步函数的关键词（"我要开始一个异步任务"）
> - **`await`** 是**使用**异步函数的关键词（"我现在需要等待这个异步任务完成"）
> - `async` 用于**声明**，`await` 用于**调用**

### 🔁 异步函数 vs 异步生成器：关键区别

| 特性 | 异步函数 | 异步生成器 |
|------|---------|---------|
| **定义方式** | `async def func():` | `async def func(): yield` |
| **返回类型** | 协程对象(Coroutine) | 异步生成器对象(AsyncGenerator) |
| **调用方式** | `await func()` | `async for item in func():` |
| **执行特点** | 一次完成，返回单一结果 | 可多次暂停，产生多个结果 |
| **适用场景** | 需要完整结果 | 流式处理，逐步获取结果 |
| **LangChain示例** | `ainvoke()` | `astream()` |

> 💡 **核心区别**：
> - **异步函数**：像点外卖，下单后等待完整餐品送达（`await llm.ainvoke("问题")`）
> - **异步生成器**：像流水线生产，逐步接收产品（`async for chunk in llm.astream("问题"):`）

> 🌰 **代码示例**：
> ```python
> # 异步函数：获取完整响应
> async def get_full_response():
>     result = await llm.ainvoke("问题")
>     return result
>
> # 异步生成器：逐步获取响应
> async def stream_response():
>     async for chunk in llm.astream("问题"):
>         print(chunk)
> ```

### 🌟 核心4方法（记住这4个就够了）

| 方法 | 记忆提示 | 使用场景 | 实际例子 |
|------|---------|---------|---------|
| `invoke()` | **基础同步调用** | 需要完整结果 | `llm.invoke("问题")` → `"答案"` |
| `stream()` | **流式输出** | 逐步显示响应 | `for chunk in llm.stream("问题"):` → 实时显示 |
| `ainvoke()` | **async + invoke** | 异步等待完整结果 | `await llm.ainvoke("问题")` |
| `astream()` | **async + stream** | 异步流式处理 | `async for chunk in llm.astream("问题"):` |

---

### 📤 返回值速查（重要！）

**所有方法返回的都是 `BaseMessage` 对象（或其迭代器/异步迭代器）**

| 常用属性 | 作用 | 示例取值 |
|---------|------|---------|
| `.content` | **文本内容**（最常用） | `result.content` → `"这是LLM的回答"` |
| `.response_metadata` | 响应元数据 | `result.response_metadata` → `{'token_usage': {...}}` |
| `.tool_calls` | 工具调用信息 | `result.tool_calls` → `[{'name': 'search', ...}]` |
| `.id` | 消息唯一ID | `result.id` → `"run-abc123..."` |

> 💡 **记忆口诀**: "content拿内容,metadata看详情,tool_calls查工具"

---

### 🌰 实际取值示例

**同步获取内容**:
```python
# invoke() - 直接取值
result = llm.invoke("问题")
print(result.content)  # "这是完整回答"

# stream() - 逐块取值
for chunk in llm.stream("问题"):
    print(chunk.content, end="")  # 实时流式输出
```

**异步获取内容**:
```python
# ainvoke() - await后取值
result = await llm.ainvoke("问题")
print(result.content)  # "这是完整回答"

# astream() - async for逐块取值
async for chunk in llm.astream("问题"):
    print(chunk.content, end="")  # 实时流式输出
```

**获取元数据**:
```python
result = llm.invoke("问题")
print(result.response_metadata)
# {'token_usage': {'prompt_tokens': 10, 'completion_tokens': 20}, ...}
```

---

### 📦 扩展方法（了解即可）

| 方法 | 与核心方法的关系 | 适用场景 |
|------|-----------------|---------|
| `batch()` | = 多个`invoke()` | 批量处理多个输入 |
| `abatch()` | = 多个`ainvoke()` | 异步批量处理 |
| `stream_log()` | = `stream()` + 详细日志 | 调试工作流 |
| `astream_log()` | = `astream()` + 详细日志 | 异步调试 |

> ✅ **终极记忆法**：每天花2分钟看这张卡片，连续3天就能记住！