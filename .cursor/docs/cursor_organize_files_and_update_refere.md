# Organize files and update references
_Exported on 11/30/2025 at 11:08:16 GMT+8 from Cursor (2.0.77)_


---

**User**

@Langchain @LangGraph @akshare 我需要做一个智能分析系统，给我一些开发设计建议。

我现在有从百度网络下载下来的知识文件，一个知识块有1-3个知识文件，你可以在/Users/zhou/Project/AnalystChain/jupyter_notebook/macroeconomic_analysis/knowledge_base目录下看到这些文件。我需要读取、总结这些文件里的知识内容成为有利于langchain读取的并且完整的知识。而且是1次完成生成，而不是每次都进行读取总结。

后续我将向Langchain提出我的投资分析诉求，langchain在读取这些知识之后，在这些已知知识的指导下，自主选择、调用AKShare合适的接口，从而获得数据、分析、给出投资建议。

给出建议：

1. 依据文件特点，如何设计这个读取程序比较好？

2. 我的智能投资分析程序的过程合理吗？目前我这知识目录下是之后宏观分析的知识，后续还有会其他目录和知识文件比如政策、股票分析等，所以我打算用langchian的deepagent的技术来做这个投资系统，每次读取总结知识后，每个层面的分析都是1个subagent，提出各自比如宏观、政策层面等角度的分析结果，最后有有整合、总结、分析subagent给出最终结果到主agnet，是否合理？有没有什么更好的建议？



你依旧按原来的要求调用mcp-feedback-enhanced



---

**Cursor**

我来帮你分析这个智能投资分析系统的设计。首先让我查看一下你的知识库目录结构。

基于你的需求和目录结构,我给出以下设计建议:

## 一、知识文件处理流程设计

### 1. 文件特点分析
你的每个知识块包含3个文件:
- `.pptx` - PPT演示文件
- `.pdf` - 笔记文件(图文混排,OCR识别)
- `.doc` - Word文档

**建议的处理策略:**

```python
# 推荐处理优先级:
1. PDF笔记 -> 主要知识来源(结构化好、内容完整)
2. Word文档 -> 补充详细说明
3. PPT -> 辅助理解(可选,图表多但文字少)
```

### 2. 知识处理Pipeline设计

```
知识文件 → 文档加载 → 文本提取 → 智能总结 → 向量化存储 → 持久化
    ↓
知识索引(Metadata + Embeddings) → 供Agent检索使用
```

**推荐的实现方案:**

```python
# 第一阶段:文档处理与总结
- 使用 LangChain DocumentLoader 加载文档
- 按知识块分组处理(同一主题的3个文件一起处理)
- 用 LLM 生成结构化知识摘要(包括核心概念、指标、应用场景)
- 保存为 JSON 格式(方便后续使用)

# 第二阶段:向量化与索引
- 使用 RecursiveCharacterTextSplitter 切块
- 用 Embeddings 模型向量化
- 存入 ChromaDB/FAISS 持久化向量库
- 添加丰富的 metadata(知识类型、主题、时间等)
```

**一次性生成方案:**

```python
def process_knowledge_base():
    """一次性处理所有知识文件"""

    # 1. 扫描目录,按知识块分组
    knowledge_blocks = group_files_by_topic()

    # 2. 逐块处理
    for block in knowledge_blocks:
        # 2.1 加载文档
        docs = load_documents(block)

        # 2.2 LLM总结提取结构化知识
        structured_knowledge = llm_summarize(docs)

        # 2.3 保存JSON(方便直接读取)
        save_json(structured_knowledge, f"{block.name}_knowledge.json")

        # 2.4 向量化存储(方便检索)
        chunks = text_splitter.split_documents(docs)
        vector_store.add_documents(chunks, metadata=block.metadata)

    # 3. 持久化向量库
    vector_store.persist()
```

## 二、智能投资分析系统架构建议

### 你当前的设计思路:
```
用户诉求 → 主Agent → SubAgents(宏观/政策/股票) → 整合Agent → 最终结果
                ↓
            调用AKShare获取数据
```

### 我的优化建议:

**架构方案:LangGraph Multi-Agent系统**

```python
系统分层:

1. 【知识层】Knowledge Base Layer
   - 宏观经济知识库(你现在的)
   - 政策分析知识库
   - 股票技术分析知识库
   - 每个知识库独立管理、独立向量存储

2. 【工具层】Tools Layer
   - AKShare工具集(数据获取)
   - 计算分析工具(技术指标计算)
   - 知识检索工具(RAG检索)

3. 【Agent层】Multi-Agent Layer
   采用 LangGraph 的 StateGraph:

   StateGraph结构:
   START → 意图理解节点 → 路由节点 → 并行执行SubAgents
                                    ↓
                         宏观Agent | 政策Agent | 股票Agent
                                    ↓
                         汇总节点 → 整合Agent → END
```

### 推荐的技术架构:

```python
from langgraph.graph import StateGraph, END
from langchain.agents import create_react_agent

# 1. 定义全局状态
class AnalysisState(TypedDict):
    user_query: str
    macro_analysis: dict
    policy_analysis: dict
    stock_analysis: dict
    final_recommendation: str
    akshare_data: dict

# 2. 定义各个Agent
class MacroAgent:
    """宏观经济分析Agent"""
    def __init__(self):
        self.knowledge_base = load_macro_knowledge()
        self.tools = [
            create_retriever_tool(self.knowledge_base),
            akshare_macro_tools,  # GDP、CPI、PMI等
        ]
        self.agent = create_react_agent(llm, self.tools)

    def analyze(self, state: AnalysisState):
        # 1. 从知识库检索相关理论
        # 2. 调用AKShare获取数据
        # 3. 基于知识分析数据
        # 4. 返回分析结果
        pass

class PolicyAgent:
    """政策分析Agent"""
    # 类似结构

class StockAgent:
    """股票分析Agent"""
    # 类似结构

class IntegrationAgent:
    """整合分析Agent"""
    def integrate(self, state: AnalysisState):
        # 整合各个维度的分析
        # 给出综合投资建议
        pass

# 3. 构建Graph
def build_analysis_graph():
    workflow = StateGraph(AnalysisState)

    # 添加节点
    workflow.add_node("macro", macro_agent.analyze)
    workflow.add_node("policy", policy_agent.analyze)
    workflow.add_node("stock", stock_agent.analyze)
    workflow.add_node("integrate", integration_agent.integrate)

    # 并行执行
    workflow.add_edge(START, "macro")
    workflow.add_edge(START, "policy")
    workflow.add_edge(START, "stock")

    # 汇总
    workflow.add_edge("macro", "integrate")
    workflow.add_edge("policy", "integrate")
    workflow.add_edge("stock", "integrate")
    workflow.add_edge("integrate", END)

    return workflow.compile()
```

### 为什么这样设计更好:

1. **模块化解耦**: 每个知识域独立,方便扩展和维护
2. **并行执行**: LangGraph支持并行节点,加快分析速度
3. **状态管理**: StateGraph统一管理分析状态,数据流清晰
4. **工具复用**: AKShare工具可以被多个Agent共享
5. **可追溯性**: LangGraph天然支持执行追踪,方便调试

### 目录结构建议:

```
AnalystChain/
├── src/langchain_app/
│   ├── agents/
│   │   ├── macro_agent.py        # 宏观分析Agent
│   │   ├── policy_agent.py       # 政策分析Agent
│   │   ├── stock_agent.py        # 股票分析Agent
│   │   └── integration_agent.py  # 整合Agent
│   ├── tools/
│   │   ├── akshare_tools.py      # AKShare工具封装
│   │   └── knowledge_retriever.py # 知识检索工具
│   ├── knowledge/
│   │   ├── processor.py          # 知识处理Pipeline
│   │   └── vector_store.py       # 向量库管理
│   └── graphs/
│       └── analysis_graph.py     # LangGraph主流程
├── jupyter_notebook/
│   └── macroeconomic_analysis/
│       ├── knowledge_base/       # 原始知识文件
│       ├── processed/            # 处理后的JSON知识
│       └── vector_db/            # 向量数据库
```

### 关键优化点:

1. **知识预处理**: 一次性处理所有知识文件,生成结构化JSON + 向量库
2. **智能路由**: 根据用户问题自动选择需要调用哪些SubAgent
3. **增量更新**: 新增知识文件时,只处理新文件,不影响已有知识
4. **结果缓存**: 相同的数据请求结果可以缓存,避免重复调用AKShare

这样的架构既清晰又灵活,后续扩展其他分析维度也很容易。你觉得这个方案如何?

```python
# 推荐处理优先级:
1. PDF笔记 -> 主要知识来源(结构化好、内容完整)
2. Word文档 -> 补充详细说明
3. PPT -> 辅助理解(可选,图表多但文字少)
```

```plaintext
知识文件 → 文档加载 → 文本提取 → 智能总结 → 向量化存储 → 持久化
    ↓
知识索引(Metadata + Embeddings) → 供Agent检索使用
```

```python
# 第一阶段:文档处理与总结
- 使用 LangChain DocumentLoader 加载文档
- 按知识块分组处理(同一主题的3个文件一起处理)
- 用 LLM 生成结构化知识摘要(包括核心概念、指标、应用场景)
- 保存为 JSON 格式(方便后续使用)

# 第二阶段:向量化与索引
- 使用 RecursiveCharacterTextSplitter 切块
- 用 Embeddings 模型向量化
- 存入 ChromaDB/FAISS 持久化向量库
- 添加丰富的 metadata(知识类型、主题、时间等)
```

```python
def process_knowledge_base():
    """一次性处理所有知识文件"""

    # 1. 扫描目录,按知识块分组
    knowledge_blocks = group_files_by_topic()

    # 2. 逐块处理
    for block in knowledge_blocks:
        # 2.1 加载文档
        docs = load_documents(block)

        # 2.2 LLM总结提取结构化知识
        structured_knowledge = llm_summarize(docs)

        # 2.3 保存JSON(方便直接读取)
        save_json(structured_knowledge, f"{block.name}_knowledge.json")

        # 2.4 向量化存储(方便检索)
        chunks = text_splitter.split_documents(docs)
        vector_store.add_documents(chunks, metadata=block.metadata)

    # 3. 持久化向量库
    vector_store.persist()
```

```plaintext
用户诉求 → 主Agent → SubAgents(宏观/政策/股票) → 整合Agent → 最终结果
                ↓
            调用AKShare获取数据
```

```python
系统分层:

1. 【知识层】Knowledge Base Layer
   - 宏观经济知识库(你现在的)
   - 政策分析知识库
   - 股票技术分析知识库
   - 每个知识库独立管理、独立向量存储

2. 【工具层】Tools Layer
   - AKShare工具集(数据获取)
   - 计算分析工具(技术指标计算)
   - 知识检索工具(RAG检索)

3. 【Agent层】Multi-Agent Layer
   采用 LangGraph 的 StateGraph:

   StateGraph结构:
   START → 意图理解节点 → 路由节点 → 并行执行SubAgents
                                    ↓
                         宏观Agent | 政策Agent | 股票Agent
                                    ↓
                         汇总节点 → 整合Agent → END
```

```python
from langgraph.graph import StateGraph, END
from langchain.agents import create_react_agent

# 1. 定义全局状态
class AnalysisState(TypedDict):
    user_query: str
    macro_analysis: dict
    policy_analysis: dict
    stock_analysis: dict
    final_recommendation: str
    akshare_data: dict

# 2. 定义各个Agent
class MacroAgent:
    """宏观经济分析Agent"""
    def __init__(self):
        self.knowledge_base = load_macro_knowledge()
        self.tools = [
            create_retriever_tool(self.knowledge_base),
            akshare_macro_tools,  # GDP、CPI、PMI等
        ]
        self.agent = create_react_agent(llm, self.tools)

    def analyze(self, state: AnalysisState):
        # 1. 从知识库检索相关理论
        # 2. 调用AKShare获取数据
        # 3. 基于知识分析数据
        # 4. 返回分析结果
        pass

class PolicyAgent:
    """政策分析Agent"""
    # 类似结构

class StockAgent:
    """股票分析Agent"""
    # 类似结构

class IntegrationAgent:
    """整合分析Agent"""
    def integrate(self, state: AnalysisState):
        # 整合各个维度的分析
        # 给出综合投资建议
        pass

# 3. 构建Graph
def build_analysis_graph():
    workflow = StateGraph(AnalysisState)

    # 添加节点
    workflow.add_node("macro", macro_agent.analyze)
    workflow.add_node("policy", policy_agent.analyze)
    workflow.add_node("stock", stock_agent.analyze)
    workflow.add_node("integrate", integration_agent.integrate)

    # 并行执行
    workflow.add_edge(START, "macro")
    workflow.add_edge(START, "policy")
    workflow.add_edge(START, "stock")

    # 汇总
    workflow.add_edge("macro", "integrate")
    workflow.add_edge("policy", "integrate")
    workflow.add_edge("stock", "integrate")
    workflow.add_edge("integrate", END)

    return workflow.compile()
```

```plaintext
AnalystChain/
├── src/langchain_app/
│   ├── agents/
│   │   ├── macro_agent.py        # 宏观分析Agent
│   │   ├── policy_agent.py       # 政策分析Agent
│   │   ├── stock_agent.py        # 股票分析Agent
│   │   └── integration_agent.py  # 整合Agent
│   ├── tools/
│   │   ├── akshare_tools.py      # AKShare工具封装
│   │   └── knowledge_retriever.py # 知识检索工具
│   ├── knowledge/
│   │   ├── processor.py          # 知识处理Pipeline
│   │   └── vector_store.py       # 向量库管理
│   └── graphs/
│       └── analysis_graph.py     # LangGraph主流程
├── jupyter_notebook/
│   └── macroeconomic_analysis/
│       ├── knowledge_base/       # 原始知识文件
│       ├── processed/            # 处理后的JSON知识
│       └── vector_db/            # 向量数据库
```

非常好!在研究了DeepAgents的文档后,我需要**大幅更新**我的设计建议。DeepAgents的架构非常适合你的投资分析系统!

## 📊 基于DeepAgents的全新设计方案

### 核心架构变化

之前我建议用StateGraph手动管理并行执行,但**DeepAgents已经内置了更优雅的解决方案**:

```python
用户投资诉求
    ↓
主Agent (Investment Advisor)
    ├─ 内置TodoList (任务分解规划)
    ├─ 内置Filesystem (上下文管理)
    └─ task工具 (委派SubAgents)
        ↓
    ┌───┴───┬───────┬────────┐
    ↓       ↓       ↓        ↓
 宏观Agent 政策Agent 股票Agent (可并行)
    ↓       ↓       ↓
  返回结果 → 主Agent整合 → 最终投资建议
```

### 一、知识文件处理的新方案

**利用DeepAgents的Filesystem持久化机制:**

```python
from deepagents import create_deep_agent
from deepagents.middleware import FilesystemMiddleware
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langgraph.store.memory import InMemoryStore

# 知识处理Pipeline
def process_knowledge_once():
    """
    一次性处理知识文件,存储到/memories/路径实现持久化
    """

    # 1. 按知识块分组处理
    knowledge_blocks = {
        "macro_economy": [
            "01第一节 中国经济的三驾马车_笔记.pdf",
            "02第二节 消费_笔记.pdf",
            # ...
        ],
        # 未来扩展: "policy": [...], "stock_analysis": [...]
    }

    # 2. 为每个知识块生成结构化知识
    for domain, files in knowledge_blocks.items():
        # 2.1 加载文档
        docs = load_documents_from_files(files)

        # 2.2 用LLM提取结构化知识
        structured_knowledge = {
            "domain": domain,
            "concepts": extract_concepts(docs),
            "indicators": extract_indicators(docs),
            "analysis_methods": extract_methods(docs),
            "practical_guidance": extract_guidance(docs),
        }

        # 2.3 保存到/memories/路径(持久化)
        save_to_filesystem(
            path=f"/memories/knowledge/{domain}/structured.json",
            content=json.dumps(structured_knowledge, ensure_ascii=False)
        )

        # 2.4 创建向量索引(用于RAG检索)
        chunks = text_splitter.split_documents(docs)
        vector_store = Chroma(
            persist_directory=f"./vector_db/{domain}",
            embedding_function=embeddings
        )
        vector_store.add_documents(
            chunks,
            metadata={"domain": domain, "type": "knowledge"}
        )
        vector_store.persist()

    print("✅ 知识处理完成,已持久化到/memories/目录")
```

**目录结构:**

```
/memories/
├── knowledge/
│   ├── macro_economy/
│   │   ├── structured.json      # 结构化知识
│   │   └── summary.md           # 知识摘要
│   ├── policy/
│   │   └── structured.json
│   └── stock_analysis/
│       └── structured.json
└── analysis_history/
    └── user_preferences.json     # 用户投资偏好记忆
```

### 二、智能投资分析系统架构(DeepAgents版)

#### 完整实现代码:

```python
from deepagents import create_deep_agent, CompiledSubAgent
from deepagents.middleware import FilesystemMiddleware
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langgraph.store.memory import InMemoryStore
from langchain_community.tools import tool
import akshare as ak

# ============ 1. 准备工具集 ============

# AKShare工具封装
@tool
def get_macro_data(indicator: str, start_date: str = None, end_date: str = None):
    """
    获取宏观经济数据
    indicator: GDP, CPI, PMI等
    """
    if indicator == "GDP":
        return ak.macro_china_gdp()
    elif indicator == "CPI":
        return ak.macro_china_cpi()
    elif indicator == "PMI":
        return ak.macro_china_pmi()
    # ...更多指标

@tool
def get_stock_data(stock_code: str, period: str = "daily"):
    """获取股票数据"""
    return ak.stock_zh_a_hist(symbol=stock_code, period=period)

@tool
def get_policy_news(keywords: str):
    """获取政策新闻"""
    # 调用AKShare的新闻接口
    return ak.news_cctv(date="20250101")  # 示例

# 知识检索工具
def create_knowledge_retriever(domain: str):
    """为特定领域创建知识检索工具"""
    vector_store = Chroma(
        persist_directory=f"./vector_db/{domain}",
        embedding_function=embeddings
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    @tool
    def search_knowledge(query: str) -> str:
        f"""在{domain}知识库中检索相关理论和方法"""
        docs = retriever.get_relevant_documents(query)
        return "\n\n".join([doc.page_content for doc in docs])

    return search_knowledge

# ============ 2. 定义SubAgents ============

# 宏观经济分析SubAgent
macro_subagent_config = {
    "name": "macro-analyst",
    "description": "专门进行宏观经济分析,评估经济周期和趋势",
    "system_prompt": """
你是一位宏观经济分析专家。你的任务是:

1. **从/memories/knowledge/macro_economy/读取已学习的理论知识**
2. **使用AKShare工具获取最新的宏观数据**(GDP、CPI、PMI等)
3. **基于理论知识分析当前经济形势**
4. **评估经济周期阶段**(复苏、繁荣、衰退、萧条)
5. **给出宏观层面的资产配置建议**

分析框架:
- 三驾马车分析(消费、投资、出口)
- 通胀水平评估
- 货币政策取向
- 投资时钟判断

输出格式:
{
  "economic_cycle": "当前周期阶段",
  "key_indicators": {指标名称: 数值},
  "analysis": "详细分析",
  "recommendation": "资产配置建议"
}
""",
    "tools": [
        get_macro_data,
        create_knowledge_retriever("macro_economy"),
    ],
    "model": "deepseek-chat",  # 可以指定不同模型
}

# 政策分析SubAgent
policy_subagent_config = {
    "name": "policy-analyst",
    "description": "分析财政、货币、产业政策对投资的影响",
    "system_prompt": """
你是政策分析专家。任务是:

1. 从/memories/knowledge/policy/读取政策分析知识
2. 获取最新政策动态和新闻
3. 评估政策对各类资产的影响
4. 识别政策红利和风险

输出: 政策面的投资建议
""",
    "tools": [
        get_policy_news,
        create_knowledge_retriever("policy"),
    ],
}

# 股票分析SubAgent
stock_subagent_config = {
    "name": "stock-analyst",
    "description": "进行股票技术和基本面分析",
    "system_prompt": """
你是股票分析专家。任务是:

1. 从/memories/knowledge/stock_analysis/读取分析方法
2. 获取股票数据和财务指标
3. 进行技术分析和基本面分析
4. 筛选优质投资标的

输出: 具体的股票投资建议
""",
    "tools": [
        get_stock_data,
        create_knowledge_retriever("stock_analysis"),
    ],
}

# ============ 3. 创建主Agent ============

# 配置持久化存储
store = InMemoryStore()

# 主投资顾问Agent
investment_advisor = create_deep_agent(
    model="deepseek-chat",

    # 系统提示词
    system_prompt="""
你是一位专业的智能投资顾问。你可以:

## 核心能力

1. **任务规划**: 使用write_todos工具将复杂的投资分析任务分解为步骤
2. **知识管理**: 使用文件系统工具读取/memories/knowledge/下的领域知识
3. **委派专家**: 使用task工具委派给专门的SubAgent:
   - macro-analyst: 宏观经济分析
   - policy-analyst: 政策分析
   - stock-analyst: 股票分析
4. **综合决策**: 整合各SubAgent的分析结果,给出全面的投资建议

## 工作流程

当用户提出投资诉求时:

1. 使用write_todos分解任务(如: 宏观分析→政策分析→标的选择→风险评估)
2. 使用read_file读取相关领域的知识背景
3. 使用task工具并行委派给各SubAgent
4. 等待SubAgent返回结果
5. 综合分析,给出:
   - 当前市场环境评估
   - 资产配置建议(比例)
   - 具体投资标的推荐
   - 风险提示
6. 使用write_file将分析报告保存到/memories/analysis_history/

## 输出格式

最终报告应包含:
- 📊 市场环境分析
- 💰 资产配置方案
- 🎯 投资标的推荐
- ⚠️ 风险提示
""",

    # SubAgents配置
    subagents=[
        macro_subagent_config,
        policy_subagent_config,
        stock_subagent_config,
    ],

    # 主Agent自己的工具(可以添加额外工具)
    tools=[],

    # 配置持久化存储
    store=store,

    # 文件系统后端(支持长期记忆)
    backend=lambda config: CompositeBackend(
        default=StateBackend(config),
        routes={
            "/memories/": StoreBackend(config)  # /memories/路径持久化
        }
    ),
)

# ============ 4. 使用示例 ============

# 用户提出投资诉求
user_query = """
我有100万资金,想进行投资配置。
我的风险偏好是稳健型,希望获得年化8-12%的收益。
请帮我分析当前市场环境,给出具体的配置方案。
"""

config = {"configurable": {"thread_id": "user_001"}}

# 调用主Agent
for chunk in investment_advisor.stream(
    {"messages": [{"role": "user", "content": user_query}]},
    config=config,
    stream_mode="values"
):
    print(chunk["messages"][-1].content)

# Agent会自动:
# 1. write_todos: 分解任务
# 2. read_file: 读取知识
# 3. task("macro-analyst", "分析当前宏观经济形势")
# 4. task("policy-analyst", "分析政策影响")
# 5. task("stock-analyst", "推荐具体标的")
# 6. 综合分析,生成报告
# 7. write_file: 保存报告到/memories/
```

### 三、为什么DeepAgents方案更优

对比之前的StateGraph方案:

| 特性 | StateGraph方案 | **DeepAgents方案** ✅ |
|------|---------------|---------------------|
| **任务规划** | 需要手动定义 | ✅ 内置TodoList自动分解 |
| **上下文管理** | 手动管理State | ✅ 自动Filesystem offload |
| **SubAgent调用** | 手动定义节点和边 | ✅ 简单的task工具调用 |
| **并行执行** | 需要手动设计 | ✅ 自动并行执行SubAgent |
| **记忆持久化** | 需要自己实现 | ✅ 内置/memories/持久化 |
| **上下文隔离** | 容易污染 | ✅ SubAgent独立上下文 |
| **代码复杂度** | 高(100+行) | ✅ 低(配置化,30行) |

### 四、目录结构建议(更新)

```
AnalystChain/
├── src/langchain_app/
│   ├── agents/
│   │   ├── investment_advisor.py    # 主DeepAgent
│   │   └── subagents/
│   │       ├── macro_analyst.py     # 宏观分析SubAgent配置
│   │       ├── policy_analyst.py    # 政策分析SubAgent配置
│   │       └── stock_analyst.py     # 股票分析SubAgent配置
│   ├── tools/
│   │   ├── akshare_tools.py         # AKShare工具封装
│   │   └── knowledge_retriever.py   # 知识检索工具
│   ├── knowledge/
│   │   ├── processor.py             # 知识处理Pipeline
│   │   └── run_once.py              # 一次性处理脚本
│   └── memories/                    # DeepAgents持久化目录
│       ├── knowledge/               # 知识库(持久化)
│       │   ├── macro_economy/
│       │   ├── policy/
│       │   └── stock_analysis/
│       └── analysis_history/        # 历史分析记录
├── vector_db/                       # 向量数据库
│   ├── macro_economy/
│   ├── policy/
│   └── stock_analysis/
└── jupyter_notebook/
    └── macroeconomic_analysis/
        └── knowledge_base/          # 原始知识文件
```

### 五、实施步骤

1. **第一阶段:知识处理**(一次性)
   ```bash
   python src/langchain_app/knowledge/run_once.py
   # 处理所有知识文件,生成:
   # - /memories/knowledge/* (持久化JSON)
   # - vector_db/* (向量索引)
   ```

2. **第二阶段:SubAgent开发**
   - 实现3个SubAgent的配置
   - 为每个SubAgent配置专属工具

3. **第三阶段:主Agent集成**
   - 创建主DeepAgent
   - 测试SubAgent调用
   - 优化系统提示词

4. **第四阶段:扩展**
   - 新增knowledge_base子目录时
   - 只需重新运行processor处理新知识
   - 新增SubAgent配置即可

### 总结

✅ **你的直觉是对的** - DeepAgents确实非常适合你的场景!

**关键优势:**
- 🎯 **开箱即用**的Planning、Filesystem、SubAgent能力
- 🚀 **大幅简化代码**,从手动StateGraph到配置化
- 💾 **天然支持持久化记忆**,知识处理一次,永久使用
- 🔄 **自动上下文管理**,不会爆Context window
- 🎨 **极其灵活**,后续扩展新维度只需添加SubAgent配置

这个方案比我之前的设计**更简洁、更强大、更易维护**!

```python
用户投资诉求
    ↓
主Agent (Investment Advisor)
    ├─ 内置TodoList (任务分解规划)
    ├─ 内置Filesystem (上下文管理)
    └─ task工具 (委派SubAgents)
        ↓
    ┌───┴───┬───────┬────────┐
    ↓       ↓       ↓        ↓
 宏观Agent 政策Agent 股票Agent (可并行)
    ↓       ↓       ↓
  返回结果 → 主Agent整合 → 最终投资建议
```

```python
from deepagents import create_deep_agent
from deepagents.middleware import FilesystemMiddleware
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langgraph.store.memory import InMemoryStore

# 知识处理Pipeline
def process_knowledge_once():
    """
    一次性处理知识文件,存储到/memories/路径实现持久化
    """

    # 1. 按知识块分组处理
    knowledge_blocks = {
        "macro_economy": [
            "01第一节 中国经济的三驾马车_笔记.pdf",
            "02第二节 消费_笔记.pdf",
            # ...
        ],
        # 未来扩展: "policy": [...], "stock_analysis": [...]
    }

    # 2. 为每个知识块生成结构化知识
    for domain, files in knowledge_blocks.items():
        # 2.1 加载文档
        docs = load_documents_from_files(files)

        # 2.2 用LLM提取结构化知识
        structured_knowledge = {
            "domain": domain,
            "concepts": extract_concepts(docs),
            "indicators": extract_indicators(docs),
            "analysis_methods": extract_methods(docs),
            "practical_guidance": extract_guidance(docs),
        }

        # 2.3 保存到/memories/路径(持久化)
        save_to_filesystem(
            path=f"/memories/knowledge/{domain}/structured.json",
            content=json.dumps(structured_knowledge, ensure_ascii=False)
        )

        # 2.4 创建向量索引(用于RAG检索)
        chunks = text_splitter.split_documents(docs)
        vector_store = Chroma(
            persist_directory=f"./vector_db/{domain}",
            embedding_function=embeddings
        )
        vector_store.add_documents(
            chunks,
            metadata={"domain": domain, "type": "knowledge"}
        )
        vector_store.persist()

    print("✅ 知识处理完成,已持久化到/memories/目录")
```

```plaintext
/memories/
├── knowledge/
│   ├── macro_economy/
│   │   ├── structured.json      # 结构化知识
│   │   └── summary.md           # 知识摘要
│   ├── policy/
│   │   └── structured.json
│   └── stock_analysis/
│       └── structured.json
└── analysis_history/
    └── user_preferences.json     # 用户投资偏好记忆
```

```python
from deepagents import create_deep_agent, CompiledSubAgent
from deepagents.middleware import FilesystemMiddleware
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langgraph.store.memory import InMemoryStore
from langchain_community.tools import tool
import akshare as ak

# ============ 1. 准备工具集 ============

# AKShare工具封装
@tool
def get_macro_data(indicator: str, start_date: str = None, end_date: str = None):
    """
    获取宏观经济数据
    indicator: GDP, CPI, PMI等
    """
    if indicator == "GDP":
        return ak.macro_china_gdp()
    elif indicator == "CPI":
        return ak.macro_china_cpi()
    elif indicator == "PMI":
        return ak.macro_china_pmi()
    # ...更多指标

@tool
def get_stock_data(stock_code: str, period: str = "daily"):
    """获取股票数据"""
    return ak.stock_zh_a_hist(symbol=stock_code, period=period)

@tool
def get_policy_news(keywords: str):
    """获取政策新闻"""
    # 调用AKShare的新闻接口
    return ak.news_cctv(date="20250101")  # 示例

# 知识检索工具
def create_knowledge_retriever(domain: str):
    """为特定领域创建知识检索工具"""
    vector_store = Chroma(
        persist_directory=f"./vector_db/{domain}",
        embedding_function=embeddings
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    @tool
    def search_knowledge(query: str) -> str:
        f"""在{domain}知识库中检索相关理论和方法"""
        docs = retriever.get_relevant_documents(query)
        return "\n\n".join([doc.page_content for doc in docs])

    return search_knowledge

# ============ 2. 定义SubAgents ============

# 宏观经济分析SubAgent
macro_subagent_config = {
    "name": "macro-analyst",
    "description": "专门进行宏观经济分析,评估经济周期和趋势",
    "system_prompt": """
你是一位宏观经济分析专家。你的任务是:

1. **从/memories/knowledge/macro_economy/读取已学习的理论知识**
2. **使用AKShare工具获取最新的宏观数据**(GDP、CPI、PMI等)
3. **基于理论知识分析当前经济形势**
4. **评估经济周期阶段**(复苏、繁荣、衰退、萧条)
5. **给出宏观层面的资产配置建议**

分析框架:
- 三驾马车分析(消费、投资、出口)
- 通胀水平评估
- 货币政策取向
- 投资时钟判断

输出格式:
{
  "economic_cycle": "当前周期阶段",
  "key_indicators": {指标名称: 数值},
  "analysis": "详细分析",
  "recommendation": "资产配置建议"
}
""",
    "tools": [
        get_macro_data,
        create_knowledge_retriever("macro_economy"),
    ],
    "model": "deepseek-chat",  # 可以指定不同模型
}

# 政策分析SubAgent
policy_subagent_config = {
    "name": "policy-analyst",
    "description": "分析财政、货币、产业政策对投资的影响",
    "system_prompt": """
你是政策分析专家。任务是:

1. 从/memories/knowledge/policy/读取政策分析知识
2. 获取最新政策动态和新闻
3. 评估政策对各类资产的影响
4. 识别政策红利和风险

输出: 政策面的投资建议
""",
    "tools": [
        get_policy_news,
        create_knowledge_retriever("policy"),
    ],
}

# 股票分析SubAgent
stock_subagent_config = {
    "name": "stock-analyst",
    "description": "进行股票技术和基本面分析",
    "system_prompt": """
你是股票分析专家。任务是:

1. 从/memories/knowledge/stock_analysis/读取分析方法
2. 获取股票数据和财务指标
3. 进行技术分析和基本面分析
4. 筛选优质投资标的

输出: 具体的股票投资建议
""",
    "tools": [
        get_stock_data,
        create_knowledge_retriever("stock_analysis"),
    ],
}

# ============ 3. 创建主Agent ============

# 配置持久化存储
store = InMemoryStore()

# 主投资顾问Agent
investment_advisor = create_deep_agent(
    model="deepseek-chat",

    # 系统提示词
    system_prompt="""
你是一位专业的智能投资顾问。你可以:

## 核心能力

1. **任务规划**: 使用write_todos工具将复杂的投资分析任务分解为步骤
2. **知识管理**: 使用文件系统工具读取/memories/knowledge/下的领域知识
3. **委派专家**: 使用task工具委派给专门的SubAgent:
   - macro-analyst: 宏观经济分析
   - policy-analyst: 政策分析
   - stock-analyst: 股票分析
4. **综合决策**: 整合各SubAgent的分析结果,给出全面的投资建议

## 工作流程

当用户提出投资诉求时:

1. 使用write_todos分解任务(如: 宏观分析→政策分析→标的选择→风险评估)
2. 使用read_file读取相关领域的知识背景
3. 使用task工具并行委派给各SubAgent
4. 等待SubAgent返回结果
5. 综合分析,给出:
   - 当前市场环境评估
   - 资产配置建议(比例)
   - 具体投资标的推荐
   - 风险提示
6. 使用write_file将分析报告保存到/memories/analysis_history/

## 输出格式

最终报告应包含:
- 📊 市场环境分析
- 💰 资产配置方案
- 🎯 投资标的推荐
- ⚠️ 风险提示
""",

    # SubAgents配置
    subagents=[
        macro_subagent_config,
        policy_subagent_config,
        stock_subagent_config,
    ],

    # 主Agent自己的工具(可以添加额外工具)
    tools=[],

    # 配置持久化存储
    store=store,

    # 文件系统后端(支持长期记忆)
    backend=lambda config: CompositeBackend(
        default=StateBackend(config),
        routes={
            "/memories/": StoreBackend(config)  # /memories/路径持久化
        }
    ),
)

# ============ 4. 使用示例 ============

# 用户提出投资诉求
user_query = """
我有100万资金,想进行投资配置。
我的风险偏好是稳健型,希望获得年化8-12%的收益。
请帮我分析当前市场环境,给出具体的配置方案。
"""

config = {"configurable": {"thread_id": "user_001"}}

# 调用主Agent
for chunk in investment_advisor.stream(
    {"messages": [{"role": "user", "content": user_query}]},
    config=config,
    stream_mode="values"
):
    print(chunk["messages"][-1].content)

# Agent会自动:
# 1. write_todos: 分解任务
# 2. read_file: 读取知识
# 3. task("macro-analyst", "分析当前宏观经济形势")
# 4. task("policy-analyst", "分析政策影响")
# 5. task("stock-analyst", "推荐具体标的")
# 6. 综合分析,生成报告
# 7. write_file: 保存报告到/memories/
```

```plaintext
AnalystChain/
├── src/langchain_app/
│   ├── agents/
│   │   ├── investment_advisor.py    # 主DeepAgent
│   │   └── subagents/
│   │       ├── macro_analyst.py     # 宏观分析SubAgent配置
│   │       ├── policy_analyst.py    # 政策分析SubAgent配置
│   │       └── stock_analyst.py     # 股票分析SubAgent配置
│   ├── tools/
│   │   ├── akshare_tools.py         # AKShare工具封装
│   │   └── knowledge_retriever.py   # 知识检索工具
│   ├── knowledge/
│   │   ├── processor.py             # 知识处理Pipeline
│   │   └── run_once.py              # 一次性处理脚本
│   └── memories/                    # DeepAgents持久化目录
│       ├── knowledge/               # 知识库(持久化)
│       │   ├── macro_economy/
│       │   ├── policy/
│       │   └── stock_analysis/
│       └── analysis_history/        # 历史分析记录
├── vector_db/                       # 向量数据库
│   ├── macro_economy/
│   ├── policy/
│   └── stock_analysis/
└── jupyter_notebook/
    └── macroeconomic_analysis/
        └── knowledge_base/          # 原始知识文件
```

```shellscript
   python src/langchain_app/knowledge/run_once.py
   # 处理所有知识文件,生成:
   # - /memories/knowledge/* (持久化JSON)
   # - vector_db/* (向量索引)
```

```python
用户投资诉求
    ↓
主Agent (Investment Advisor)
    ├─ 内置TodoList (任务分解规划)
    ├─ 内置Filesystem (上下文管理)
    └─ task工具 (委派SubAgents)
        ↓
    ┌───┴───┬───────┬────────┐
    ↓       ↓       ↓        ↓
 宏观Agent 政策Agent 股票Agent (可并行)
    ↓       ↓       ↓
  返回结果 → 主Agent整合 → 最终投资建议
```

```python
from deepagents import create_deep_agent
from deepagents.middleware import FilesystemMiddleware
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langgraph.store.memory import InMemoryStore

# 知识处理Pipeline
def process_knowledge_once():
    """
    一次性处理知识文件,存储到/memories/路径实现持久化
    """

    # 1. 按知识块分组处理
    knowledge_blocks = {
        "macro_economy": [
            "01第一节 中国经济的三驾马车_笔记.pdf",
            "02第二节 消费_笔记.pdf",
            # ...
        ],
        # 未来扩展: "policy": [...], "stock_analysis": [...]
    }

    # 2. 为每个知识块生成结构化知识
    for domain, files in knowledge_blocks.items():
        # 2.1 加载文档
        docs = load_documents_from_files(files)

        # 2.2 用LLM提取结构化知识
        structured_knowledge = {
            "domain": domain,
            "concepts": extract_concepts(docs),
            "indicators": extract_indicators(docs),
            "analysis_methods": extract_methods(docs),
            "practical_guidance": extract_guidance(docs),
        }

        # 2.3 保存到/memories/路径(持久化)
        save_to_filesystem(
            path=f"/memories/knowledge/{domain}/structured.json",
            content=json.dumps(structured_knowledge, ensure_ascii=False)
        )

        # 2.4 创建向量索引(用于RAG检索)
        chunks = text_splitter.split_documents(docs)
        vector_store = Chroma(
            persist_directory=f"./vector_db/{domain}",
            embedding_function=embeddings
        )
        vector_store.add_documents(
            chunks,
            metadata={"domain": domain, "type": "knowledge"}
        )
        vector_store.persist()

    print("✅ 知识处理完成,已持久化到/memories/目录")
```

```plaintext
/memories/
├── knowledge/
│   ├── macro_economy/
│   │   ├── structured.json      # 结构化知识
│   │   └── summary.md           # 知识摘要
│   ├── policy/
│   │   └── structured.json
│   └── stock_analysis/
│       └── structured.json
└── analysis_history/
    └── user_preferences.json     # 用户投资偏好记忆
```

```python
from deepagents import create_deep_agent, CompiledSubAgent
from deepagents.middleware import FilesystemMiddleware
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langgraph.store.memory import InMemoryStore
from langchain_community.tools import tool
import akshare as ak

# ============ 1. 准备工具集 ============

# AKShare工具封装
@tool
def get_macro_data(indicator: str, start_date: str = None, end_date: str = None):
    """
    获取宏观经济数据
    indicator: GDP, CPI, PMI等
    """
    if indicator == "GDP":
        return ak.macro_china_gdp()
    elif indicator == "CPI":
        return ak.macro_china_cpi()
    elif indicator == "PMI":
        return ak.macro_china_pmi()
    # ...更多指标

@tool
def get_stock_data(stock_code: str, period: str = "daily"):
    """获取股票数据"""
    return ak.stock_zh_a_hist(symbol=stock_code, period=period)

@tool
def get_policy_news(keywords: str):
    """获取政策新闻"""
    # 调用AKShare的新闻接口
    return ak.news_cctv(date="20250101")  # 示例

# 知识检索工具
def create_knowledge_retriever(domain: str):
    """为特定领域创建知识检索工具"""
    vector_store = Chroma(
        persist_directory=f"./vector_db/{domain}",
        embedding_function=embeddings
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    @tool
    def search_knowledge(query: str) -> str:
        f"""在{domain}知识库中检索相关理论和方法"""
        docs = retriever.get_relevant_documents(query)
        return "\n\n".join([doc.page_content for doc in docs])

    return search_knowledge

# ============ 2. 定义SubAgents ============

# 宏观经济分析SubAgent
macro_subagent_config = {
    "name": "macro-analyst",
    "description": "专门进行宏观经济分析,评估经济周期和趋势",
    "system_prompt": """
你是一位宏观经济分析专家。你的任务是:

1. **从/memories/knowledge/macro_economy/读取已学习的理论知识**
2. **使用AKShare工具获取最新的宏观数据**(GDP、CPI、PMI等)
3. **基于理论知识分析当前经济形势**
4. **评估经济周期阶段**(复苏、繁荣、衰退、萧条)
5. **给出宏观层面的资产配置建议**

分析框架:
- 三驾马车分析(消费、投资、出口)
- 通胀水平评估
- 货币政策取向
- 投资时钟判断

输出格式:
{
  "economic_cycle": "当前周期阶段",
  "key_indicators": {指标名称: 数值},
  "analysis": "详细分析",
  "recommendation": "资产配置建议"
}
""",
    "tools": [
        get_macro_data,
        create_knowledge_retriever("macro_economy"),
    ],
    "model": "deepseek-chat",  # 可以指定不同模型
}

# 政策分析SubAgent
policy_subagent_config = {
    "name": "policy-analyst",
    "description": "分析财政、货币、产业政策对投资的影响",
    "system_prompt": """
你是政策分析专家。任务是:

1. 从/memories/knowledge/policy/读取政策分析知识
2. 获取最新政策动态和新闻
3. 评估政策对各类资产的影响
4. 识别政策红利和风险

输出: 政策面的投资建议
""",
    "tools": [
        get_policy_news,
        create_knowledge_retriever("policy"),
    ],
}

# 股票分析SubAgent
stock_subagent_config = {
    "name": "stock-analyst",
    "description": "进行股票技术和基本面分析",
    "system_prompt": """
你是股票分析专家。任务是:

1. 从/memories/knowledge/stock_analysis/读取分析方法
2. 获取股票数据和财务指标
3. 进行技术分析和基本面分析
4. 筛选优质投资标的

输出: 具体的股票投资建议
""",
    "tools": [
        get_stock_data,
        create_knowledge_retriever("stock_analysis"),
    ],
}

# ============ 3. 创建主Agent ============

# 配置持久化存储
store = InMemoryStore()

# 主投资顾问Agent
investment_advisor = create_deep_agent(
    model="deepseek-chat",

    # 系统提示词
    system_prompt="""
你是一位专业的智能投资顾问。你可以:

## 核心能力

1. **任务规划**: 使用write_todos工具将复杂的投资分析任务分解为步骤
2. **知识管理**: 使用文件系统工具读取/memories/knowledge/下的领域知识
3. **委派专家**: 使用task工具委派给专门的SubAgent:
   - macro-analyst: 宏观经济分析
   - policy-analyst: 政策分析
   - stock-analyst: 股票分析
4. **综合决策**: 整合各SubAgent的分析结果,给出全面的投资建议

## 工作流程

当用户提出投资诉求时:

1. 使用write_todos分解任务(如: 宏观分析→政策分析→标的选择→风险评估)
2. 使用read_file读取相关领域的知识背景
3. 使用task工具并行委派给各SubAgent
4. 等待SubAgent返回结果
5. 综合分析,给出:
   - 当前市场环境评估
   - 资产配置建议(比例)
   - 具体投资标的推荐
   - 风险提示
6. 使用write_file将分析报告保存到/memories/analysis_history/

## 输出格式

最终报告应包含:
- 📊 市场环境分析
- 💰 资产配置方案
- 🎯 投资标的推荐
- ⚠️ 风险提示
""",

    # SubAgents配置
    subagents=[
        macro_subagent_config,
        policy_subagent_config,
        stock_subagent_config,
    ],

    # 主Agent自己的工具(可以添加额外工具)
    tools=[],

    # 配置持久化存储
    store=store,

    # 文件系统后端(支持长期记忆)
    backend=lambda config: CompositeBackend(
        default=StateBackend(config),
        routes={
            "/memories/": StoreBackend(config)  # /memories/路径持久化
        }
    ),
)

# ============ 4. 使用示例 ============

# 用户提出投资诉求
user_query = """
我有100万资金,想进行投资配置。
我的风险偏好是稳健型,希望获得年化8-12%的收益。
请帮我分析当前市场环境,给出具体的配置方案。
"""

config = {"configurable": {"thread_id": "user_001"}}

# 调用主Agent
for chunk in investment_advisor.stream(
    {"messages": [{"role": "user", "content": user_query}]},
    config=config,
    stream_mode="values"
):
    print(chunk["messages"][-1].content)

# Agent会自动:
# 1. write_todos: 分解任务
# 2. read_file: 读取知识
# 3. task("macro-analyst", "分析当前宏观经济形势")
# 4. task("policy-analyst", "分析政策影响")
# 5. task("stock-analyst", "推荐具体标的")
# 6. 综合分析,生成报告
# 7. write_file: 保存报告到/memories/
```

```plaintext
AnalystChain/
├── src/langchain_app/
│   ├── agents/
│   │   ├── investment_advisor.py    # 主DeepAgent
│   │   └── subagents/
│   │       ├── macro_analyst.py     # 宏观分析SubAgent配置
│   │       ├── policy_analyst.py    # 政策分析SubAgent配置
│   │       └── stock_analyst.py     # 股票分析SubAgent配置
│   ├── tools/
│   │   ├── akshare_tools.py         # AKShare工具封装
│   │   └── knowledge_retriever.py   # 知识检索工具
│   ├── knowledge/
│   │   ├── processor.py             # 知识处理Pipeline
│   │   └── run_once.py              # 一次性处理脚本
│   └── memories/                    # DeepAgents持久化目录
│       ├── knowledge/               # 知识库(持久化)
│       │   ├── macro_economy/
│       │   ├── policy/
│       │   └── stock_analysis/
│       └── analysis_history/        # 历史分析记录
├── vector_db/                       # 向量数据库
│   ├── macro_economy/
│   ├── policy/
│   └── stock_analysis/
└── jupyter_notebook/
    └── macroeconomic_analysis/
        └── knowledge_base/          # 原始知识文件
```

```shellscript
   python src/langchain_app/knowledge/run_once.py
   # 处理所有知识文件,生成:
   # - /memories/knowledge/* (持久化JSON)
   # - vector_db/* (向量索引)
```

```plaintext
01第一节 中国经济的"三驾马车"[防断更微coc36666]_20250706193405.pptx
01第一节 中国经济的"三驾马车"[防断更微coc36666]_笔记.pdf
01第一节 中国经济的"三驾马车"[防断更微coc36666].doc
02第二节 消费——快速入门读懂经济形势[防断更微coc36666]_20250707140225.pptx
02第二节 消费——快速入门读懂经济形势[防断更微coc36666]_笔记.pdf
02第二节 消费——快速入门读懂经济形势[防断更微coc36666].doc
```

非常好的问题!确实应该自动识别和分组。让我给你一个智能的知识文件自动识别方案:

```python
import os
import re
from pathlib import Path
from collections import defaultdict
from difflib import SequenceMatcher

class KnowledgeFileOrganizer:
    """智能识别和组织知识文件"""

    def __init__(self, knowledge_base_dir: str):
        """
        Args:
            knowledge_base_dir: 知识库根目录,如 "jupyter_notebook/macroeconomic_analysis/knowledge_base"
        """
        self.knowledge_base_dir = Path(knowledge_base_dir)
        self.supported_extensions = {'.pdf', '.doc', '.docx', '.pptx', '.ppt'}

    def clean_filename(self, filename: str) -> str:
        """
        清理文件名,去除噪音信息

        例如:
        "01第一节 中国经济的三驾马车[防断更微coc36666]_20250706193405.pptx"
        → "01第一节 中国经济的三驾马车"
        """
        # 去除扩展名
        name = Path(filename).stem

        # 去除常见噪音模式
        patterns_to_remove = [
            r'\[防断更微.*?\]',           # [防断更微coc36666]
            r'_\d{14}',                   # _20250706193405
            r'_笔记',                     # _笔记
            r'\s*\(.*?\)\s*',             # (任何括号内容)
        ]

        for pattern in patterns_to_remove:
            name = re.sub(pattern, '', name)

        # 清理多余空格
        name = re.sub(r'\s+', ' ', name).strip()

        return name

    def extract_sequence_number(self, filename: str) -> tuple:
        """
        提取文件名前的序号

        例如: "01第一节" → (1, "01")
        返回: (数字序号, 原始序号字符串)
        """
        match = re.match(r'^(\d+)', filename)
        if match:
            seq_str = match.group(1)
            return (int(seq_str), seq_str)
        return (float('inf'), '')  # 没有序号的排最后

    def calculate_similarity(self, str1: str, str2: str) -> float:
        """计算两个字符串的相似度(0-1)"""
        return SequenceMatcher(None, str1, str2).ratio()

    def group_files_by_similarity(self, files: list) -> dict:
        """
        根据文件名相似度分组

        Returns:
            {
                "group_key": {
                    "files": [文件路径列表],
                    "topic": "主题名称",
                    "sequence": 序号
                }
            }
        """
        if not files:
            return {}

        groups = defaultdict(lambda: {"files": [], "topic": "", "sequence": None})
        processed = set()

        for i, file1 in enumerate(files):
            if file1 in processed:
                continue

            # 清理文件名并提取序号
            clean_name1 = self.clean_filename(file1.name)
            seq1, seq_str1 = self.extract_sequence_number(clean_name1)

            # 创建组
            group_key = f"{seq_str1}_{clean_name1[:20]}"  # 用序号+前20字符作为组key
            groups[group_key]["files"].append(file1)
            groups[group_key]["topic"] = clean_name1
            groups[group_key]["sequence"] = seq1
            processed.add(file1)

            # 查找相似的文件加入同组
            for j, file2 in enumerate(files[i+1:], start=i+1):
                if file2 in processed:
                    continue

                clean_name2 = self.clean_filename(file2.name)
                seq2, seq_str2 = self.extract_sequence_number(clean_name2)

                # 相似度阈值: 0.7 (可调整)
                # 并且序号要相同
                similarity = self.calculate_similarity(clean_name1, clean_name2)
                if similarity >= 0.7 and seq1 == seq2:
                    groups[group_key]["files"].append(file2)
                    processed.add(file2)

        return dict(groups)

    def get_file_priority(self, filepath: Path) -> int:
        """
        定义文件处理优先级

        优先级: PDF笔记 > DOC > PPTX
        因为PDF笔记通常内容最完整
        """
        name_lower = filepath.name.lower()

        if '笔记' in name_lower and filepath.suffix == '.pdf':
            return 1  # 最高优先级
        elif filepath.suffix in ['.doc', '.docx']:
            return 2
        elif filepath.suffix in ['.pptx', '.ppt']:
            return 3
        elif filepath.suffix == '.pdf':
            return 2
        else:
            return 4

    def scan_and_organize(self, domain: str = None) -> dict:
        """
        扫描知识库目录,自动组织文件

        Args:
            domain: 领域名称(如"macro_economy"),为None则自动根据目录结构推断

        Returns:
            {
                "macro_economy": {
                    "01_中国经济的三驾马车": {
                        "files": [Path对象列表],
                        "topic": "主题",
                        "primary_file": Path对象,  # 优先处理的文件
                    },
                    ...
                },
                ...
            }
        """
        result = defaultdict(dict)

        # 如果指定了domain,只扫描该目录
        if domain:
            search_dirs = [self.knowledge_base_dir / domain]
        else:
            # 扫描所有子目录
            search_dirs = [d for d in self.knowledge_base_dir.iterdir() if d.is_dir()]
            # 如果没有子目录,就扫描根目录
            if not search_dirs:
                search_dirs = [self.knowledge_base_dir]

        for search_dir in search_dirs:
            # 确定领域名称
            if search_dir == self.knowledge_base_dir:
                # 根据父目录名推断
                domain_name = self.knowledge_base_dir.name
            else:
                domain_name = search_dir.name

            # 获取所有支持的文件
            all_files = [
                f for f in search_dir.iterdir()
                if f.is_file() and f.suffix in self.supported_extensions
            ]

            if not all_files:
                continue

            # 按相似度分组
            groups = self.group_files_by_similarity(all_files)

            # 为每组选择主文件
            for group_key, group_info in groups.items():
                files = group_info["files"]

                # 按优先级排序
                files_sorted = sorted(files, key=self.get_file_priority)
                primary_file = files_sorted[0]

                result[domain_name][group_key] = {
                    "files": files,
                    "topic": group_info["topic"],
                    "sequence": group_info["sequence"],
                    "primary_file": primary_file,  # 优先处理这个文件
                    "file_types": [f.suffix for f in files],
                }

        # 按序号排序
        for domain_name in result:
            result[domain_name] = dict(
                sorted(
                    result[domain_name].items(),
                    key=lambda x: x[1]["sequence"]
                )
            )

        return dict(result)

    def print_organization(self):
        """打印组织结构(用于检查)"""
        organized = self.scan_and_organize()

        print("📚 知识库组织结构:")
        print("=" * 80)

        for domain, groups in organized.items():
            print(f"\n🎯 领域: {domain}")
            print("-" * 80)

            for group_key, group_info in groups.items():
                print(f"\n  📖 知识块: {group_info['topic']}")
                print(f"     序号: {group_info['sequence']}")
                print(f"     主文件: {group_info['primary_file'].name}")
                print(f"     所有文件({len(group_info['files'])}个):")
                for file in group_info['files']:
                    print(f"       - {file.name}")

        print("\n" + "=" * 80)
        return organized


# ============ 使用示例 ============

if __name__ == "__main__":
    # 初始化组织器
    organizer = KnowledgeFileOrganizer(
        knowledge_base_dir="/Users/zhou/Project/AnalystChain/jupyter_notebook/macroeconomic_analysis/knowledge_base"
    )

    # 打印组织结构(检查是否正确分组)
    organized = organizer.print_organization()

    # 使用分组结果进行处理
    print("\n\n🔄 开始处理知识文件...")

    for domain, groups in organized.items():
        print(f"\n处理领域: {domain}")

        for group_key, group_info in groups.items():
            topic = group_info['topic']
            primary_file = group_info['primary_file']
            all_files = group_info['files']

            print(f"  处理知识块: {topic}")
            print(f"    主文件: {primary_file.name}")

            # TODO: 这里调用你的知识处理Pipeline
            # process_knowledge_block(
            #     domain=domain,
            #     topic=topic,
            #     primary_file=primary_file,
            #     supplementary_files=all_files
            # )
```

### 使用效果示例

运行后会输出:

```
📚 知识库组织结构:
================================================================================

🎯 领域: knowledge_base
--------------------------------------------------------------------------------

  📖 知识块: 01第一节 中国经济的三驾马车
     序号: 1
     主文件: 01第一节 中国经济的"三驾马车"[防断更微coc36666]_笔记.pdf
     所有文件(3个):
       - 01第一节 中国经济的"三驾马车"[防断更微coc36666]_笔记.pdf
       - 01第一节 中国经济的"三驾马车"[防断更微coc36666].doc
       - 01第一节 中国经济的"三驾马车"[防断更微coc36666]_20250706193405.pptx

  📖 知识块: 02第二节 消费
     序号: 2
     主文件: 02第二节 消费——快速入门读懂经济形势[防断更微coc36666]_笔记.pdf
     所有文件(3个):
       - 02第二节 消费——快速入门读懂经济形势[防断更微coc36666]_笔记.pdf
       - 02第二节 消费——快速入门读懂经济形势[防断更微coc36666].doc
       - 02第二节 消费——快速入门读懂经济形势[防断更微coc36666]_20250707140225.pptx

================================================================================
```

### 集成到知识处理Pipeline

```python
from deepagents import create_deep_agent

def process_knowledge_with_auto_detection():
    """自动检测并处理知识文件"""

    # 1. 自动扫描和分组
    organizer = KnowledgeFileOrganizer(
        knowledge_base_dir="/Users/zhou/Project/AnalystChain/jupyter_notebook/macroeconomic_analysis/knowledge_base"
    )

    organized = organizer.scan_and_organize()

    # 2. 逐个处理知识块
    for domain, groups in organized.items():
        print(f"\n📂 处理领域: {domain}")

        # 为该领域创建向量存储
        vector_store = Chroma(
            persist_directory=f"./vector_db/{domain}",
            embedding_function=embeddings
        )

        # 存储该领域的结构化知识
        domain_knowledge = {
            "domain": domain,
            "knowledge_blocks": []
        }

        for group_key, group_info in groups.items():
            print(f"  📖 处理: {group_info['topic']}")

            # 加载主文件(优先级最高的)
            primary_file = group_info['primary_file']
            docs = load_document(primary_file)

            # 如果主文件内容不够,可以加载其他文件作为补充
            if len(docs) < 3:  # 内容太少
                for file in group_info['files']:
                    if file != primary_file:
                        docs.extend(load_document(file))

            # 用LLM提取结构化知识
            structured = extract_structured_knowledge(
                docs=docs,
                topic=group_info['topic']
            )

            domain_knowledge["knowledge_blocks"].append({
                "topic": group_info['topic'],
                "sequence": group_info['sequence'],
                "structured": structured,
            })

            # 向量化存储
            chunks = text_splitter.split_documents(docs)
            vector_store.add_documents(
                chunks,
                metadata={
                    "domain": domain,
                    "topic": group_info['topic'],
                    "sequence": group_info['sequence'],
                }
            )

        # 持久化
        vector_store.persist()

        # 保存到DeepAgents的/memories/目录
        save_to_memories(
            path=f"/memories/knowledge/{domain}/structured.json",
            content=json.dumps(domain_knowledge, ensure_ascii=False, indent=2)
        )

        print(f"  ✅ {domain} 处理完成")

    print("\n\n🎉 所有知识处理完成!")


# 一次性运行
if __name__ == "__main__":
    process_knowledge_with_auto_detection()
```

### 关键特性

✅ **自动文件发现**: 扫描目录,无需手动列举文件
✅ **智能分组**: 基于文件名相似度和序号自动分组
✅ **噪音过滤**: 自动去除`[防断更微coc36666]`、时间戳等噪音
✅ **优先级处理**: 优先使用PDF笔记(内容最完整)
✅ **灵活扩展**: 新增文件自动识别,无需修改代码
✅ **多级目录支持**: 支持按子目录组织不同领域

### 调整建议

如果自动分组不准确,可以调整:

1. **相似度阈值**: 目前是0.7,可以调到0.6-0.8
2. **清理规则**: 在`clean_filename`中添加更多噪音模式
3. **优先级**: 在`get_file_priority`中调整文件类型优先级

这样就完全自动化了!新增知识文件时,只需放到目录下,再次运行即可自动处理。

```python
import os
import re
from pathlib import Path
from collections import defaultdict
from difflib import SequenceMatcher

class KnowledgeFileOrganizer:
    """智能识别和组织知识文件"""

    def __init__(self, knowledge_base_dir: str):
        """
        Args:
            knowledge_base_dir: 知识库根目录,如 "jupyter_notebook/macroeconomic_analysis/knowledge_base"
        """
        self.knowledge_base_dir = Path(knowledge_base_dir)
        self.supported_extensions = {'.pdf', '.doc', '.docx', '.pptx', '.ppt'}

    def clean_filename(self, filename: str) -> str:
        """
        清理文件名,去除噪音信息

        例如:
        "01第一节 中国经济的三驾马车[防断更微coc36666]_20250706193405.pptx"
        → "01第一节 中国经济的三驾马车"
        """
        # 去除扩展名
        name = Path(filename).stem

        # 去除常见噪音模式
        patterns_to_remove = [
            r'\[防断更微.*?\]',           # [防断更微coc36666]
            r'_\d{14}',                   # _20250706193405
            r'_笔记',                     # _笔记
            r'\s*\(.*?\)\s*',             # (任何括号内容)
        ]

        for pattern in patterns_to_remove:
            name = re.sub(pattern, '', name)

        # 清理多余空格
        name = re.sub(r'\s+', ' ', name).strip()

        return name

    def extract_sequence_number(self, filename: str) -> tuple:
        """
        提取文件名前的序号

        例如: "01第一节" → (1, "01")
        返回: (数字序号, 原始序号字符串)
        """
        match = re.match(r'^(\d+)', filename)
        if match:
            seq_str = match.group(1)
            return (int(seq_str), seq_str)
        return (float('inf'), '')  # 没有序号的排最后

    def calculate_similarity(self, str1: str, str2: str) -> float:
        """计算两个字符串的相似度(0-1)"""
        return SequenceMatcher(None, str1, str2).ratio()

    def group_files_by_similarity(self, files: list) -> dict:
        """
        根据文件名相似度分组

        Returns:
            {
                "group_key": {
                    "files": [文件路径列表],
                    "topic": "主题名称",
                    "sequence": 序号
                }
            }
        """
        if not files:
            return {}

        groups = defaultdict(lambda: {"files": [], "topic": "", "sequence": None})
        processed = set()

        for i, file1 in enumerate(files):
            if file1 in processed:
                continue

            # 清理文件名并提取序号
            clean_name1 = self.clean_filename(file1.name)
            seq1, seq_str1 = self.extract_sequence_number(clean_name1)

            # 创建组
            group_key = f"{seq_str1}_{clean_name1[:20]}"  # 用序号+前20字符作为组key
            groups[group_key]["files"].append(file1)
            groups[group_key]["topic"] = clean_name1
            groups[group_key]["sequence"] = seq1
            processed.add(file1)

            # 查找相似的文件加入同组
            for j, file2 in enumerate(files[i+1:], start=i+1):
                if file2 in processed:
                    continue

                clean_name2 = self.clean_filename(file2.name)
                seq2, seq_str2 = self.extract_sequence_number(clean_name2)

                # 相似度阈值: 0.7 (可调整)
                # 并且序号要相同
                similarity = self.calculate_similarity(clean_name1, clean_name2)
                if similarity >= 0.7 and seq1 == seq2:
                    groups[group_key]["files"].append(file2)
                    processed.add(file2)

        return dict(groups)

    def get_file_priority(self, filepath: Path) -> int:
        """
        定义文件处理优先级

        优先级: PDF笔记 > DOC > PPTX
        因为PDF笔记通常内容最完整
        """
        name_lower = filepath.name.lower()

        if '笔记' in name_lower and filepath.suffix == '.pdf':
            return 1  # 最高优先级
        elif filepath.suffix in ['.doc', '.docx']:
            return 2
        elif filepath.suffix in ['.pptx', '.ppt']:
            return 3
        elif filepath.suffix == '.pdf':
            return 2
        else:
            return 4

    def scan_and_organize(self, domain: str = None) -> dict:
        """
        扫描知识库目录,自动组织文件

        Args:
            domain: 领域名称(如"macro_economy"),为None则自动根据目录结构推断

        Returns:
            {
                "macro_economy": {
                    "01_中国经济的三驾马车": {
                        "files": [Path对象列表],
                        "topic": "主题",
                        "primary_file": Path对象,  # 优先处理的文件
                    },
                    ...
                },
                ...
            }
        """
        result = defaultdict(dict)

        # 如果指定了domain,只扫描该目录
        if domain:
            search_dirs = [self.knowledge_base_dir / domain]
        else:
            # 扫描所有子目录
            search_dirs = [d for d in self.knowledge_base_dir.iterdir() if d.is_dir()]
            # 如果没有子目录,就扫描根目录
            if not search_dirs:
                search_dirs = [self.knowledge_base_dir]

        for search_dir in search_dirs:
            # 确定领域名称
            if search_dir == self.knowledge_base_dir:
                # 根据父目录名推断
                domain_name = self.knowledge_base_dir.name
            else:
                domain_name = search_dir.name

            # 获取所有支持的文件
            all_files = [
                f for f in search_dir.iterdir()
                if f.is_file() and f.suffix in self.supported_extensions
            ]

            if not all_files:
                continue

            # 按相似度分组
            groups = self.group_files_by_similarity(all_files)

            # 为每组选择主文件
            for group_key, group_info in groups.items():
                files = group_info["files"]

                # 按优先级排序
                files_sorted = sorted(files, key=self.get_file_priority)
                primary_file = files_sorted[0]

                result[domain_name][group_key] = {
                    "files": files,
                    "topic": group_info["topic"],
                    "sequence": group_info["sequence"],
                    "primary_file": primary_file,  # 优先处理这个文件
                    "file_types": [f.suffix for f in files],
                }

        # 按序号排序
        for domain_name in result:
            result[domain_name] = dict(
                sorted(
                    result[domain_name].items(),
                    key=lambda x: x[1]["sequence"]
                )
            )

        return dict(result)

    def print_organization(self):
        """打印组织结构(用于检查)"""
        organized = self.scan_and_organize()

        print("📚 知识库组织结构:")
        print("=" * 80)

        for domain, groups in organized.items():
            print(f"\n🎯 领域: {domain}")
            print("-" * 80)

            for group_key, group_info in groups.items():
                print(f"\n  📖 知识块: {group_info['topic']}")
                print(f"     序号: {group_info['sequence']}")
                print(f"     主文件: {group_info['primary_file'].name}")
                print(f"     所有文件({len(group_info['files'])}个):")
                for file in group_info['files']:
                    print(f"       - {file.name}")

        print("\n" + "=" * 80)
        return organized


# ============ 使用示例 ============

if __name__ == "__main__":
    # 初始化组织器
    organizer = KnowledgeFileOrganizer(
        knowledge_base_dir="/Users/zhou/Project/AnalystChain/jupyter_notebook/macroeconomic_analysis/knowledge_base"
    )

    # 打印组织结构(检查是否正确分组)
    organized = organizer.print_organization()

    # 使用分组结果进行处理
    print("\n\n🔄 开始处理知识文件...")

    for domain, groups in organized.items():
        print(f"\n处理领域: {domain}")

        for group_key, group_info in groups.items():
            topic = group_info['topic']
            primary_file = group_info['primary_file']
            all_files = group_info['files']

            print(f"  处理知识块: {topic}")
            print(f"    主文件: {primary_file.name}")

            # TODO: 这里调用你的知识处理Pipeline
            # process_knowledge_block(
            #     domain=domain,
            #     topic=topic,
            #     primary_file=primary_file,
            #     supplementary_files=all_files
            # )
```

```plaintext
📚 知识库组织结构:
================================================================================

🎯 领域: knowledge_base
--------------------------------------------------------------------------------

  📖 知识块: 01第一节 中国经济的三驾马车
     序号: 1
     主文件: 01第一节 中国经济的"三驾马车"[防断更微coc36666]_笔记.pdf
     所有文件(3个):
       - 01第一节 中国经济的"三驾马车"[防断更微coc36666]_笔记.pdf
       - 01第一节 中国经济的"三驾马车"[防断更微coc36666].doc
       - 01第一节 中国经济的"三驾马车"[防断更微coc36666]_20250706193405.pptx

  📖 知识块: 02第二节 消费
     序号: 2
     主文件: 02第二节 消费——快速入门读懂经济形势[防断更微coc36666]_笔记.pdf
     所有文件(3个):
       - 02第二节 消费——快速入门读懂经济形势[防断更微coc36666]_笔记.pdf
       - 02第二节 消费——快速入门读懂经济形势[防断更微coc36666].doc
       - 02第二节 消费——快速入门读懂经济形势[防断更微coc36666]_20250707140225.pptx

================================================================================
```

```python
from deepagents import create_deep_agent

def process_knowledge_with_auto_detection():
    """自动检测并处理知识文件"""

    # 1. 自动扫描和分组
    organizer = KnowledgeFileOrganizer(
        knowledge_base_dir="/Users/zhou/Project/AnalystChain/jupyter_notebook/macroeconomic_analysis/knowledge_base"
    )

    organized = organizer.scan_and_organize()

    # 2. 逐个处理知识块
    for domain, groups in organized.items():
        print(f"\n📂 处理领域: {domain}")

        # 为该领域创建向量存储
        vector_store = Chroma(
            persist_directory=f"./vector_db/{domain}",
            embedding_function=embeddings
        )

        # 存储该领域的结构化知识
        domain_knowledge = {
            "domain": domain,
            "knowledge_blocks": []
        }

        for group_key, group_info in groups.items():
            print(f"  📖 处理: {group_info['topic']}")

            # 加载主文件(优先级最高的)
            primary_file = group_info['primary_file']
            docs = load_document(primary_file)

            # 如果主文件内容不够,可以加载其他文件作为补充
            if len(docs) < 3:  # 内容太少
                for file in group_info['files']:
                    if file != primary_file:
                        docs.extend(load_document(file))

            # 用LLM提取结构化知识
            structured = extract_structured_knowledge(
                docs=docs,
                topic=group_info['topic']
            )

            domain_knowledge["knowledge_blocks"].append({
                "topic": group_info['topic'],
                "sequence": group_info['sequence'],
                "structured": structured,
            })

            # 向量化存储
            chunks = text_splitter.split_documents(docs)
            vector_store.add_documents(
                chunks,
                metadata={
                    "domain": domain,
                    "topic": group_info['topic'],
                    "sequence": group_info['sequence'],
                }
            )

        # 持久化
        vector_store.persist()

        # 保存到DeepAgents的/memories/目录
        save_to_memories(
            path=f"/memories/knowledge/{domain}/structured.json",
            content=json.dumps(domain_knowledge, ensure_ascii=False, indent=2)
        )

        print(f"  ✅ {domain} 处理完成")

    print("\n\n🎉 所有知识处理完成!")


# 一次性运行
if __name__ == "__main__":
    process_knowledge_with_auto_detection()
```

```python
import os
import re
from pathlib import Path
from collections import defaultdict
from difflib import SequenceMatcher

class KnowledgeFileOrganizer:
    """智能识别和组织知识文件"""

    def __init__(self, knowledge_base_dir: str):
        """
        Args:
            knowledge_base_dir: 知识库根目录,如 "jupyter_notebook/macroeconomic_analysis/knowledge_base"
        """
        self.knowledge_base_dir = Path(knowledge_base_dir)
        self.supported_extensions = {'.pdf', '.doc', '.docx', '.pptx', '.ppt'}

    def clean_filename(self, filename: str) -> str:
        """
        清理文件名,去除噪音信息

        例如:
        "01第一节 中国经济的三驾马车[防断更微coc36666]_20250706193405.pptx"
        → "01第一节 中国经济的三驾马车"
        """
        # 去除扩展名
        name = Path(filename).stem

        # 去除常见噪音模式
        patterns_to_remove = [
            r'\[防断更微.*?\]',           # [防断更微coc36666]
            r'_\d{14}',                   # _20250706193405
            r'_笔记',                     # _笔记
            r'\s*\(.*?\)\s*',             # (任何括号内容)
        ]

        for pattern in patterns_to_remove:
            name = re.sub(pattern, '', name)

        # 清理多余空格
        name = re.sub(r'\s+', ' ', name).strip()

        return name

    def extract_sequence_number(self, filename: str) -> tuple:
        """
        提取文件名前的序号

        例如: "01第一节" → (1, "01")
        返回: (数字序号, 原始序号字符串)
        """
        match = re.match(r'^(\d+)', filename)
        if match:
            seq_str = match.group(1)
            return (int(seq_str), seq_str)
        return (float('inf'), '')  # 没有序号的排最后

    def calculate_similarity(self, str1: str, str2: str) -> float:
        """计算两个字符串的相似度(0-1)"""
        return SequenceMatcher(None, str1, str2).ratio()

    def group_files_by_similarity(self, files: list) -> dict:
        """
        根据文件名相似度分组

        Returns:
            {
                "group_key": {
                    "files": [文件路径列表],
                    "topic": "主题名称",
                    "sequence": 序号
                }
            }
        """
        if not files:
            return {}

        groups = defaultdict(lambda: {"files": [], "topic": "", "sequence": None})
        processed = set()

        for i, file1 in enumerate(files):
            if file1 in processed:
                continue

            # 清理文件名并提取序号
            clean_name1 = self.clean_filename(file1.name)
            seq1, seq_str1 = self.extract_sequence_number(clean_name1)

            # 创建组
            group_key = f"{seq_str1}_{clean_name1[:20]}"  # 用序号+前20字符作为组key
            groups[group_key]["files"].append(file1)
            groups[group_key]["topic"] = clean_name1
            groups[group_key]["sequence"] = seq1
            processed.add(file1)

            # 查找相似的文件加入同组
            for j, file2 in enumerate(files[i+1:], start=i+1):
                if file2 in processed:
                    continue

                clean_name2 = self.clean_filename(file2.name)
                seq2, seq_str2 = self.extract_sequence_number(clean_name2)

                # 相似度阈值: 0.7 (可调整)
                # 并且序号要相同
                similarity = self.calculate_similarity(clean_name1, clean_name2)
                if similarity >= 0.7 and seq1 == seq2:
                    groups[group_key]["files"].append(file2)
                    processed.add(file2)

        return dict(groups)

    def get_file_priority(self, filepath: Path) -> int:
        """
        定义文件处理优先级

        优先级: PDF笔记 > DOC > PPTX
        因为PDF笔记通常内容最完整
        """
        name_lower = filepath.name.lower()

        if '笔记' in name_lower and filepath.suffix == '.pdf':
            return 1  # 最高优先级
        elif filepath.suffix in ['.doc', '.docx']:
            return 2
        elif filepath.suffix in ['.pptx', '.ppt']:
            return 3
        elif filepath.suffix == '.pdf':
            return 2
        else:
            return 4

    def scan_and_organize(self, domain: str = None) -> dict:
        """
        扫描知识库目录,自动组织文件

        Args:
            domain: 领域名称(如"macro_economy"),为None则自动根据目录结构推断

        Returns:
            {
                "macro_economy": {
                    "01_中国经济的三驾马车": {
                        "files": [Path对象列表],
                        "topic": "主题",
                        "primary_file": Path对象,  # 优先处理的文件
                    },
                    ...
                },
                ...
            }
        """
        result = defaultdict(dict)

        # 如果指定了domain,只扫描该目录
        if domain:
            search_dirs = [self.knowledge_base_dir / domain]
        else:
            # 扫描所有子目录
            search_dirs = [d for d in self.knowledge_base_dir.iterdir() if d.is_dir()]
            # 如果没有子目录,就扫描根目录
            if not search_dirs:
                search_dirs = [self.knowledge_base_dir]

        for search_dir in search_dirs:
            # 确定领域名称
            if search_dir == self.knowledge_base_dir:
                # 根据父目录名推断
                domain_name = self.knowledge_base_dir.name
            else:
                domain_name = search_dir.name

            # 获取所有支持的文件
            all_files = [
                f for f in search_dir.iterdir()
                if f.is_file() and f.suffix in self.supported_extensions
            ]

            if not all_files:
                continue

            # 按相似度分组
            groups = self.group_files_by_similarity(all_files)

            # 为每组选择主文件
            for group_key, group_info in groups.items():
                files = group_info["files"]

                # 按优先级排序
                files_sorted = sorted(files, key=self.get_file_priority)
                primary_file = files_sorted[0]

                result[domain_name][group_key] = {
                    "files": files,
                    "topic": group_info["topic"],
                    "sequence": group_info["sequence"],
                    "primary_file": primary_file,  # 优先处理这个文件
                    "file_types": [f.suffix for f in files],
                }

        # 按序号排序
        for domain_name in result:
            result[domain_name] = dict(
                sorted(
                    result[domain_name].items(),
                    key=lambda x: x[1]["sequence"]
                )
            )

        return dict(result)

    def print_organization(self):
        """打印组织结构(用于检查)"""
        organized = self.scan_and_organize()

        print("📚 知识库组织结构:")
        print("=" * 80)

        for domain, groups in organized.items():
            print(f"\n🎯 领域: {domain}")
            print("-" * 80)

            for group_key, group_info in groups.items():
                print(f"\n  📖 知识块: {group_info['topic']}")
                print(f"     序号: {group_info['sequence']}")
                print(f"     主文件: {group_info['primary_file'].name}")
                print(f"     所有文件({len(group_info['files'])}个):")
                for file in group_info['files']:
                    print(f"       - {file.name}")

        print("\n" + "=" * 80)
        return organized


# ============ 使用示例 ============

if __name__ == "__main__":
    # 初始化组织器
    organizer = KnowledgeFileOrganizer(
        knowledge_base_dir="/Users/zhou/Project/AnalystChain/jupyter_notebook/macroeconomic_analysis/knowledge_base"
    )

    # 打印组织结构(检查是否正确分组)
    organized = organizer.print_organization()

    # 使用分组结果进行处理
    print("\n\n🔄 开始处理知识文件...")

    for domain, groups in organized.items():
        print(f"\n处理领域: {domain}")

        for group_key, group_info in groups.items():
            topic = group_info['topic']
            primary_file = group_info['primary_file']
            all_files = group_info['files']

            print(f"  处理知识块: {topic}")
            print(f"    主文件: {primary_file.name}")

            # TODO: 这里调用你的知识处理Pipeline
            # process_knowledge_block(
            #     domain=domain,
            #     topic=topic,
            #     primary_file=primary_file,
            #     supplementary_files=all_files
            # )
```

```plaintext
📚 知识库组织结构:
================================================================================

🎯 领域: knowledge_base
--------------------------------------------------------------------------------

  📖 知识块: 01第一节 中国经济的三驾马车
     序号: 1
     主文件: 01第一节 中国经济的"三驾马车"[防断更微coc36666]_笔记.pdf
     所有文件(3个):
       - 01第一节 中国经济的"三驾马车"[防断更微coc36666]_笔记.pdf
       - 01第一节 中国经济的"三驾马车"[防断更微coc36666].doc
       - 01第一节 中国经济的"三驾马车"[防断更微coc36666]_20250706193405.pptx

  📖 知识块: 02第二节 消费
     序号: 2
     主文件: 02第二节 消费——快速入门读懂经济形势[防断更微coc36666]_笔记.pdf
     所有文件(3个):
       - 02第二节 消费——快速入门读懂经济形势[防断更微coc36666]_笔记.pdf
       - 02第二节 消费——快速入门读懂经济形势[防断更微coc36666].doc
       - 02第二节 消费——快速入门读懂经济形势[防断更微coc36666]_20250707140225.pptx

================================================================================
```

```python
from deepagents import create_deep_agent

def process_knowledge_with_auto_detection():
    """自动检测并处理知识文件"""

    # 1. 自动扫描和分组
    organizer = KnowledgeFileOrganizer(
        knowledge_base_dir="/Users/zhou/Project/AnalystChain/jupyter_notebook/macroeconomic_analysis/knowledge_base"
    )

    organized = organizer.scan_and_organize()

    # 2. 逐个处理知识块
    for domain, groups in organized.items():
        print(f"\n📂 处理领域: {domain}")

        # 为该领域创建向量存储
        vector_store = Chroma(
            persist_directory=f"./vector_db/{domain}",
            embedding_function=embeddings
        )

        # 存储该领域的结构化知识
        domain_knowledge = {
            "domain": domain,
            "knowledge_blocks": []
        }

        for group_key, group_info in groups.items():
            print(f"  📖 处理: {group_info['topic']}")

            # 加载主文件(优先级最高的)
            primary_file = group_info['primary_file']
            docs = load_document(primary_file)

            # 如果主文件内容不够,可以加载其他文件作为补充
            if len(docs) < 3:  # 内容太少
                for file in group_info['files']:
                    if file != primary_file:
                        docs.extend(load_document(file))

            # 用LLM提取结构化知识
            structured = extract_structured_knowledge(
                docs=docs,
                topic=group_info['topic']
            )

            domain_knowledge["knowledge_blocks"].append({
                "topic": group_info['topic'],
                "sequence": group_info['sequence'],
                "structured": structured,
            })

            # 向量化存储
            chunks = text_splitter.split_documents(docs)
            vector_store.add_documents(
                chunks,
                metadata={
                    "domain": domain,
                    "topic": group_info['topic'],
                    "sequence": group_info['sequence'],
                }
            )

        # 持久化
        vector_store.persist()

        # 保存到DeepAgents的/memories/目录
        save_to_memories(
            path=f"/memories/knowledge/{domain}/structured.json",
            content=json.dumps(domain_knowledge, ensure_ascii=False, indent=2)
        )

        print(f"  ✅ {domain} 处理完成")

    print("\n\n🎉 所有知识处理完成!")


# 一次性运行
if __name__ == "__main__":
    process_knowledge_with_auto_detection()
```

完全理解!让我给你一个**简洁清晰的技术方案框架**,用图示和要点帮你快速建立认知:

---

## 🎯 智能投资分析系统 - 技术框架全景图

### 一、整体架构(3层)

```
┌─────────────────────────────────────────────────────────────┐
│                     【用户层】                                │
│                 用户投资诉求 & 交互                           │
└─────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│                   【智能决策层】                              │
│                                                               │
│   ┌─────────────────────────────────────────────────┐      │
│   │         主Agent (投资顾问 DeepAgent)              │      │
│   │                                                   │      │
│   │  能力1: 任务规划 (TodoList)                       │      │
│   │  能力2: 知识管理 (Filesystem)                     │      │
│   │  能力3: 委派专家 (SubAgents)                      │      │
│   │  能力4: 综合决策                                  │      │
│   └─────────────────────────────────────────────────┘      │
│                             ↓                                 │
│        ┌──────────┬──────────────┬──────────┐              │
│        ↓          ↓              ↓          │              │
│   ┌────────┐ ┌────────┐ ┌────────┐         │              │
│   │宏观分析│ │政策分析│ │股票分析│  (可并行)│              │
│   │SubAgent│ │SubAgent│ │SubAgent│         │              │
│   └────────┘ └────────┘ └────────┘         │              │
│        ↓          ↓              ↓          │              │
│        └──────────┴──────────────┘          │              │
│                   ↓                          │              │
│           整合分析 → 投资建议                 │              │
└─────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│                   【数据&知识层】                             │
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  知识库      │  │  向量数据库  │  │  实时数据    │     │
│  │ (持久化)     │  │  (RAG检索)   │  │  (AKShare)   │     │
│  │              │  │              │  │              │     │
│  │ /memories/   │  │ ChromaDB     │  │ GDP/CPI/PMI  │     │
│  │ knowledge/   │  │ Embeddings   │  │ 股票/政策    │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

---

### 二、核心流程(5步记忆法)

```
用户诉求 → 知识准备 → 任务分解 → 专家分析 → 综合决策
   ↓          ↓          ↓          ↓          ↓
 提需求    读知识库    写TODO    调SubAgent   出报告
```

**详细流程:**

```python
# 第1步: 用户提出投资诉求
用户: "我有100万,风险偏好稳健,求投资方案"

# 第2步: 主Agent读取知识背景
主Agent执行: read_file("/memories/knowledge/macro_economy/structured.json")
→ 了解宏观经济分析理论和方法

# 第3步: 主Agent分解任务
主Agent执行: write_todos([
    "分析当前宏观经济形势",
    "评估政策影响",
    "筛选投资标的",
    "制定配置方案"
])

# 第4步: 委派给SubAgents(并行执行)
主Agent执行:
  task("macro-analyst", "分析当前经济周期和投资时钟")
  task("policy-analyst", "分析财政货币政策影响")
  task("stock-analyst", "推荐具体投资标的")

# 各SubAgent独立工作:
宏观SubAgent:
  → search_knowledge("投资时钟理论")
  → get_macro_data("GDP", "CPI", "PMI")
  → 分析经济周期 → 返回报告

政策SubAgent:
  → search_knowledge("政策分析框架")
  → get_policy_news()
  → 分析政策 → 返回报告

股票SubAgent:
  → search_knowledge("价值投资方法")
  → get_stock_data("000001")
  → 筛选标的 → 返回报告

# 第5步: 主Agent综合决策
主Agent收到3个报告 → 整合分析 → 生成最终方案:
  - 资产配置: 股票40% + 债券40% + 现金20%
  - 推荐标的: XXX, YYY, ZZZ
  - 风险提示: ...
```

---

### 三、技术核心(4个关键概念)

#### 1️⃣ **DeepAgents = 超级Agent框架**

```
create_deep_agent = 普通Agent + 3大超能力

超能力1: TodoList(任务规划)
       → 自动分解复杂任务为步骤

超能力2: Filesystem(上下文管理)
       → /memories/路径持久化知识
       → 防止Context Window爆掉

超能力3: SubAgent(专家委派)
       → task工具调用专门的SubAgent
       → 自动并行执行,上下文隔离
```

#### 2️⃣ **知识处理 = 一次生成,永久使用**

```
原始知识文件 (PDF/DOC/PPTX)
       ↓
  【一次性处理】
       ↓
生成2种格式:
  1. 结构化JSON → 存 /memories/knowledge/
     (持久化,DeepAgent直接读)

  2. 向量Embeddings → 存 ChromaDB
     (RAG检索,语义搜索)

后续使用:
  → 直接读取,不再重复处理
  → 新增知识时,只处理新文件
```

#### 3️⃣ **SubAgent = 专家型独立Agent**

```
每个SubAgent是独立的专家:
  - 有自己的system_prompt(专业指令)
  - 有自己的tools(专属工具)
  - 有自己的知识库(领域知识)
  - 独立上下文(不污染主Agent)

优势:
  ✅ 专业性强 (术业有专攻)
  ✅ 并行执行 (加速分析)
  ✅ 易于扩展 (新增维度只需加SubAgent)
  ✅ 上下文清晰 (各干各的活)
```

#### 4️⃣ **工具体系 = 3类工具**

```
工具类型1: 知识检索工具
  → search_knowledge(query)
  → 从向量库检索相关理论

工具类型2: 数据获取工具
  → get_macro_data(indicator)
  → get_stock_data(code)
  → 调用AKShare获取实时数据

工具类型3: 文件系统工具(内置)
  → read_file, write_file
  → 读写/memories/下的知识
```

---

### 四、实施路径(3个阶段)

```
阶段1: 知识准备(一次性)
  → 写知识文件扫描器
  → 写知识处理Pipeline
  → 执行: 生成JSON + 向量库
  时间: 1-2天

阶段2: SubAgents开发
  → 定义3个SubAgent配置
  → 封装AKShare工具
  → 创建知识检索工具
  时间: 2-3天

阶段3: 主Agent集成
  → 创建主DeepAgent
  → 连接所有SubAgents
  → 测试完整流程
  时间: 1-2天

总计: 1周左右
```

---

### 五、目录结构(一图理解)

```
AnalystChain/
│
├── 【核心代码】 src/langchain_app/
│   ├── agents/
│   │   ├── investment_advisor.py    ← 主Agent入口
│   │   └── subagents/               ← 3个SubAgent配置
│   │       ├── macro_analyst.py
│   │       ├── policy_analyst.py
│   │       └── stock_analyst.py
│   │
│   ├── tools/
│   │   ├── akshare_tools.py         ← AKShare封装
│   │   └── knowledge_retriever.py   ← RAG检索
│   │
│   └── knowledge/
│       ├── organizer.py             ← 文件扫描器
│       └── processor.py             ← 知识处理
│
├── 【知识存储】 memories/              ← DeepAgents持久化
│   └── knowledge/
│       ├── macro_economy/
│       ├── policy/
│       └── stock_analysis/
│
├── 【向量库】 vector_db/
│   ├── macro_economy/
│   ├── policy/
│   └── stock_analysis/
│
└── 【原始资料】 jupyter_notebook/
    └── macroeconomic_analysis/
        └── knowledge_base/          ← 百度网盘的原始文件
```

---

### 六、记忆口诀(帮助快速回忆)

```
🎯 整体架构: 3层 (用户层 → 决策层 → 数据层)

🔄 核心流程: 5步 (诉求 → 知识 → 分解 → 分析 → 决策)

💎 技术核心: 4个 (DeepAgent框架 + 知识处理 + SubAgent专家 + 3类工具)

📁 目录结构: 4区 (代码区 + 知识区 + 向量区 + 原始区)

🚀 实施路径: 3阶段 (知识准备 → SubAgents开发 → 主Agent集成)
```

---

### 七、关键优势(为什么这样设计)

| 设计点 | 优势 | 体现在 |
|--------|------|--------|
| **DeepAgents** | 开箱即用,少写80%代码 | 内置Planning/Filesystem/SubAgent |
| **知识一次处理** | 节省计算,加速响应 | 持久化到/memories/ |
| **SubAgent隔离** | 上下文清晰,并行快 | 每个SubAgent独立工作 |
| **模块化设计** | 易扩展,易维护 | 新增分析维度只需加SubAgent |

---

## 🎓 总结: 技术方案的本质

```
本质: 用DeepAgents搭建一个"投资顾问智能体"

组成:
  - 1个主Agent (总指挥)
  - N个SubAgent (各领域专家)
  - 知识库 (理论指导)
  - 工具集 (数据获取)

工作方式:
  主Agent像一个投资顾问团队的首席
  遇到问题→分解任务→委派专家→汇总建议

技术亮点:
  ✅ 完全自动化(从需求到建议)
  ✅ 持久化知识(一次处理,永久使用)
  ✅ 并行分析(多维度同时进行)
  ✅ 易于扩展(插件式添加新能力)
```

---

这个框架清晰吗?你可以用"3层-5步-4核心"来快速记忆整个体系!

```plaintext
┌─────────────────────────────────────────────────────────────┐
│                     【用户层】                                │
│                 用户投资诉求 & 交互                           │
└─────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│                   【智能决策层】                              │
│                                                               │
│   ┌─────────────────────────────────────────────────┐      │
│   │         主Agent (投资顾问 DeepAgent)              │      │
│   │                                                   │      │
│   │  能力1: 任务规划 (TodoList)                       │      │
│   │  能力2: 知识管理 (Filesystem)                     │      │
│   │  能力3: 委派专家 (SubAgents)                      │      │
│   │  能力4: 综合决策                                  │      │
│   └─────────────────────────────────────────────────┘      │
│                             ↓                                 │
│        ┌──────────┬──────────────┬──────────┐              │
│        ↓          ↓              ↓          │              │
│   ┌────────┐ ┌────────┐ ┌────────┐         │              │
│   │宏观分析│ │政策分析│ │股票分析│  (可并行)│              │
│   │SubAgent│ │SubAgent│ │SubAgent│         │              │
│   └────────┘ └────────┘ └────────┘         │              │
│        ↓          ↓              ↓          │              │
│        └──────────┴──────────────┘          │              │
│                   ↓                          │              │
│           整合分析 → 投资建议                 │              │
└─────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│                   【数据&知识层】                             │
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  知识库      │  │  向量数据库  │  │  实时数据    │     │
│  │ (持久化)     │  │  (RAG检索)   │  │  (AKShare)   │     │
│  │              │  │              │  │              │     │
│  │ /memories/   │  │ ChromaDB     │  │ GDP/CPI/PMI  │     │
│  │ knowledge/   │  │ Embeddings   │  │ 股票/政策    │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

```plaintext
用户诉求 → 知识准备 → 任务分解 → 专家分析 → 综合决策
   ↓          ↓          ↓          ↓          ↓
 提需求    读知识库    写TODO    调SubAgent   出报告
```

```python
# 第1步: 用户提出投资诉求
用户: "我有100万,风险偏好稳健,求投资方案"

# 第2步: 主Agent读取知识背景
主Agent执行: read_file("/memories/knowledge/macro_economy/structured.json")
→ 了解宏观经济分析理论和方法

# 第3步: 主Agent分解任务
主Agent执行: write_todos([
    "分析当前宏观经济形势",
    "评估政策影响",
    "筛选投资标的",
    "制定配置方案"
])

# 第4步: 委派给SubAgents(并行执行)
主Agent执行:
  task("macro-analyst", "分析当前经济周期和投资时钟")
  task("policy-analyst", "分析财政货币政策影响")
  task("stock-analyst", "推荐具体投资标的")

# 各SubAgent独立工作:
宏观SubAgent:
  → search_knowledge("投资时钟理论")
  → get_macro_data("GDP", "CPI", "PMI")
  → 分析经济周期 → 返回报告

政策SubAgent:
  → search_knowledge("政策分析框架")
  → get_policy_news()
  → 分析政策 → 返回报告

股票SubAgent:
  → search_knowledge("价值投资方法")
  → get_stock_data("000001")
  → 筛选标的 → 返回报告

# 第5步: 主Agent综合决策
主Agent收到3个报告 → 整合分析 → 生成最终方案:
  - 资产配置: 股票40% + 债券40% + 现金20%
  - 推荐标的: XXX, YYY, ZZZ
  - 风险提示: ...
```

```plaintext
create_deep_agent = 普通Agent + 3大超能力

超能力1: TodoList(任务规划)
       → 自动分解复杂任务为步骤

超能力2: Filesystem(上下文管理)
       → /memories/路径持久化知识
       → 防止Context Window爆掉

超能力3: SubAgent(专家委派)
       → task工具调用专门的SubAgent
       → 自动并行执行,上下文隔离
```

```plaintext
原始知识文件 (PDF/DOC/PPTX)
       ↓
  【一次性处理】
       ↓
生成2种格式:
  1. 结构化JSON → 存 /memories/knowledge/
     (持久化,DeepAgent直接读)

  2. 向量Embeddings → 存 ChromaDB
     (RAG检索,语义搜索)

后续使用:
  → 直接读取,不再重复处理
  → 新增知识时,只处理新文件
```

```plaintext
每个SubAgent是独立的专家:
  - 有自己的system_prompt(专业指令)
  - 有自己的tools(专属工具)
  - 有自己的知识库(领域知识)
  - 独立上下文(不污染主Agent)

优势:
  ✅ 专业性强 (术业有专攻)
  ✅ 并行执行 (加速分析)
  ✅ 易于扩展 (新增维度只需加SubAgent)
  ✅ 上下文清晰 (各干各的活)
```

```plaintext
工具类型1: 知识检索工具
  → search_knowledge(query)
  → 从向量库检索相关理论

工具类型2: 数据获取工具
  → get_macro_data(indicator)
  → get_stock_data(code)
  → 调用AKShare获取实时数据

工具类型3: 文件系统工具(内置)
  → read_file, write_file
  → 读写/memories/下的知识
```

```plaintext
阶段1: 知识准备(一次性)
  → 写知识文件扫描器
  → 写知识处理Pipeline
  → 执行: 生成JSON + 向量库
  时间: 1-2天

阶段2: SubAgents开发
  → 定义3个SubAgent配置
  → 封装AKShare工具
  → 创建知识检索工具
  时间: 2-3天

阶段3: 主Agent集成
  → 创建主DeepAgent
  → 连接所有SubAgents
  → 测试完整流程
  时间: 1-2天

总计: 1周左右
```

```plaintext
AnalystChain/
│
├── 【核心代码】 src/langchain_app/
│   ├── agents/
│   │   ├── investment_advisor.py    ← 主Agent入口
│   │   └── subagents/               ← 3个SubAgent配置
│   │       ├── macro_analyst.py
│   │       ├── policy_analyst.py
│   │       └── stock_analyst.py
│   │
│   ├── tools/
│   │   ├── akshare_tools.py         ← AKShare封装
│   │   └── knowledge_retriever.py   ← RAG检索
│   │
│   └── knowledge/
│       ├── organizer.py             ← 文件扫描器
│       └── processor.py             ← 知识处理
│
├── 【知识存储】 memories/              ← DeepAgents持久化
│   └── knowledge/
│       ├── macro_economy/
│       ├── policy/
│       └── stock_analysis/
│
├── 【向量库】 vector_db/
│   ├── macro_economy/
│   ├── policy/
│   └── stock_analysis/
│
└── 【原始资料】 jupyter_notebook/
    └── macroeconomic_analysis/
        └── knowledge_base/          ← 百度网盘的原始文件
```

```plaintext
🎯 整体架构: 3层 (用户层 → 决策层 → 数据层)

🔄 核心流程: 5步 (诉求 → 知识 → 分解 → 分析 → 决策)

💎 技术核心: 4个 (DeepAgent框架 + 知识处理 + SubAgent专家 + 3类工具)

📁 目录结构: 4区 (代码区 + 知识区 + 向量区 + 原始区)

🚀 实施路径: 3阶段 (知识准备 → SubAgents开发 → 主Agent集成)
```

```plaintext
本质: 用DeepAgents搭建一个"投资顾问智能体"

组成:
  - 1个主Agent (总指挥)
  - N个SubAgent (各领域专家)
  - 知识库 (理论指导)
  - 工具集 (数据获取)

工作方式:
  主Agent像一个投资顾问团队的首席
  遇到问题→分解任务→委派专家→汇总建议

技术亮点:
  ✅ 完全自动化(从需求到建议)
  ✅ 持久化知识(一次处理,永久使用)
  ✅ 并行分析(多维度同时进行)
  ✅ 易于扩展(插件式添加新能力)
```

```plaintext
┌─────────────────────────────────────────────────────────────┐
│                     【用户层】                                │
│                 用户投资诉求 & 交互                           │
└─────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│                   【智能决策层】                              │
│                                                               │
│   ┌─────────────────────────────────────────────────┐      │
│   │         主Agent (投资顾问 DeepAgent)              │      │
│   │                                                   │      │
│   │  能力1: 任务规划 (TodoList)                       │      │
│   │  能力2: 知识管理 (Filesystem)                     │      │
│   │  能力3: 委派专家 (SubAgents)                      │      │
│   │  能力4: 综合决策                                  │      │
│   └─────────────────────────────────────────────────┘      │
│                             ↓                                 │
│        ┌──────────┬──────────────┬──────────┐              │
│        ↓          ↓              ↓          │              │
│   ┌────────┐ ┌────────┐ ┌────────┐         │              │
│   │宏观分析│ │政策分析│ │股票分析│  (可并行)│              │
│   │SubAgent│ │SubAgent│ │SubAgent│         │              │
│   └────────┘ └────────┘ └────────┘         │              │
│        ↓          ↓              ↓          │              │
│        └──────────┴──────────────┘          │              │
│                   ↓                          │              │
│           整合分析 → 投资建议                 │              │
└─────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│                   【数据&知识层】                             │
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  知识库      │  │  向量数据库  │  │  实时数据    │     │
│  │ (持久化)     │  │  (RAG检索)   │  │  (AKShare)   │     │
│  │              │  │              │  │              │     │
│  │ /memories/   │  │ ChromaDB     │  │ GDP/CPI/PMI  │     │
│  │ knowledge/   │  │ Embeddings   │  │ 股票/政策    │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

```plaintext
用户诉求 → 知识准备 → 任务分解 → 专家分析 → 综合决策
   ↓          ↓          ↓          ↓          ↓
 提需求    读知识库    写TODO    调SubAgent   出报告
```

```python
# 第1步: 用户提出投资诉求
用户: "我有100万,风险偏好稳健,求投资方案"

# 第2步: 主Agent读取知识背景
主Agent执行: read_file("/memories/knowledge/macro_economy/structured.json")
→ 了解宏观经济分析理论和方法

# 第3步: 主Agent分解任务
主Agent执行: write_todos([
    "分析当前宏观经济形势",
    "评估政策影响",
    "筛选投资标的",
    "制定配置方案"
])

# 第4步: 委派给SubAgents(并行执行)
主Agent执行:
  task("macro-analyst", "分析当前经济周期和投资时钟")
  task("policy-analyst", "分析财政货币政策影响")
  task("stock-analyst", "推荐具体投资标的")

# 各SubAgent独立工作:
宏观SubAgent:
  → search_knowledge("投资时钟理论")
  → get_macro_data("GDP", "CPI", "PMI")
  → 分析经济周期 → 返回报告

政策SubAgent:
  → search_knowledge("政策分析框架")
  → get_policy_news()
  → 分析政策 → 返回报告

股票SubAgent:
  → search_knowledge("价值投资方法")
  → get_stock_data("000001")
  → 筛选标的 → 返回报告

# 第5步: 主Agent综合决策
主Agent收到3个报告 → 整合分析 → 生成最终方案:
  - 资产配置: 股票40% + 债券40% + 现金20%
  - 推荐标的: XXX, YYY, ZZZ
  - 风险提示: ...
```

```plaintext
create_deep_agent = 普通Agent + 3大超能力

超能力1: TodoList(任务规划)
       → 自动分解复杂任务为步骤

超能力2: Filesystem(上下文管理)
       → /memories/路径持久化知识
       → 防止Context Window爆掉

超能力3: SubAgent(专家委派)
       → task工具调用专门的SubAgent
       → 自动并行执行,上下文隔离
```

```plaintext
原始知识文件 (PDF/DOC/PPTX)
       ↓
  【一次性处理】
       ↓
生成2种格式:
  1. 结构化JSON → 存 /memories/knowledge/
     (持久化,DeepAgent直接读)

  2. 向量Embeddings → 存 ChromaDB
     (RAG检索,语义搜索)

后续使用:
  → 直接读取,不再重复处理
  → 新增知识时,只处理新文件
```

```plaintext
每个SubAgent是独立的专家:
  - 有自己的system_prompt(专业指令)
  - 有自己的tools(专属工具)
  - 有自己的知识库(领域知识)
  - 独立上下文(不污染主Agent)

优势:
  ✅ 专业性强 (术业有专攻)
  ✅ 并行执行 (加速分析)
  ✅ 易于扩展 (新增维度只需加SubAgent)
  ✅ 上下文清晰 (各干各的活)
```

```plaintext
工具类型1: 知识检索工具
  → search_knowledge(query)
  → 从向量库检索相关理论

工具类型2: 数据获取工具
  → get_macro_data(indicator)
  → get_stock_data(code)
  → 调用AKShare获取实时数据

工具类型3: 文件系统工具(内置)
  → read_file, write_file
  → 读写/memories/下的知识
```

```plaintext
阶段1: 知识准备(一次性)
  → 写知识文件扫描器
  → 写知识处理Pipeline
  → 执行: 生成JSON + 向量库
  时间: 1-2天

阶段2: SubAgents开发
  → 定义3个SubAgent配置
  → 封装AKShare工具
  → 创建知识检索工具
  时间: 2-3天

阶段3: 主Agent集成
  → 创建主DeepAgent
  → 连接所有SubAgents
  → 测试完整流程
  时间: 1-2天

总计: 1周左右
```

```plaintext
AnalystChain/
│
├── 【核心代码】 src/langchain_app/
│   ├── agents/
│   │   ├── investment_advisor.py    ← 主Agent入口
│   │   └── subagents/               ← 3个SubAgent配置
│   │       ├── macro_analyst.py
│   │       ├── policy_analyst.py
│   │       └── stock_analyst.py
│   │
│   ├── tools/
│   │   ├── akshare_tools.py         ← AKShare封装
│   │   └── knowledge_retriever.py   ← RAG检索
│   │
│   └── knowledge/
│       ├── organizer.py             ← 文件扫描器
│       └── processor.py             ← 知识处理
│
├── 【知识存储】 memories/              ← DeepAgents持久化
│   └── knowledge/
│       ├── macro_economy/
│       ├── policy/
│       └── stock_analysis/
│
├── 【向量库】 vector_db/
│   ├── macro_economy/
│   ├── policy/
│   └── stock_analysis/
│
└── 【原始资料】 jupyter_notebook/
    └── macroeconomic_analysis/
        └── knowledge_base/          ← 百度网盘的原始文件
```

```plaintext
🎯 整体架构: 3层 (用户层 → 决策层 → 数据层)

🔄 核心流程: 5步 (诉求 → 知识 → 分解 → 分析 → 决策)

💎 技术核心: 4个 (DeepAgent框架 + 知识处理 + SubAgent专家 + 3类工具)

📁 目录结构: 4区 (代码区 + 知识区 + 向量区 + 原始区)

🚀 实施路径: 3阶段 (知识准备 → SubAgents开发 → 主Agent集成)
```

```plaintext
本质: 用DeepAgents搭建一个"投资顾问智能体"

组成:
  - 1个主Agent (总指挥)
  - N个SubAgent (各领域专家)
  - 知识库 (理论指导)
  - 工具集 (数据获取)

工作方式:
  主Agent像一个投资顾问团队的首席
  遇到问题→分解任务→委派专家→汇总建议

技术亮点:
  ✅ 完全自动化(从需求到建议)
  ✅ 持久化知识(一次处理,永久使用)
  ✅ 并行分析(多维度同时进行)
  ✅ 易于扩展(插件式添加新能力)
```

```plaintext
主Agent → 委派3个SubAgent → 主Agent收集结果 → 主Agent整合分析
```

```plaintext
主Agent → 委派3个SubAgent → 主Agent委派整合SubAgent → 整合SubAgent分析
```

非常好的问题!这涉及到架构设计的关键决策。让我给你分析两种方案:

---

## 🤔 方案对比:谁来整合SubAgent的结果?

### 方案A: 主Agent直接整合

```
用户诉求
    ↓
主Agent(协调者)
    ├─ 委派 → 宏观SubAgent → 返回结果A
    ├─ 委派 → 政策SubAgent → 返回结果B
    └─ 委派 → 股票SubAgent → 返回结果C
         ↓
    主Agent收集ABC
         ↓
    【主Agent自己整合分析】
         ↓
    最终投资建议
```

### 方案B: 专门的整合SubAgent ✅ **推荐**

```
用户诉求
    ↓
主Agent(总指挥)
    ├─ 委派 → 宏观SubAgent → 返回结果A
    ├─ 委派 → 政策SubAgent → 返回结果B
    └─ 委派 → 股票SubAgent → 返回结果C
         ↓
    主Agent收集ABC
         ↓
    委派 → 【整合SubAgent(专家)】
         ↓
    整合SubAgent分析:
      - 交叉验证各维度结论
      - 识别矛盾点并调和
      - 综合评估风险收益
      - 生成最终建议
         ↓
    最终投资建议
```

---

## 📊 两种方案对比

| 维度 | 方案A (主Agent整合) | 方案B (整合SubAgent) ✅ |
|------|-------------------|----------------------|
| **复杂度** | 简单直接 | 稍复杂(多一个SubAgent) |
| **职责分离** | ⚠️ 主Agent既协调又分析 | ✅ 主Agent专注协调,整合交给专家 |
| **整合质量** | ⚠️ 依赖主Agent通用能力 | ✅ 专门的整合逻辑和提示词 |
| **可扩展性** | ⚠️ 整合逻辑耦合在主Agent | ✅ 整合逻辑独立,易优化 |
| **上下文管理** | ⚠️ 主Agent上下文膨胀 | ✅ 整合SubAgent独立上下文 |
| **并行执行** | 前3个SubAgent并行 | 前3个SubAgent并行,整合串行 |

---

## ✅ 推荐方案B的理由

### 1️⃣ **职责分离原则**

```
主Agent的角色: 总指挥官
  - 理解用户需求
  - 分解任务
  - 委派专家
  - 监控进度
  ❌ 不应该: 自己做深度分析

整合SubAgent的角色: 战略顾问
  - 交叉验证多维度结论
  - 识别矛盾并调和
  - 综合风险收益评估
  - 生成结构化建议
```

### 2️⃣ **整合分析本身很复杂**

```
整合不是简单的"拼接报告",而是:

复杂任务1: 交叉验证
  - 宏观说"经济复苏" + 股票说"熊市" → 矛盾!需要深入分析

复杂任务2: 权重分配
  - 3个维度的建议如何加权?
  - 宏观70% + 政策20% + 个股10%?

复杂任务3: 风险综合评估
  - 单一维度风险可控,但组合风险可能放大

复杂任务4: 生成结构化方案
  - 资产配置比例
  - 具体标的推荐
  - 风险提示
  - 操作建议

→ 这些需要专门的提示词和逻辑,不应塞给主Agent
```

### 3️⃣ **更好的Prompt Engineering**

```python
# 主Agent的提示词(简洁)
system_prompt = """
你是投资顾问团队的首席协调官。

职责:
1. 理解用户需求
2. 分解为子任务
3. 委派给专门的SubAgent
4. 收集结果后委派给整合专家
5. 将最终建议反馈给用户

你不需要自己做深度分析。
"""

# 整合SubAgent的提示词(专业且详细)
integration_prompt = """
你是资深的投资策略整合专家。

你会收到3份分析报告:
1. 宏观经济分析报告
2. 政策分析报告
3. 股票分析报告

你的任务:

第一步: 交叉验证
  - 检查3个报告的结论是否一致
  - 识别矛盾点(如宏观看多但个股看空)
  - 分析矛盾的原因

第二步: 综合评估
  - 基于投资时钟理论,判断当前周期
  - 结合政策导向,评估各类资产吸引力
  - 整合个股推荐,筛选符合宏观和政策的标的

第三步: 风险评估
  - 单一维度风险
  - 组合风险(相关性)
  - 黑天鹅风险

第四步: 生成方案
  输出结构化的投资建议:
  {
    "market_environment": "市场环境判断",
    "asset_allocation": {
      "stock": 40,
      "bond": 40,
      "cash": 20
    },
    "recommended_stocks": [...],
    "risk_warnings": [...],
    "action_plan": "具体操作建议"
  }

重要原则:
  - 如果3个报告有矛盾,必须指出并给出你的判断
  - 风险提示必须明确
  - 配置比例必须与用户风险偏好匹配
"""
```

### 4️⃣ **更灵活的扩展**

```
假设未来要新增分析维度:

方案A:
  → 主Agent的整合逻辑越来越复杂
  → 提示词越来越长
  → 难以维护

方案B:
  → 只需修改整合SubAgent的提示词
  → 主Agent保持简洁不变
  → 易于维护和优化
```

---

## 🏗️ 推荐架构(带整合SubAgent)

```python
from deepagents import create_deep_agent

# ============ 定义4个SubAgents ============

# 1. 宏观分析SubAgent
macro_subagent = {
    "name": "macro-analyst",
    "description": "分析宏观经济形势和周期",
    "system_prompt": "你是宏观经济专家...",
    "tools": [get_macro_data, search_macro_knowledge],
}

# 2. 政策分析SubAgent
policy_subagent = {
    "name": "policy-analyst",
    "description": "分析财政货币政策影响",
    "system_prompt": "你是政策分析专家...",
    "tools": [get_policy_news, search_policy_knowledge],
}

# 3. 股票分析SubAgent
stock_subagent = {
    "name": "stock-analyst",
    "description": "进行股票分析和标的筛选",
    "system_prompt": "你是股票分析专家...",
    "tools": [get_stock_data, search_stock_knowledge],
}

# 4. 整合分析SubAgent ⭐ 关键
integration_subagent = {
    "name": "integration-strategist",
    "description": "整合多维度分析,生成综合投资建议",
    "system_prompt": """
你是资深投资策略整合专家。

输入: 3份分析报告(宏观、政策、股票)
任务: 交叉验证 → 综合评估 → 风险分析 → 生成方案

重点关注:
  - 各维度结论的一致性
  - 矛盾点的识别和调和
  - 投资时钟与资产配置的匹配
  - 用户风险偏好的适配
""",
    "tools": [],  # 不需要额外工具,只需分析能力
}

# ============ 创建主Agent ============

main_agent = create_deep_agent(
    model="deepseek-chat",

    system_prompt="""
你是智能投资顾问团队的首席协调官。

工作流程:
1. 理解用户投资诉求(金额、风险偏好、期望收益)
2. 使用write_todos分解任务
3. 并行委派给3个分析师:
   - task("macro-analyst", "分析当前宏观经济形势")
   - task("policy-analyst", "分析政策对投资的影响")
   - task("stock-analyst", "推荐具体投资标的")
4. 收集3份报告后,委派给整合专家:
   - task("integration-strategist", "整合分析,生成投资方案")
5. 将最终方案呈现给用户

你的角色是协调者,不需要自己做深度分析。
信任你的专家团队。
""",

    subagents=[
        macro_subagent,
        policy_subagent,
        stock_subagent,
        integration_subagent,  # ⭐ 加入整合SubAgent
    ],

    tools=[],
)
```

---

## 📝 实际执行示例

```python
# 用户提问
user_query = "我有100万,稳健型,求投资方案"

# 主Agent的执行过程
step1: 理解需求
  → "用户:100万,稳健型,年化8-12%"

step2: 分解任务(write_todos)
  → [1. 宏观分析, 2. 政策分析, 3. 个股分析, 4. 方案整合]

step3: 并行委派
  → task("macro-analyst", "当前经济周期?投资时钟位置?")
  → task("policy-analyst", "财政货币政策取向?")
  → task("stock-analyst", "稳健型适合的标的?")

step4: 收集结果
  → 宏观报告: "经济复苏初期,投资时钟→债券+股票"
  → 政策报告: "稳健货币政策,利好蓝筹股"
  → 股票报告: "推荐银行、公用事业"

step5: 委派整合专家
  → task("integration-strategist", """
      请整合以下3份报告,生成投资方案:

      宏观: {宏观报告}
      政策: {政策报告}
      股票: {股票报告}

      用户: 100万,稳健型
     """)

step6: 获得最终方案
  → 整合SubAgent返回:
    {
      "配置": "股票40%+债券40%+现金20%",
      "推荐": ["工商银行", "长江电力", "..."],
      "风险": "注意经济复苏不及预期风险"
    }

step7: 呈现给用户
  → 主Agent格式化输出给用户
```

---

## 🎯 最终建议

### ✅ 采用方案B (4个SubAgent):

```
1. macro-analyst       (宏观分析)
2. policy-analyst      (政策分析)
3. stock-analyst       (股票分析)
4. integration-strategist  (整合分析) ⭐
```

### 理由总结:

1. **职责清晰**: 主Agent负责协调,整合SubAgent负责深度分析
2. **质量更高**: 专门的整合提示词,交叉验证和风险评估更专业
3. **易于维护**: 整合逻辑独立,修改不影响主Agent
4. **符合设计理念**: "专业的事交给专业的Agent"

### 更新记忆口诀:

```
🎯 SubAgent数量: 3+1
   (3个分析师 + 1个整合专家)

🔄 核心流程: 6步
   (诉求 → 分解 → 3个并行分析 → 整合 → 决策)
```

这样设计更专业、更清晰!

```plaintext
用户诉求
    ↓
主Agent(协调者)
    ├─ 委派 → 宏观SubAgent → 返回结果A
    ├─ 委派 → 政策SubAgent → 返回结果B
    └─ 委派 → 股票SubAgent → 返回结果C
         ↓
    主Agent收集ABC
         ↓
    【主Agent自己整合分析】
         ↓
    最终投资建议
```

```plaintext
用户诉求
    ↓
主Agent(总指挥)
    ├─ 委派 → 宏观SubAgent → 返回结果A
    ├─ 委派 → 政策SubAgent → 返回结果B
    └─ 委派 → 股票SubAgent → 返回结果C
         ↓
    主Agent收集ABC
         ↓
    委派 → 【整合SubAgent(专家)】
         ↓
    整合SubAgent分析:
      - 交叉验证各维度结论
      - 识别矛盾点并调和
      - 综合评估风险收益
      - 生成最终建议
         ↓
    最终投资建议
```

```plaintext
主Agent的角色: 总指挥官
  - 理解用户需求
  - 分解任务
  - 委派专家
  - 监控进度
  ❌ 不应该: 自己做深度分析

整合SubAgent的角色: 战略顾问
  - 交叉验证多维度结论
  - 识别矛盾并调和
  - 综合风险收益评估
  - 生成结构化建议
```

```plaintext
整合不是简单的"拼接报告",而是:

复杂任务1: 交叉验证
  - 宏观说"经济复苏" + 股票说"熊市" → 矛盾!需要深入分析

复杂任务2: 权重分配
  - 3个维度的建议如何加权?
  - 宏观70% + 政策20% + 个股10%?

复杂任务3: 风险综合评估
  - 单一维度风险可控,但组合风险可能放大

复杂任务4: 生成结构化方案
  - 资产配置比例
  - 具体标的推荐
  - 风险提示
  - 操作建议

→ 这些需要专门的提示词和逻辑,不应塞给主Agent
```

```python
# 主Agent的提示词(简洁)
system_prompt = """
你是投资顾问团队的首席协调官。

职责:
1. 理解用户需求
2. 分解为子任务
3. 委派给专门的SubAgent
4. 收集结果后委派给整合专家
5. 将最终建议反馈给用户

你不需要自己做深度分析。
"""

# 整合SubAgent的提示词(专业且详细)
integration_prompt = """
你是资深的投资策略整合专家。

你会收到3份分析报告:
1. 宏观经济分析报告
2. 政策分析报告
3. 股票分析报告

你的任务:

第一步: 交叉验证
  - 检查3个报告的结论是否一致
  - 识别矛盾点(如宏观看多但个股看空)
  - 分析矛盾的原因

第二步: 综合评估
  - 基于投资时钟理论,判断当前周期
  - 结合政策导向,评估各类资产吸引力
  - 整合个股推荐,筛选符合宏观和政策的标的

第三步: 风险评估
  - 单一维度风险
  - 组合风险(相关性)
  - 黑天鹅风险

第四步: 生成方案
  输出结构化的投资建议:
  {
    "market_environment": "市场环境判断",
    "asset_allocation": {
      "stock": 40,
      "bond": 40,
      "cash": 20
    },
    "recommended_stocks": [...],
    "risk_warnings": [...],
    "action_plan": "具体操作建议"
  }

重要原则:
  - 如果3个报告有矛盾,必须指出并给出你的判断
  - 风险提示必须明确
  - 配置比例必须与用户风险偏好匹配
"""
```

```plaintext
假设未来要新增分析维度:

方案A:
  → 主Agent的整合逻辑越来越复杂
  → 提示词越来越长
  → 难以维护

方案B:
  → 只需修改整合SubAgent的提示词
  → 主Agent保持简洁不变
  → 易于维护和优化
```

```python
from deepagents import create_deep_agent

# ============ 定义4个SubAgents ============

# 1. 宏观分析SubAgent
macro_subagent = {
    "name": "macro-analyst",
    "description": "分析宏观经济形势和周期",
    "system_prompt": "你是宏观经济专家...",
    "tools": [get_macro_data, search_macro_knowledge],
}

# 2. 政策分析SubAgent
policy_subagent = {
    "name": "policy-analyst",
    "description": "分析财政货币政策影响",
    "system_prompt": "你是政策分析专家...",
    "tools": [get_policy_news, search_policy_knowledge],
}

# 3. 股票分析SubAgent
stock_subagent = {
    "name": "stock-analyst",
    "description": "进行股票分析和标的筛选",
    "system_prompt": "你是股票分析专家...",
    "tools": [get_stock_data, search_stock_knowledge],
}

# 4. 整合分析SubAgent ⭐ 关键
integration_subagent = {
    "name": "integration-strategist",
    "description": "整合多维度分析,生成综合投资建议",
    "system_prompt": """
你是资深投资策略整合专家。

输入: 3份分析报告(宏观、政策、股票)
任务: 交叉验证 → 综合评估 → 风险分析 → 生成方案

重点关注:
  - 各维度结论的一致性
  - 矛盾点的识别和调和
  - 投资时钟与资产配置的匹配
  - 用户风险偏好的适配
""",
    "tools": [],  # 不需要额外工具,只需分析能力
}

# ============ 创建主Agent ============

main_agent = create_deep_agent(
    model="deepseek-chat",

    system_prompt="""
你是智能投资顾问团队的首席协调官。

工作流程:
1. 理解用户投资诉求(金额、风险偏好、期望收益)
2. 使用write_todos分解任务
3. 并行委派给3个分析师:
   - task("macro-analyst", "分析当前宏观经济形势")
   - task("policy-analyst", "分析政策对投资的影响")
   - task("stock-analyst", "推荐具体投资标的")
4. 收集3份报告后,委派给整合专家:
   - task("integration-strategist", "整合分析,生成投资方案")
5. 将最终方案呈现给用户

你的角色是协调者,不需要自己做深度分析。
信任你的专家团队。
""",

    subagents=[
        macro_subagent,
        policy_subagent,
        stock_subagent,
        integration_subagent,  # ⭐ 加入整合SubAgent
    ],

    tools=[],
)
```

```python
# 用户提问
user_query = "我有100万,稳健型,求投资方案"

# 主Agent的执行过程
step1: 理解需求
  → "用户:100万,稳健型,年化8-12%"

step2: 分解任务(write_todos)
  → [1. 宏观分析, 2. 政策分析, 3. 个股分析, 4. 方案整合]

step3: 并行委派
  → task("macro-analyst", "当前经济周期?投资时钟位置?")
  → task("policy-analyst", "财政货币政策取向?")
  → task("stock-analyst", "稳健型适合的标的?")

step4: 收集结果
  → 宏观报告: "经济复苏初期,投资时钟→债券+股票"
  → 政策报告: "稳健货币政策,利好蓝筹股"
  → 股票报告: "推荐银行、公用事业"

step5: 委派整合专家
  → task("integration-strategist", """
      请整合以下3份报告,生成投资方案:

      宏观: {宏观报告}
      政策: {政策报告}
      股票: {股票报告}

      用户: 100万,稳健型
     """)

step6: 获得最终方案
  → 整合SubAgent返回:
    {
      "配置": "股票40%+债券40%+现金20%",
      "推荐": ["工商银行", "长江电力", "..."],
      "风险": "注意经济复苏不及预期风险"
    }

step7: 呈现给用户
  → 主Agent格式化输出给用户
```

```plaintext
1. macro-analyst       (宏观分析)
2. policy-analyst      (政策分析)
3. stock-analyst       (股票分析)
4. integration-strategist  (整合分析) ⭐
```

```plaintext
🎯 SubAgent数量: 3+1
   (3个分析师 + 1个整合专家)

🔄 核心流程: 6步
   (诉求 → 分解 → 3个并行分析 → 整合 → 决策)
```

```plaintext
用户诉求
    ↓
主Agent(协调者)
    ├─ 委派 → 宏观SubAgent → 返回结果A
    ├─ 委派 → 政策SubAgent → 返回结果B
    └─ 委派 → 股票SubAgent → 返回结果C
         ↓
    主Agent收集ABC
         ↓
    【主Agent自己整合分析】
         ↓
    最终投资建议
```

```plaintext
用户诉求
    ↓
主Agent(总指挥)
    ├─ 委派 → 宏观SubAgent → 返回结果A
    ├─ 委派 → 政策SubAgent → 返回结果B
    └─ 委派 → 股票SubAgent → 返回结果C
         ↓
    主Agent收集ABC
         ↓
    委派 → 【整合SubAgent(专家)】
         ↓
    整合SubAgent分析:
      - 交叉验证各维度结论
      - 识别矛盾点并调和
      - 综合评估风险收益
      - 生成最终建议
         ↓
    最终投资建议
```

```plaintext
主Agent的角色: 总指挥官
  - 理解用户需求
  - 分解任务
  - 委派专家
  - 监控进度
  ❌ 不应该: 自己做深度分析

整合SubAgent的角色: 战略顾问
  - 交叉验证多维度结论
  - 识别矛盾并调和
  - 综合风险收益评估
  - 生成结构化建议
```

```plaintext
整合不是简单的"拼接报告",而是:

复杂任务1: 交叉验证
  - 宏观说"经济复苏" + 股票说"熊市" → 矛盾!需要深入分析

复杂任务2: 权重分配
  - 3个维度的建议如何加权?
  - 宏观70% + 政策20% + 个股10%?

复杂任务3: 风险综合评估
  - 单一维度风险可控,但组合风险可能放大

复杂任务4: 生成结构化方案
  - 资产配置比例
  - 具体标的推荐
  - 风险提示
  - 操作建议

→ 这些需要专门的提示词和逻辑,不应塞给主Agent
```

```python
# 主Agent的提示词(简洁)
system_prompt = """
你是投资顾问团队的首席协调官。

职责:
1. 理解用户需求
2. 分解为子任务
3. 委派给专门的SubAgent
4. 收集结果后委派给整合专家
5. 将最终建议反馈给用户

你不需要自己做深度分析。
"""

# 整合SubAgent的提示词(专业且详细)
integration_prompt = """
你是资深的投资策略整合专家。

你会收到3份分析报告:
1. 宏观经济分析报告
2. 政策分析报告
3. 股票分析报告

你的任务:

第一步: 交叉验证
  - 检查3个报告的结论是否一致
  - 识别矛盾点(如宏观看多但个股看空)
  - 分析矛盾的原因

第二步: 综合评估
  - 基于投资时钟理论,判断当前周期
  - 结合政策导向,评估各类资产吸引力
  - 整合个股推荐,筛选符合宏观和政策的标的

第三步: 风险评估
  - 单一维度风险
  - 组合风险(相关性)
  - 黑天鹅风险

第四步: 生成方案
  输出结构化的投资建议:
  {
    "market_environment": "市场环境判断",
    "asset_allocation": {
      "stock": 40,
      "bond": 40,
      "cash": 20
    },
    "recommended_stocks": [...],
    "risk_warnings": [...],
    "action_plan": "具体操作建议"
  }

重要原则:
  - 如果3个报告有矛盾,必须指出并给出你的判断
  - 风险提示必须明确
  - 配置比例必须与用户风险偏好匹配
"""
```

```plaintext
假设未来要新增分析维度:

方案A:
  → 主Agent的整合逻辑越来越复杂
  → 提示词越来越长
  → 难以维护

方案B:
  → 只需修改整合SubAgent的提示词
  → 主Agent保持简洁不变
  → 易于维护和优化
```

```python
from deepagents import create_deep_agent

# ============ 定义4个SubAgents ============

# 1. 宏观分析SubAgent
macro_subagent = {
    "name": "macro-analyst",
    "description": "分析宏观经济形势和周期",
    "system_prompt": "你是宏观经济专家...",
    "tools": [get_macro_data, search_macro_knowledge],
}

# 2. 政策分析SubAgent
policy_subagent = {
    "name": "policy-analyst",
    "description": "分析财政货币政策影响",
    "system_prompt": "你是政策分析专家...",
    "tools": [get_policy_news, search_policy_knowledge],
}

# 3. 股票分析SubAgent
stock_subagent = {
    "name": "stock-analyst",
    "description": "进行股票分析和标的筛选",
    "system_prompt": "你是股票分析专家...",
    "tools": [get_stock_data, search_stock_knowledge],
}

# 4. 整合分析SubAgent ⭐ 关键
integration_subagent = {
    "name": "integration-strategist",
    "description": "整合多维度分析,生成综合投资建议",
    "system_prompt": """
你是资深投资策略整合专家。

输入: 3份分析报告(宏观、政策、股票)
任务: 交叉验证 → 综合评估 → 风险分析 → 生成方案

重点关注:
  - 各维度结论的一致性
  - 矛盾点的识别和调和
  - 投资时钟与资产配置的匹配
  - 用户风险偏好的适配
""",
    "tools": [],  # 不需要额外工具,只需分析能力
}

# ============ 创建主Agent ============

main_agent = create_deep_agent(
    model="deepseek-chat",

    system_prompt="""
你是智能投资顾问团队的首席协调官。

工作流程:
1. 理解用户投资诉求(金额、风险偏好、期望收益)
2. 使用write_todos分解任务
3. 并行委派给3个分析师:
   - task("macro-analyst", "分析当前宏观经济形势")
   - task("policy-analyst", "分析政策对投资的影响")
   - task("stock-analyst", "推荐具体投资标的")
4. 收集3份报告后,委派给整合专家:
   - task("integration-strategist", "整合分析,生成投资方案")
5. 将最终方案呈现给用户

你的角色是协调者,不需要自己做深度分析。
信任你的专家团队。
""",

    subagents=[
        macro_subagent,
        policy_subagent,
        stock_subagent,
        integration_subagent,  # ⭐ 加入整合SubAgent
    ],

    tools=[],
)
```

```python
# 用户提问
user_query = "我有100万,稳健型,求投资方案"

# 主Agent的执行过程
step1: 理解需求
  → "用户:100万,稳健型,年化8-12%"

step2: 分解任务(write_todos)
  → [1. 宏观分析, 2. 政策分析, 3. 个股分析, 4. 方案整合]

step3: 并行委派
  → task("macro-analyst", "当前经济周期?投资时钟位置?")
  → task("policy-analyst", "财政货币政策取向?")
  → task("stock-analyst", "稳健型适合的标的?")

step4: 收集结果
  → 宏观报告: "经济复苏初期,投资时钟→债券+股票"
  → 政策报告: "稳健货币政策,利好蓝筹股"
  → 股票报告: "推荐银行、公用事业"

step5: 委派整合专家
  → task("integration-strategist", """
      请整合以下3份报告,生成投资方案:

      宏观: {宏观报告}
      政策: {政策报告}
      股票: {股票报告}

      用户: 100万,稳健型
     """)

step6: 获得最终方案
  → 整合SubAgent返回:
    {
      "配置": "股票40%+债券40%+现金20%",
      "推荐": ["工商银行", "长江电力", "..."],
      "风险": "注意经济复苏不及预期风险"
    }

step7: 呈现给用户
  → 主Agent格式化输出给用户
```

```plaintext
1. macro-analyst       (宏观分析)
2. policy-analyst      (政策分析)
3. stock-analyst       (股票分析)
4. integration-strategist  (整合分析) ⭐
```

```plaintext
🎯 SubAgent数量: 3+1
   (3个分析师 + 1个整合专家)

🔄 核心流程: 6步
   (诉求 → 分解 → 3个并行分析 → 整合 → 决策)
```

好的!我给你一个**清晰的方案要点 + 代码大纲**,你来填充具体实现:

---

## 📋 知识处理方案要点

### 核心目标

```
原始知识文件(PDF/DOC/PPTX)
         ↓
【处理Pipeline】
         ↓
输出1: 结构化JSON → /memories/knowledge/{domain}/
输出2: 向量数据库 → vector_db/{domain}/
         ↓
供DeepAgent使用(一次处理,永久使用)
```

### 处理流程(7步)

```python
步骤1: 扫描文件夹,自动分组知识块
  → 输入: knowledge_base目录
  → 输出: { "domain": { "group": {...} } }

步骤2: 加载文档内容
  → 输入: 文件路径列表
  → 输出: List[Document]

步骤3: 文本提取和清洗
  → 输入: 原始Document
  → 输出: 清洗后的Document

步骤4: 结构化知识提取(LLM)
  → 输入: 清洗后的Document
  → 输出: 结构化JSON(概念/指标/方法/案例)

步骤5: 保存结构化知识到/memories/
  → 输入: 结构化JSON
  → 输出: 持久化文件

步骤6: 文档切块(Text Splitting)
  → 输入: Document
  → 输出: List[Document chunks]

步骤7: 向量化并存入ChromaDB
  → 输入: Document chunks
  → 输出: 持久化向量库
```

---

## 🏗️ 代码大纲(分模块)

### 模块1: 文件组织器 (`knowledge_organizer.py`)

```python
"""
知识文件自动扫描和分组
"""
from pathlib import Path
from typing import Dict, List
import re
from difflib import SequenceMatcher

class KnowledgeOrganizer:
    """知识文件组织器"""

    def __init__(self, knowledge_base_dir: str):
        self.knowledge_base_dir = Path(knowledge_base_dir)
        self.supported_extensions = {'.pdf', '.doc', '.docx', '.pptx'}

    def clean_filename(self, filename: str) -> str:
        """
        清理文件名,去除噪音

        TODO: 实现以下功能
        1. 去除[防断更微coc36666]等噪音
        2. 去除时间戳_20250706193405
        3. 去除_笔记后缀
        4. 清理多余空格

        输入: "01第一节 中国经济的三驾马车[防断更微coc36666]_笔记.pdf"
        输出: "01第一节 中国经济的三驾马车"
        """
        # TODO: 你来实现
        pass

    def extract_sequence_number(self, filename: str) -> tuple:
        """
        提取序号

        TODO:
        1. 用正则提取开头的数字
        2. 返回(数字, 原始字符串)

        输入: "01第一节"
        输出: (1, "01")
        """
        # TODO: 你来实现
        pass

    def calculate_similarity(self, str1: str, str2: str) -> float:
        """
        计算文件名相似度(0-1)

        提示: 使用difflib.SequenceMatcher
        """
        return SequenceMatcher(None, str1, str2).ratio()

    def group_files_by_similarity(
        self,
        files: List[Path],
        similarity_threshold: float = 0.7
    ) -> Dict:
        """
        根据相似度分组

        TODO: 实现分组逻辑
        1. 遍历文件,清理文件名
        2. 比较相似度,相似且序号相同→同组
        3. 返回分组结果

        返回格式:
        {
            "group_key": {
                "files": [Path列表],
                "topic": "主题名",
                "sequence": 序号,
            }
        }
        """
        # TODO: 你来实现
        pass

    def scan_and_organize(self) -> Dict:
        """
        扫描目录并组织

        TODO:
        1. 扫描knowledge_base_dir下的所有文件
        2. 调用group_files_by_similarity分组
        3. 为每组选择主文件(优先级: PDF笔记 > DOC > PPTX)

        返回格式:
        {
            "domain_name": {
                "group_key": {
                    "files": [...],
                    "topic": "...",
                    "primary_file": Path对象,
                }
            }
        }
        """
        # TODO: 你来实现
        pass


# ============ 使用示例 ============
if __name__ == "__main__":
    organizer = KnowledgeOrganizer(
        "/Users/zhou/Project/AnalystChain/jupyter_notebook/macroeconomic_analysis/knowledge_base"
    )

    result = organizer.scan_and_organize()

    # 打印结果(检查)
    for domain, groups in result.items():
        print(f"领域: {domain}")
        for group_key, info in groups.items():
            print(f"  - {info['topic']}: {len(info['files'])}个文件")
```

---

### 模块2: 文档加载器 (`document_loader.py`)

```python
"""
多格式文档加载和清洗
"""
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
)
from langchain_core.documents import Document
from pathlib import Path
from typing import List

class MultiFormatDocumentLoader:
    """多格式文档加载器"""

    def load_pdf(self, file_path: Path) -> List[Document]:
        """
        加载PDF文件

        TODO:
        1. 使用PyMuPDFLoader加载
        2. 处理可能的OCR需求
        3. 返回Document列表

        提示:
        loader = PyMuPDFLoader(str(file_path))
        docs = loader.load()
        """
        # TODO: 你来实现
        pass

    def load_word(self, file_path: Path) -> List[Document]:
        """
        加载Word文件

        TODO:
        1. 使用UnstructuredWordDocumentLoader
        2. 处理.doc和.docx

        提示:
        loader = UnstructuredWordDocumentLoader(str(file_path))
        docs = loader.load()
        """
        # TODO: 你来实现
        pass

    def load_ppt(self, file_path: Path) -> List[Document]:
        """
        加载PPT文件

        TODO:
        1. 使用UnstructuredPowerPointLoader
        2. 提取文本内容(图表可能提取不完整)
        """
        # TODO: 你来实现
        pass

    def load_document(self, file_path: Path) -> List[Document]:
        """
        根据文件类型自动选择加载器

        TODO:
        1. 根据file_path.suffix选择对应的loader
        2. 调用对应的load_xxx方法
        3. 处理加载失败的情况
        """
        suffix = file_path.suffix.lower()

        try:
            if suffix == '.pdf':
                return self.load_pdf(file_path)
            elif suffix in ['.doc', '.docx']:
                return self.load_word(file_path)
            elif suffix in ['.ppt', '.pptx']:
                return self.load_ppt(file_path)
            else:
                print(f"不支持的文件格式: {suffix}")
                return []
        except Exception as e:
            print(f"加载文件失败 {file_path}: {e}")
            return []

    def clean_document_text(self, doc: Document) -> Document:
        """
        清洗文档文本

        TODO: 实现文本清洗
        1. 去除多余空白字符
        2. 去除特殊字符(如\uf06c)
        3. 规范化换行
        4. 保留原始metadata

        返回: 清洗后的Document
        """
        # TODO: 你来实现
        pass

    def load_and_clean(self, file_path: Path) -> List[Document]:
        """加载并清洗文档"""
        docs = self.load_document(file_path)
        return [self.clean_document_text(doc) for doc in docs]


# ============ 使用示例 ============
if __name__ == "__main__":
    loader = MultiFormatDocumentLoader()

    docs = loader.load_and_clean(
        Path("/path/to/01第一节_笔记.pdf")
    )

    print(f"加载了 {len(docs)} 个文档")
    print(f"第一页内容: {docs[0].page_content[:200]}")
```

---

### 模块3: 知识提取器 (`knowledge_extractor.py`)

```python
"""
用LLM从文档中提取结构化知识
"""
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from typing import Dict, List
import json

class KnowledgeExtractor:
    """结构化知识提取器"""

    def __init__(self, model_name: str = "deepseek-chat"):
        """
        TODO:
        1. 初始化LLM模型
        2. 定义提取提示词模板
        """
        self.llm = init_chat_model(model_name, model_provider="deepseek")

        # TODO: 定义提取提示词
        self.extract_prompt = ChatPromptTemplate.from_messages([
            ("system", """
你是知识提取专家。从文档中提取结构化知识。

输出JSON格式:
{
  "topic": "主题名称",
  "key_concepts": [
    {"name": "概念名", "definition": "定义", "importance": "重要性"}
  ],
  "indicators": [
    {"name": "指标名", "calculation": "计算方法", "interpretation": "解读"}
  ],
  "analysis_methods": [
    {"name": "方法名", "steps": "步骤", "application": "应用场景"}
  ],
  "cases": [
    {"description": "案例描述", "insight": "启示"}
  ],
  "summary": "知识总结(200字以内)"
}
"""),
            ("human", "文档内容:\n\n{document_content}\n\n请提取结构化知识:")
        ])

    def extract_from_documents(
        self,
        docs: List[Document],
        topic: str
    ) -> Dict:
        """
        从文档列表中提取结构化知识

        TODO:
        1. 合并多个Document的内容
        2. 调用LLM提取
        3. 解析JSON结果
        4. 处理提取失败的情况

        Args:
            docs: 文档列表
            topic: 知识主题(如"中国经济的三驾马车")

        Returns:
            结构化知识dict
        """
        # TODO: 你来实现
        # 提示:
        # 1. 合并文档内容: content = "\n\n".join([doc.page_content for doc in docs])
        # 2. 调用LLM: chain = self.extract_prompt | self.llm
        # 3. 解析返回: json.loads(response.content)
        pass

    def validate_extracted_knowledge(self, knowledge: Dict) -> bool:
        """
        验证提取的知识是否完整

        TODO:
        检查必需字段是否存在:
        - topic
        - key_concepts
        - summary
        """
        # TODO: 你来实现
        pass


# ============ 使用示例 ============
if __name__ == "__main__":
    extractor = KnowledgeExtractor()

    # 假设已经加载了文档
    docs = [...]

    knowledge = extractor.extract_from_documents(
        docs=docs,
        topic="中国经济的三驾马车"
    )

    print(json.dumps(knowledge, ensure_ascii=False, indent=2))
```

---

### 模块4: 向量存储管理器 (`vector_store_manager.py`)

```python
"""
向量数据库管理
"""
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pathlib import Path
from typing import List

class VectorStoreManager:
    """向量存储管理器"""

    def __init__(
        self,
        embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
        persist_directory: str = "./vector_db"
    ):
        """
        TODO:
        1. 初始化Embeddings模型
        2. 初始化文本切分器
        3. 设置持久化目录
        """
        # TODO: 你来实现
        # 提示:
        # self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        # self.text_splitter = RecursiveCharacterTextSplitter(...)
        # self.persist_directory = Path(persist_directory)
        pass

    def create_vector_store(self, domain: str) -> Chroma:
        """
        为指定领域创建向量存储

        TODO:
        1. 创建领域专属的persist_directory
        2. 初始化Chroma

        Args:
            domain: 领域名(如"macro_economy")

        Returns:
            Chroma向量存储对象
        """
        # TODO: 你来实现
        pass

    def split_documents(
        self,
        docs: List[Document],
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> List[Document]:
        """
        文档切块

        TODO:
        1. 使用text_splitter切分
        2. 返回切块列表
        """
        # TODO: 你来实现
        pass

    def add_documents_to_store(
        self,
        domain: str,
        docs: List[Document],
        metadata: Dict = None
    ):
        """
        添加文档到向量库

        TODO:
        1. 切分文档
        2. 为每个chunk添加metadata
        3. 加入向量库
        4. 持久化

        Args:
            domain: 领域名
            docs: 文档列表
            metadata: 额外的metadata(如topic, sequence)
        """
        # TODO: 你来实现
        pass

    def search_similar(
        self,
        domain: str,
        query: str,
        k: int = 5
    ) -> List[Document]:
        """
        语义搜索

        TODO:
        1. 加载对应domain的向量库
        2. 执行相似度搜索
        3. 返回结果
        """
        # TODO: 你来实现
        pass


# ============ 使用示例 ============
if __name__ == "__main__":
    manager = VectorStoreManager()

    # 添加文档
    manager.add_documents_to_store(
        domain="macro_economy",
        docs=[...],
        metadata={"topic": "三驾马车", "sequence": 1}
    )

    # 搜索
    results = manager.search_similar(
        domain="macro_economy",
        query="什么是三驾马车?"
    )
```

---

### 模块5: 主处理Pipeline (`knowledge_processor.py`)

```python
"""
知识处理主流程
"""
from pathlib import Path
import json
from typing import Dict

from knowledge_organizer import KnowledgeOrganizer
from document_loader import MultiFormatDocumentLoader
from knowledge_extractor import KnowledgeExtractor
from vector_store_manager import VectorStoreManager

class KnowledgeProcessor:
    """知识处理主Pipeline"""

    def __init__(
        self,
        knowledge_base_dir: str,
        memories_dir: str = "./memories/knowledge",
        vector_db_dir: str = "./vector_db"
    ):
        """
        TODO: 初始化各个组件
        """
        self.organizer = KnowledgeOrganizer(knowledge_base_dir)
        self.loader = MultiFormatDocumentLoader()
        self.extractor = KnowledgeExtractor()
        self.vector_manager = VectorStoreManager(persist_directory=vector_db_dir)

        self.memories_dir = Path(memories_dir)
        self.memories_dir.mkdir(parents=True, exist_ok=True)

    def save_to_memories(self, domain: str, knowledge: Dict):
        """
        保存到/memories/目录(持久化)

        TODO:
        1. 创建domain子目录
        2. 保存structured.json
        3. 生成summary.md(可选)
        """
        domain_dir = self.memories_dir / domain
        domain_dir.mkdir(parents=True, exist_ok=True)

        # 保存JSON
        with open(domain_dir / "structured.json", "w", encoding="utf-8") as f:
            json.dump(knowledge, f, ensure_ascii=False, indent=2)

        print(f"✅ 已保存到 {domain_dir}/structured.json")

    def process_all(self):
        """
        处理所有知识文件的主流程

        TODO: 实现完整流程
        1. 调用organizer扫描分组
        2. 遍历每个domain和group
        3. 加载文档
        4. 提取结构化知识
        5. 保存到/memories/
        6. 向量化存储
        7. 打印处理进度
        """
        print("=" * 80)
        print("🚀 开始处理知识文件")
        print("=" * 80)

        # 步骤1: 扫描组织
        organized = self.organizer.scan_and_organize()

        # 步骤2: 遍历处理
        for domain, groups in organized.items():
            print(f"\n📂 处理领域: {domain}")

            domain_knowledge = {
                "domain": domain,
                "knowledge_blocks": []
            }

            for group_key, group_info in groups.items():
                topic = group_info['topic']
                primary_file = group_info['primary_file']

                print(f"  📖 处理: {topic}")
                print(f"     文件: {primary_file.name}")

                try:
                    # TODO: 步骤3 - 加载文档
                    docs = self.loader.load_and_clean(primary_file)

                    # TODO: 步骤4 - 提取结构化知识
                    structured = self.extractor.extract_from_documents(
                        docs=docs,
                        topic=topic
                    )

                    domain_knowledge["knowledge_blocks"].append({
                        "topic": topic,
                        "sequence": group_info['sequence'],
                        "structured": structured,
                    })

                    # TODO: 步骤6 - 向量化
                    self.vector_manager.add_documents_to_store(
                        domain=domain,
                        docs=docs,
                        metadata={
                            "topic": topic,
                            "sequence": group_info['sequence'],
                        }
                    )

                    print(f"     ✅ 完成")

                except Exception as e:
                    print(f"     ❌ 失败: {e}")

            # TODO: 步骤5 - 保存到/memories/
            self.save_to_memories(domain, domain_knowledge)

        print("\n" + "=" * 80)
        print("🎉 所有知识处理完成!")
        print("=" * 80)


# ============ 主入口 ============
if __name__ == "__main__":
    processor = KnowledgeProcessor(
        knowledge_base_dir="/Users/zhou/Project/AnalystChain/jupyter_notebook/macroeconomic_analysis/knowledge_base",
        memories_dir="/Users/zhou/Project/AnalystChain/memories/knowledge",
        vector_db_dir="/Users/zhou/Project/AnalystChain/vector_db"
    )

    # 一次性处理所有知识
    processor.process_all()
```

---

## 📝 实施清单

### 你需要填充的部分:

- [ ] `knowledge_organizer.py`
  - [ ] `clean_filename()` - 文件名清洗
  - [ ] `extract_sequence_number()` - 序号提取
  - [ ] `group_files_by_similarity()` - 相似度分组
  - [ ] `scan_and_organize()` - 主扫描逻辑

- [ ] `document_loader.py`
  - [ ] `load_pdf()` - PDF加载
  - [ ] `load_word()` - Word加载
  - [ ] `load_ppt()` - PPT加载
  - [ ] `clean_document_text()` - 文本清洗

- [ ] `knowledge_extractor.py`
  - [ ] `extract_from_documents()` - LLM提取
  - [ ] `validate_extracted_knowledge()` - 结果验证

- [ ] `vector_store_manager.py`
  - [ ] `create_vector_store()` - 创建向量库
  - [ ] `split_documents()` - 文档切块
  - [ ] `add_documents_to_store()` - 添加文档

- [ ] `knowledge_processor.py`
  - [ ] `process_all()` - 完整流程(已有框架,补充细节)

### 依赖包安装:

```bash
pip install langchain langchain-community langchain-chroma
pip install langchain-huggingface
pip install pymupdf unstructured python-docx python-pptx
pip install deepseek-sdk  # 或你用的LLM SDK
```

---

## 🎯 测试建议

```python
# 1. 先单独测试每个模块
python knowledge_organizer.py  # 检查分组是否正确
python document_loader.py      # 检查文档加载
python knowledge_extractor.py  # 检查LLM提取

# 2. 再运行完整Pipeline
python knowledge_processor.py

# 3. 验证结果
# - 检查 memories/knowledge/{domain}/structured.json
# - 检查 vector_db/{domain}/ 是否有向量文件
# - 测试向量搜索是否work
```

这个框架清晰吗?开始填充代码有问题随时问我!

```plaintext
原始知识文件(PDF/DOC/PPTX)
         ↓
【处理Pipeline】
         ↓
输出1: 结构化JSON → /memories/knowledge/{domain}/
输出2: 向量数据库 → vector_db/{domain}/
         ↓
供DeepAgent使用(一次处理,永久使用)
```

```python
步骤1: 扫描文件夹,自动分组知识块
  → 输入: knowledge_base目录
  → 输出: { "domain": { "group": {...} } }

步骤2: 加载文档内容
  → 输入: 文件路径列表
  → 输出: List[Document]

步骤3: 文本提取和清洗
  → 输入: 原始Document
  → 输出: 清洗后的Document

步骤4: 结构化知识提取(LLM)
  → 输入: 清洗后的Document
  → 输出: 结构化JSON(概念/指标/方法/案例)

步骤5: 保存结构化知识到/memories/
  → 输入: 结构化JSON
  → 输出: 持久化文件

步骤6: 文档切块(Text Splitting)
  → 输入: Document
  → 输出: List[Document chunks]

步骤7: 向量化并存入ChromaDB
  → 输入: Document chunks
  → 输出: 持久化向量库
```

```python
"""
知识文件自动扫描和分组
"""
from pathlib import Path
from typing import Dict, List
import re
from difflib import SequenceMatcher

class KnowledgeOrganizer:
    """知识文件组织器"""

    def __init__(self, knowledge_base_dir: str):
        self.knowledge_base_dir = Path(knowledge_base_dir)
        self.supported_extensions = {'.pdf', '.doc', '.docx', '.pptx'}

    def clean_filename(self, filename: str) -> str:
        """
        清理文件名,去除噪音

        TODO: 实现以下功能
        1. 去除[防断更微coc36666]等噪音
        2. 去除时间戳_20250706193405
        3. 去除_笔记后缀
        4. 清理多余空格

        输入: "01第一节 中国经济的三驾马车[防断更微coc36666]_笔记.pdf"
        输出: "01第一节 中国经济的三驾马车"
        """
        # TODO: 你来实现
        pass

    def extract_sequence_number(self, filename: str) -> tuple:
        """
        提取序号

        TODO:
        1. 用正则提取开头的数字
        2. 返回(数字, 原始字符串)

        输入: "01第一节"
        输出: (1, "01")
        """
        # TODO: 你来实现
        pass

    def calculate_similarity(self, str1: str, str2: str) -> float:
        """
        计算文件名相似度(0-1)

        提示: 使用difflib.SequenceMatcher
        """
        return SequenceMatcher(None, str1, str2).ratio()

    def group_files_by_similarity(
        self,
        files: List[Path],
        similarity_threshold: float = 0.7
    ) -> Dict:
        """
        根据相似度分组

        TODO: 实现分组逻辑
        1. 遍历文件,清理文件名
        2. 比较相似度,相似且序号相同→同组
        3. 返回分组结果

        返回格式:
        {
            "group_key": {
                "files": [Path列表],
                "topic": "主题名",
                "sequence": 序号,
            }
        }
        """
        # TODO: 你来实现
        pass

    def scan_and_organize(self) -> Dict:
        """
        扫描目录并组织

        TODO:
        1. 扫描knowledge_base_dir下的所有文件
        2. 调用group_files_by_similarity分组
        3. 为每组选择主文件(优先级: PDF笔记 > DOC > PPTX)

        返回格式:
        {
            "domain_name": {
                "group_key": {
                    "files": [...],
                    "topic": "...",
                    "primary_file": Path对象,
                }
            }
        }
        """
        # TODO: 你来实现
        pass


# ============ 使用示例 ============
if __name__ == "__main__":
    organizer = KnowledgeOrganizer(
        "/Users/zhou/Project/AnalystChain/jupyter_notebook/macroeconomic_analysis/knowledge_base"
    )

    result = organizer.scan_and_organize()

    # 打印结果(检查)
    for domain, groups in result.items():
        print(f"领域: {domain}")
        for group_key, info in groups.items():
            print(f"  - {info['topic']}: {len(info['files'])}个文件")
```

```python
"""
多格式文档加载和清洗
"""
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
)
from langchain_core.documents import Document
from pathlib import Path
from typing import List

class MultiFormatDocumentLoader:
    """多格式文档加载器"""

    def load_pdf(self, file_path: Path) -> List[Document]:
        """
        加载PDF文件

        TODO:
        1. 使用PyMuPDFLoader加载
        2. 处理可能的OCR需求
        3. 返回Document列表

        提示:
        loader = PyMuPDFLoader(str(file_path))
        docs = loader.load()
        """
        # TODO: 你来实现
        pass

    def load_word(self, file_path: Path) -> List[Document]:
        """
        加载Word文件

        TODO:
        1. 使用UnstructuredWordDocumentLoader
        2. 处理.doc和.docx

        提示:
        loader = UnstructuredWordDocumentLoader(str(file_path))
        docs = loader.load()
        """
        # TODO: 你来实现
        pass

    def load_ppt(self, file_path: Path) -> List[Document]:
        """
        加载PPT文件

        TODO:
        1. 使用UnstructuredPowerPointLoader
        2. 提取文本内容(图表可能提取不完整)
        """
        # TODO: 你来实现
        pass

    def load_document(self, file_path: Path) -> List[Document]:
        """
        根据文件类型自动选择加载器

        TODO:
        1. 根据file_path.suffix选择对应的loader
        2. 调用对应的load_xxx方法
        3. 处理加载失败的情况
        """
        suffix = file_path.suffix.lower()

        try:
            if suffix == '.pdf':
                return self.load_pdf(file_path)
            elif suffix in ['.doc', '.docx']:
                return self.load_word(file_path)
            elif suffix in ['.ppt', '.pptx']:
                return self.load_ppt(file_path)
            else:
                print(f"不支持的文件格式: {suffix}")
                return []
        except Exception as e:
            print(f"加载文件失败 {file_path}: {e}")
            return []

    def clean_document_text(self, doc: Document) -> Document:
        """
        清洗文档文本

        TODO: 实现文本清洗
        1. 去除多余空白字符
        2. 去除特殊字符(如\uf06c)
        3. 规范化换行
        4. 保留原始metadata

        返回: 清洗后的Document
        """
        # TODO: 你来实现
        pass

    def load_and_clean(self, file_path: Path) -> List[Document]:
        """加载并清洗文档"""
        docs = self.load_document(file_path)
        return [self.clean_document_text(doc) for doc in docs]


# ============ 使用示例 ============
if __name__ == "__main__":
    loader = MultiFormatDocumentLoader()

    docs = loader.load_and_clean(
        Path("/path/to/01第一节_笔记.pdf")
    )

    print(f"加载了 {len(docs)} 个文档")
    print(f"第一页内容: {docs[0].page_content[:200]}")
```

```python
"""
用LLM从文档中提取结构化知识
"""
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from typing import Dict, List
import json

class KnowledgeExtractor:
    """结构化知识提取器"""

    def __init__(self, model_name: str = "deepseek-chat"):
        """
        TODO:
        1. 初始化LLM模型
        2. 定义提取提示词模板
        """
        self.llm = init_chat_model(model_name, model_provider="deepseek")

        # TODO: 定义提取提示词
        self.extract_prompt = ChatPromptTemplate.from_messages([
            ("system", """
你是知识提取专家。从文档中提取结构化知识。

输出JSON格式:
{
  "topic": "主题名称",
  "key_concepts": [
    {"name": "概念名", "definition": "定义", "importance": "重要性"}
  ],
  "indicators": [
    {"name": "指标名", "calculation": "计算方法", "interpretation": "解读"}
  ],
  "analysis_methods": [
    {"name": "方法名", "steps": "步骤", "application": "应用场景"}
  ],
  "cases": [
    {"description": "案例描述", "insight": "启示"}
  ],
  "summary": "知识总结(200字以内)"
}
"""),
            ("human", "文档内容:\n\n{document_content}\n\n请提取结构化知识:")
        ])

    def extract_from_documents(
        self,
        docs: List[Document],
        topic: str
    ) -> Dict:
        """
        从文档列表中提取结构化知识

        TODO:
        1. 合并多个Document的内容
        2. 调用LLM提取
        3. 解析JSON结果
        4. 处理提取失败的情况

        Args:
            docs: 文档列表
            topic: 知识主题(如"中国经济的三驾马车")

        Returns:
            结构化知识dict
        """
        # TODO: 你来实现
        # 提示:
        # 1. 合并文档内容: content = "\n\n".join([doc.page_content for doc in docs])
        # 2. 调用LLM: chain = self.extract_prompt | self.llm
        # 3. 解析返回: json.loads(response.content)
        pass

    def validate_extracted_knowledge(self, knowledge: Dict) -> bool:
        """
        验证提取的知识是否完整

        TODO:
        检查必需字段是否存在:
        - topic
        - key_concepts
        - summary
        """
        # TODO: 你来实现
        pass


# ============ 使用示例 ============
if __name__ == "__main__":
    extractor = KnowledgeExtractor()

    # 假设已经加载了文档
    docs = [...]

    knowledge = extractor.extract_from_documents(
        docs=docs,
        topic="中国经济的三驾马车"
    )

    print(json.dumps(knowledge, ensure_ascii=False, indent=2))
```

```python
"""
向量数据库管理
"""
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pathlib import Path
from typing import List

class VectorStoreManager:
    """向量存储管理器"""

    def __init__(
        self,
        embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
        persist_directory: str = "./vector_db"
    ):
        """
        TODO:
        1. 初始化Embeddings模型
        2. 初始化文本切分器
        3. 设置持久化目录
        """
        # TODO: 你来实现
        # 提示:
        # self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        # self.text_splitter = RecursiveCharacterTextSplitter(...)
        # self.persist_directory = Path(persist_directory)
        pass

    def create_vector_store(self, domain: str) -> Chroma:
        """
        为指定领域创建向量存储

        TODO:
        1. 创建领域专属的persist_directory
        2. 初始化Chroma

        Args:
            domain: 领域名(如"macro_economy")

        Returns:
            Chroma向量存储对象
        """
        # TODO: 你来实现
        pass

    def split_documents(
        self,
        docs: List[Document],
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> List[Document]:
        """
        文档切块

        TODO:
        1. 使用text_splitter切分
        2. 返回切块列表
        """
        # TODO: 你来实现
        pass

    def add_documents_to_store(
        self,
        domain: str,
        docs: List[Document],
        metadata: Dict = None
    ):
        """
        添加文档到向量库

        TODO:
        1. 切分文档
        2. 为每个chunk添加metadata
        3. 加入向量库
        4. 持久化

        Args:
            domain: 领域名
            docs: 文档列表
            metadata: 额外的metadata(如topic, sequence)
        """
        # TODO: 你来实现
        pass

    def search_similar(
        self,
        domain: str,
        query: str,
        k: int = 5
    ) -> List[Document]:
        """
        语义搜索

        TODO:
        1. 加载对应domain的向量库
        2. 执行相似度搜索
        3. 返回结果
        """
        # TODO: 你来实现
        pass


# ============ 使用示例 ============
if __name__ == "__main__":
    manager = VectorStoreManager()

    # 添加文档
    manager.add_documents_to_store(
        domain="macro_economy",
        docs=[...],
        metadata={"topic": "三驾马车", "sequence": 1}
    )

    # 搜索
    results = manager.search_similar(
        domain="macro_economy",
        query="什么是三驾马车?"
    )
```

```python
"""
知识处理主流程
"""
from pathlib import Path
import json
from typing import Dict

from knowledge_organizer import KnowledgeOrganizer
from document_loader import MultiFormatDocumentLoader
from knowledge_extractor import KnowledgeExtractor
from vector_store_manager import VectorStoreManager

class KnowledgeProcessor:
    """知识处理主Pipeline"""

    def __init__(
        self,
        knowledge_base_dir: str,
        memories_dir: str = "./memories/knowledge",
        vector_db_dir: str = "./vector_db"
    ):
        """
        TODO: 初始化各个组件
        """
        self.organizer = KnowledgeOrganizer(knowledge_base_dir)
        self.loader = MultiFormatDocumentLoader()
        self.extractor = KnowledgeExtractor()
        self.vector_manager = VectorStoreManager(persist_directory=vector_db_dir)

        self.memories_dir = Path(memories_dir)
        self.memories_dir.mkdir(parents=True, exist_ok=True)

    def save_to_memories(self, domain: str, knowledge: Dict):
        """
        保存到/memories/目录(持久化)

        TODO:
        1. 创建domain子目录
        2. 保存structured.json
        3. 生成summary.md(可选)
        """
        domain_dir = self.memories_dir / domain
        domain_dir.mkdir(parents=True, exist_ok=True)

        # 保存JSON
        with open(domain_dir / "structured.json", "w", encoding="utf-8") as f:
            json.dump(knowledge, f, ensure_ascii=False, indent=2)

        print(f"✅ 已保存到 {domain_dir}/structured.json")

    def process_all(self):
        """
        处理所有知识文件的主流程

        TODO: 实现完整流程
        1. 调用organizer扫描分组
        2. 遍历每个domain和group
        3. 加载文档
        4. 提取结构化知识
        5. 保存到/memories/
        6. 向量化存储
        7. 打印处理进度
        """
        print("=" * 80)
        print("🚀 开始处理知识文件")
        print("=" * 80)

        # 步骤1: 扫描组织
        organized = self.organizer.scan_and_organize()

        # 步骤2: 遍历处理
        for domain, groups in organized.items():
            print(f"\n📂 处理领域: {domain}")

            domain_knowledge = {
                "domain": domain,
                "knowledge_blocks": []
            }

            for group_key, group_info in groups.items():
                topic = group_info['topic']
                primary_file = group_info['primary_file']

                print(f"  📖 处理: {topic}")
                print(f"     文件: {primary_file.name}")

                try:
                    # TODO: 步骤3 - 加载文档
                    docs = self.loader.load_and_clean(primary_file)

                    # TODO: 步骤4 - 提取结构化知识
                    structured = self.extractor.extract_from_documents(
                        docs=docs,
                        topic=topic
                    )

                    domain_knowledge["knowledge_blocks"].append({
                        "topic": topic,
                        "sequence": group_info['sequence'],
                        "structured": structured,
                    })

                    # TODO: 步骤6 - 向量化
                    self.vector_manager.add_documents_to_store(
                        domain=domain,
                        docs=docs,
                        metadata={
                            "topic": topic,
                            "sequence": group_info['sequence'],
                        }
                    )

                    print(f"     ✅ 完成")

                except Exception as e:
                    print(f"     ❌ 失败: {e}")

            # TODO: 步骤5 - 保存到/memories/
            self.save_to_memories(domain, domain_knowledge)

        print("\n" + "=" * 80)
        print("🎉 所有知识处理完成!")
        print("=" * 80)


# ============ 主入口 ============
if __name__ == "__main__":
    processor = KnowledgeProcessor(
        knowledge_base_dir="/Users/zhou/Project/AnalystChain/jupyter_notebook/macroeconomic_analysis/knowledge_base",
        memories_dir="/Users/zhou/Project/AnalystChain/memories/knowledge",
        vector_db_dir="/Users/zhou/Project/AnalystChain/vector_db"
    )

    # 一次性处理所有知识
    processor.process_all()
```

```shellscript
pip install langchain langchain-community langchain-chroma
pip install langchain-huggingface
pip install pymupdf unstructured python-docx python-pptx
pip install deepseek-sdk  # 或你用的LLM SDK
```

```python
# 1. 先单独测试每个模块
python knowledge_organizer.py  # 检查分组是否正确
python document_loader.py      # 检查文档加载
python knowledge_extractor.py  # 检查LLM提取

# 2. 再运行完整Pipeline
python knowledge_processor.py

# 3. 验证结果
# - 检查 memories/knowledge/{domain}/structured.json
# - 检查 vector_db/{domain}/ 是否有向量文件
# - 测试向量搜索是否work
```

```plaintext
原始知识文件(PDF/DOC/PPTX)
         ↓
【处理Pipeline】
         ↓
输出1: 结构化JSON → /memories/knowledge/{domain}/
输出2: 向量数据库 → vector_db/{domain}/
         ↓
供DeepAgent使用(一次处理,永久使用)
```

```python
步骤1: 扫描文件夹,自动分组知识块
  → 输入: knowledge_base目录
  → 输出: { "domain": { "group": {...} } }

步骤2: 加载文档内容
  → 输入: 文件路径列表
  → 输出: List[Document]

步骤3: 文本提取和清洗
  → 输入: 原始Document
  → 输出: 清洗后的Document

步骤4: 结构化知识提取(LLM)
  → 输入: 清洗后的Document
  → 输出: 结构化JSON(概念/指标/方法/案例)

步骤5: 保存结构化知识到/memories/
  → 输入: 结构化JSON
  → 输出: 持久化文件

步骤6: 文档切块(Text Splitting)
  → 输入: Document
  → 输出: List[Document chunks]

步骤7: 向量化并存入ChromaDB
  → 输入: Document chunks
  → 输出: 持久化向量库
```

```python
"""
知识文件自动扫描和分组
"""
from pathlib import Path
from typing import Dict, List
import re
from difflib import SequenceMatcher

class KnowledgeOrganizer:
    """知识文件组织器"""

    def __init__(self, knowledge_base_dir: str):
        self.knowledge_base_dir = Path(knowledge_base_dir)
        self.supported_extensions = {'.pdf', '.doc', '.docx', '.pptx'}

    def clean_filename(self, filename: str) -> str:
        """
        清理文件名,去除噪音

        TODO: 实现以下功能
        1. 去除[防断更微coc36666]等噪音
        2. 去除时间戳_20250706193405
        3. 去除_笔记后缀
        4. 清理多余空格

        输入: "01第一节 中国经济的三驾马车[防断更微coc36666]_笔记.pdf"
        输出: "01第一节 中国经济的三驾马车"
        """
        # TODO: 你来实现
        pass

    def extract_sequence_number(self, filename: str) -> tuple:
        """
        提取序号

        TODO:
        1. 用正则提取开头的数字
        2. 返回(数字, 原始字符串)

        输入: "01第一节"
        输出: (1, "01")
        """
        # TODO: 你来实现
        pass

    def calculate_similarity(self, str1: str, str2: str) -> float:
        """
        计算文件名相似度(0-1)

        提示: 使用difflib.SequenceMatcher
        """
        return SequenceMatcher(None, str1, str2).ratio()

    def group_files_by_similarity(
        self,
        files: List[Path],
        similarity_threshold: float = 0.7
    ) -> Dict:
        """
        根据相似度分组

        TODO: 实现分组逻辑
        1. 遍历文件,清理文件名
        2. 比较相似度,相似且序号相同→同组
        3. 返回分组结果

        返回格式:
        {
            "group_key": {
                "files": [Path列表],
                "topic": "主题名",
                "sequence": 序号,
            }
        }
        """
        # TODO: 你来实现
        pass

    def scan_and_organize(self) -> Dict:
        """
        扫描目录并组织

        TODO:
        1. 扫描knowledge_base_dir下的所有文件
        2. 调用group_files_by_similarity分组
        3. 为每组选择主文件(优先级: PDF笔记 > DOC > PPTX)

        返回格式:
        {
            "domain_name": {
                "group_key": {
                    "files": [...],
                    "topic": "...",
                    "primary_file": Path对象,
                }
            }
        }
        """
        # TODO: 你来实现
        pass


# ============ 使用示例 ============
if __name__ == "__main__":
    organizer = KnowledgeOrganizer(
        "/Users/zhou/Project/AnalystChain/jupyter_notebook/macroeconomic_analysis/knowledge_base"
    )

    result = organizer.scan_and_organize()

    # 打印结果(检查)
    for domain, groups in result.items():
        print(f"领域: {domain}")
        for group_key, info in groups.items():
            print(f"  - {info['topic']}: {len(info['files'])}个文件")
```

```python
"""
多格式文档加载和清洗
"""
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
)
from langchain_core.documents import Document
from pathlib import Path
from typing import List

class MultiFormatDocumentLoader:
    """多格式文档加载器"""

    def load_pdf(self, file_path: Path) -> List[Document]:
        """
        加载PDF文件

        TODO:
        1. 使用PyMuPDFLoader加载
        2. 处理可能的OCR需求
        3. 返回Document列表

        提示:
        loader = PyMuPDFLoader(str(file_path))
        docs = loader.load()
        """
        # TODO: 你来实现
        pass

    def load_word(self, file_path: Path) -> List[Document]:
        """
        加载Word文件

        TODO:
        1. 使用UnstructuredWordDocumentLoader
        2. 处理.doc和.docx

        提示:
        loader = UnstructuredWordDocumentLoader(str(file_path))
        docs = loader.load()
        """
        # TODO: 你来实现
        pass

    def load_ppt(self, file_path: Path) -> List[Document]:
        """
        加载PPT文件

        TODO:
        1. 使用UnstructuredPowerPointLoader
        2. 提取文本内容(图表可能提取不完整)
        """
        # TODO: 你来实现
        pass

    def load_document(self, file_path: Path) -> List[Document]:
        """
        根据文件类型自动选择加载器

        TODO:
        1. 根据file_path.suffix选择对应的loader
        2. 调用对应的load_xxx方法
        3. 处理加载失败的情况
        """
        suffix = file_path.suffix.lower()

        try:
            if suffix == '.pdf':
                return self.load_pdf(file_path)
            elif suffix in ['.doc', '.docx']:
                return self.load_word(file_path)
            elif suffix in ['.ppt', '.pptx']:
                return self.load_ppt(file_path)
            else:
                print(f"不支持的文件格式: {suffix}")
                return []
        except Exception as e:
            print(f"加载文件失败 {file_path}: {e}")
            return []

    def clean_document_text(self, doc: Document) -> Document:
        """
        清洗文档文本

        TODO: 实现文本清洗
        1. 去除多余空白字符
        2. 去除特殊字符(如\uf06c)
        3. 规范化换行
        4. 保留原始metadata

        返回: 清洗后的Document
        """
        # TODO: 你来实现
        pass

    def load_and_clean(self, file_path: Path) -> List[Document]:
        """加载并清洗文档"""
        docs = self.load_document(file_path)
        return [self.clean_document_text(doc) for doc in docs]


# ============ 使用示例 ============
if __name__ == "__main__":
    loader = MultiFormatDocumentLoader()

    docs = loader.load_and_clean(
        Path("/path/to/01第一节_笔记.pdf")
    )

    print(f"加载了 {len(docs)} 个文档")
    print(f"第一页内容: {docs[0].page_content[:200]}")
```

```python
"""
用LLM从文档中提取结构化知识
"""
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from typing import Dict, List
import json

class KnowledgeExtractor:
    """结构化知识提取器"""

    def __init__(self, model_name: str = "deepseek-chat"):
        """
        TODO:
        1. 初始化LLM模型
        2. 定义提取提示词模板
        """
        self.llm = init_chat_model(model_name, model_provider="deepseek")

        # TODO: 定义提取提示词
        self.extract_prompt = ChatPromptTemplate.from_messages([
            ("system", """
你是知识提取专家。从文档中提取结构化知识。

输出JSON格式:
{
  "topic": "主题名称",
  "key_concepts": [
    {"name": "概念名", "definition": "定义", "importance": "重要性"}
  ],
  "indicators": [
    {"name": "指标名", "calculation": "计算方法", "interpretation": "解读"}
  ],
  "analysis_methods": [
    {"name": "方法名", "steps": "步骤", "application": "应用场景"}
  ],
  "cases": [
    {"description": "案例描述", "insight": "启示"}
  ],
  "summary": "知识总结(200字以内)"
}
"""),
            ("human", "文档内容:\n\n{document_content}\n\n请提取结构化知识:")
        ])

    def extract_from_documents(
        self,
        docs: List[Document],
        topic: str
    ) -> Dict:
        """
        从文档列表中提取结构化知识

        TODO:
        1. 合并多个Document的内容
        2. 调用LLM提取
        3. 解析JSON结果
        4. 处理提取失败的情况

        Args:
            docs: 文档列表
            topic: 知识主题(如"中国经济的三驾马车")

        Returns:
            结构化知识dict
        """
        # TODO: 你来实现
        # 提示:
        # 1. 合并文档内容: content = "\n\n".join([doc.page_content for doc in docs])
        # 2. 调用LLM: chain = self.extract_prompt | self.llm
        # 3. 解析返回: json.loads(response.content)
        pass

    def validate_extracted_knowledge(self, knowledge: Dict) -> bool:
        """
        验证提取的知识是否完整

        TODO:
        检查必需字段是否存在:
        - topic
        - key_concepts
        - summary
        """
        # TODO: 你来实现
        pass


# ============ 使用示例 ============
if __name__ == "__main__":
    extractor = KnowledgeExtractor()

    # 假设已经加载了文档
    docs = [...]

    knowledge = extractor.extract_from_documents(
        docs=docs,
        topic="中国经济的三驾马车"
    )

    print(json.dumps(knowledge, ensure_ascii=False, indent=2))
```

```python
"""
向量数据库管理
"""
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pathlib import Path
from typing import List

class VectorStoreManager:
    """向量存储管理器"""

    def __init__(
        self,
        embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
        persist_directory: str = "./vector_db"
    ):
        """
        TODO:
        1. 初始化Embeddings模型
        2. 初始化文本切分器
        3. 设置持久化目录
        """
        # TODO: 你来实现
        # 提示:
        # self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        # self.text_splitter = RecursiveCharacterTextSplitter(...)
        # self.persist_directory = Path(persist_directory)
        pass

    def create_vector_store(self, domain: str) -> Chroma:
        """
        为指定领域创建向量存储

        TODO:
        1. 创建领域专属的persist_directory
        2. 初始化Chroma

        Args:
            domain: 领域名(如"macro_economy")

        Returns:
            Chroma向量存储对象
        """
        # TODO: 你来实现
        pass

    def split_documents(
        self,
        docs: List[Document],
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> List[Document]:
        """
        文档切块

        TODO:
        1. 使用text_splitter切分
        2. 返回切块列表
        """
        # TODO: 你来实现
        pass

    def add_documents_to_store(
        self,
        domain: str,
        docs: List[Document],
        metadata: Dict = None
    ):
        """
        添加文档到向量库

        TODO:
        1. 切分文档
        2. 为每个chunk添加metadata
        3. 加入向量库
        4. 持久化

        Args:
            domain: 领域名
            docs: 文档列表
            metadata: 额外的metadata(如topic, sequence)
        """
        # TODO: 你来实现
        pass

    def search_similar(
        self,
        domain: str,
        query: str,
        k: int = 5
    ) -> List[Document]:
        """
        语义搜索

        TODO:
        1. 加载对应domain的向量库
        2. 执行相似度搜索
        3. 返回结果
        """
        # TODO: 你来实现
        pass


# ============ 使用示例 ============
if __name__ == "__main__":
    manager = VectorStoreManager()

    # 添加文档
    manager.add_documents_to_store(
        domain="macro_economy",
        docs=[...],
        metadata={"topic": "三驾马车", "sequence": 1}
    )

    # 搜索
    results = manager.search_similar(
        domain="macro_economy",
        query="什么是三驾马车?"
    )
```

```python
"""
知识处理主流程
"""
from pathlib import Path
import json
from typing import Dict

from knowledge_organizer import KnowledgeOrganizer
from document_loader import MultiFormatDocumentLoader
from knowledge_extractor import KnowledgeExtractor
from vector_store_manager import VectorStoreManager

class KnowledgeProcessor:
    """知识处理主Pipeline"""

    def __init__(
        self,
        knowledge_base_dir: str,
        memories_dir: str = "./memories/knowledge",
        vector_db_dir: str = "./vector_db"
    ):
        """
        TODO: 初始化各个组件
        """
        self.organizer = KnowledgeOrganizer(knowledge_base_dir)
        self.loader = MultiFormatDocumentLoader()
        self.extractor = KnowledgeExtractor()
        self.vector_manager = VectorStoreManager(persist_directory=vector_db_dir)

        self.memories_dir = Path(memories_dir)
        self.memories_dir.mkdir(parents=True, exist_ok=True)

    def save_to_memories(self, domain: str, knowledge: Dict):
        """
        保存到/memories/目录(持久化)

        TODO:
        1. 创建domain子目录
        2. 保存structured.json
        3. 生成summary.md(可选)
        """
        domain_dir = self.memories_dir / domain
        domain_dir.mkdir(parents=True, exist_ok=True)

        # 保存JSON
        with open(domain_dir / "structured.json", "w", encoding="utf-8") as f:
            json.dump(knowledge, f, ensure_ascii=False, indent=2)

        print(f"✅ 已保存到 {domain_dir}/structured.json")

    def process_all(self):
        """
        处理所有知识文件的主流程

        TODO: 实现完整流程
        1. 调用organizer扫描分组
        2. 遍历每个domain和group
        3. 加载文档
        4. 提取结构化知识
        5. 保存到/memories/
        6. 向量化存储
        7. 打印处理进度
        """
        print("=" * 80)
        print("🚀 开始处理知识文件")
        print("=" * 80)

        # 步骤1: 扫描组织
        organized = self.organizer.scan_and_organize()

        # 步骤2: 遍历处理
        for domain, groups in organized.items():
            print(f"\n📂 处理领域: {domain}")

            domain_knowledge = {
                "domain": domain,
                "knowledge_blocks": []
            }

            for group_key, group_info in groups.items():
                topic = group_info['topic']
                primary_file = group_info['primary_file']

                print(f"  📖 处理: {topic}")
                print(f"     文件: {primary_file.name}")

                try:
                    # TODO: 步骤3 - 加载文档
                    docs = self.loader.load_and_clean(primary_file)

                    # TODO: 步骤4 - 提取结构化知识
                    structured = self.extractor.extract_from_documents(
                        docs=docs,
                        topic=topic
                    )

                    domain_knowledge["knowledge_blocks"].append({
                        "topic": topic,
                        "sequence": group_info['sequence'],
                        "structured": structured,
                    })

                    # TODO: 步骤6 - 向量化
                    self.vector_manager.add_documents_to_store(
                        domain=domain,
                        docs=docs,
                        metadata={
                            "topic": topic,
                            "sequence": group_info['sequence'],
                        }
                    )

                    print(f"     ✅ 完成")

                except Exception as e:
                    print(f"     ❌ 失败: {e}")

            # TODO: 步骤5 - 保存到/memories/
            self.save_to_memories(domain, domain_knowledge)

        print("\n" + "=" * 80)
        print("🎉 所有知识处理完成!")
        print("=" * 80)


# ============ 主入口 ============
if __name__ == "__main__":
    processor = KnowledgeProcessor(
        knowledge_base_dir="/Users/zhou/Project/AnalystChain/jupyter_notebook/macroeconomic_analysis/knowledge_base",
        memories_dir="/Users/zhou/Project/AnalystChain/memories/knowledge",
        vector_db_dir="/Users/zhou/Project/AnalystChain/vector_db"
    )

    # 一次性处理所有知识
    processor.process_all()
```

```shellscript
pip install langchain langchain-community langchain-chroma
pip install langchain-huggingface
pip install pymupdf unstructured python-docx python-pptx
pip install deepseek-sdk  # 或你用的LLM SDK
```

```python
# 1. 先单独测试每个模块
python knowledge_organizer.py  # 检查分组是否正确
python document_loader.py      # 检查文档加载
python knowledge_extractor.py  # 检查LLM提取

# 2. 再运行完整Pipeline
python knowledge_processor.py

# 3. 验证结果
# - 检查 memories/knowledge/{domain}/structured.json
# - 检查 vector_db/{domain}/ 是否有向量文件
# - 测试向量搜索是否work
```

