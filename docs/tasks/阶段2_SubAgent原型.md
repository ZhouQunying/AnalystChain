# 阶段2：SubAgent原型

**目标**：实现宏观经济分析SubAgent（动态工具架构）
**状态**：🟡 进行中

## 为什么这样做

**架构定位**：3层架构的【Agent层】第1个SubAgent

**为什么先做宏观SubAgent**：
- 宏观分析是投资决策的基础（自上而下分析法）
- 知识库已完成（阶段1），可直接使用
- 验证动态工具架构可行性

**核心设计决策**：动态工具选择（非预制固定工具）
- **问题**：AKShare有100+宏观接口，预制工具无法覆盖所有查询场景
- **方案**：自动发现 + 动态过滤
- **原理**：用户查询 → 检索/过滤最相关工具 → Agent仅使用选中的工具

**技术选型**：
- LLM：deepseek-chat（支持工具调用）
- 数据源：AKShare（免费、宏观数据全覆盖）
- Agent框架：LangChain + Middleware/MCP

---

## 动态工具架构方案

### 方案对比

| 方案 | 工具发现方式 | 工具维护 | 推荐场景 |
|------|------------|---------|---------|
| **B. 向量索引** | 预建向量库 + 语义检索 | 需手动/脚本更新 | 开发阶段快速验证 |
| **E. MCP Server** | Python反射扫描 | 自动发现（零维护） | 生产环境 |
| **F. Context7** | 外部文档服务 | 依赖第三方更新 | 补充文档查询 |

### 方案关系

```
[方案B：向量索引] ─────────────────────────────────────┐
  快速验证，需手动维护工具描述                           │
       ↓ 验证通过后升级                                │
[方案E：MCP Server] ← 主路径（生产环境）                │
  Python反射自动发现，零维护                            │
       ↑ 文档增强（补充）                              │
[方案F：Context7] ─────────────────────────────────────┘
  提供API文档查询，不作为工具发现主路径
```

**推荐路径**：
1. **开发阶段**：方案B（向量索引），快速验证架构
2. **生产阶段**：升级方案E（MCP Server），实现零维护
3. **补充**：方案F（Context7），提供API文档查询能力

---

## 方案B：向量索引（开发阶段）

### 架构

```
用户查询："最近的社会融资规模增量是多少？"
     ↓
[工具描述向量库] → 语义检索最相关工具
     ↓
返回：["macro_china_shrzgm", "macro_china_m2", ...]
     ↓
[Middleware] → 动态过滤Agent可用工具
     ↓
[Agent] → 仅使用检索到的工具进行调用
     ↓
返回分析结果
```

### 任务

| # | 任务 | What | Why | How Much | 状态 |
|---|------|------|-----|----------|------|
| 1 | 工具描述索引 | 构建AKShare宏观接口向量库 | 支持动态工具检索 | 50+接口描述入库 | |
| 2 | 动态工具检索器 | 实现查询→工具匹配 | 根据用户问题选最佳API | 检索Top-3准确率>80% | |
| 3 | Agent+Middleware | 实现动态工具过滤Agent | 验证架构可行性 | Agent可运行+动态选工具 | |
| 4 | 端到端测试 | 验证分析质量和性能 | 确保可用性 | 平均评分>=80+响应<30s | |

### 代码示例

**tool_index.py - 构建工具描述向量库**

```python
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# AKShare宏观接口描述（需手动维护）
AKSHARE_MACRO_TOOLS = [
    {
        "name": "macro_china_gdp",
        "api": "ak.macro_china_gdp()",
        "description": "中国GDP年率报告，季度GDP数据，包含GDP绝对值、同比增长率、三次产业数据",
        "keywords": ["GDP", "国内生产总值", "经济增长", "季度GDP"]
    },
    {
        "name": "macro_china_cpi",
        "api": "ak.macro_china_cpi()",
        "description": "中国CPI月率报告，居民消费价格指数，包含全国/城市/农村CPI同比环比数据",
        "keywords": ["CPI", "消费价格", "通胀", "物价"]
    },
    {
        "name": "macro_china_shrzgm",
        "api": "ak.macro_china_shrzgm()",
        "description": "社会融资规模增量统计，包含人民币贷款、委托贷款、信托贷款、企业债券等月度数据",
        "keywords": ["社融", "社会融资", "融资规模", "信贷"]
    },
    # ... 更多接口描述
]

def build_tool_index(tools: list, persist_dir: str):
    """构建工具描述向量库"""
    documents = []
    for tool in tools:
        content = f"{tool['description']} 关键词：{', '.join(tool['keywords'])}"
        doc = Document(
            page_content=content,
            metadata={"name": tool["name"], "api": tool["api"]}
        )
        documents.append(doc)

    embeddings = HuggingFaceEmbeddings(model_name="Alibaba-NLP/gte-Qwen2-1.5B-instruct")
    vectorstore = Chroma.from_documents(documents, embeddings, persist_directory=persist_dir)
    return vectorstore
```

**tool_retriever.py - 工具检索器**

```python
class ToolRetriever:
    """根据查询检索最相关的AKShare接口"""

    def __init__(self, persist_dir: str, embedding_model: str):
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vectorstore = Chroma(persist_directory=persist_dir, embedding_function=self.embeddings)

    def get_tool_names(self, query: str, k: int = 3) -> list[str]:
        """返回最相关的工具名称列表"""
        results = self.vectorstore.similarity_search(query, k=k)
        return [doc.metadata["name"] for doc in results]
```

### 维护说明

新增AKShare接口时需更新向量库：
1. 在`AKSHARE_MACRO_TOOLS`添加新接口描述
2. 重新运行`build_tool_index()`

---

## 方案E：MCP Server（生产环境）

### 核心优势

- **零维护**：Python反射自动扫描AKShare函数，新增接口自动发现
- **LangChain官方支持**：MCP是标准协议，使用`langchain-mcp-adapters`集成
- **完全自主**：不依赖第三方服务

### 架构

```
[AKShare MCP Server]
     ↑ 启动时反射扫描ak模块
     │ 自动发现所有macro_china_*函数
     │ 暴露为MCP工具
     ↓
[LangChain Agent]
     ↑ 通过MCP协议发现工具
     │ 运行时获取工具列表
     ↓
[用户查询] → [Agent调用工具] → [返回结果]
```

### 核心原理：Python反射（非爬虫）

```python
import akshare as ak
import inspect

def scan_akshare_macro_tools():
    """扫描AKShare所有macro_china_开头的函数"""
    tools = []
    for name in dir(ak):
        if not name.startswith("macro_china_"):
            continue
        func = getattr(ak, name)
        if not callable(func):
            continue

        # 提取docstring作为描述
        docstring = inspect.getdoc(func) or "暂无描述"
        short_desc = docstring.split('\n')[0]

        # 提取函数签名（参数）
        try:
            sig = inspect.signature(func)
            params = {
                p.name: {"type": "string", "default": str(p.default) if p.default != inspect.Parameter.empty else None}
                for p in sig.parameters.values()
            }
        except (ValueError, TypeError):
            params = {}

        tools.append({
            "name": name,
            "description": short_desc,
            "parameters": params
        })

    return tools

# 示例输出
# 发现 87 个宏观接口
# macro_china_gdp: 中国GDP年率报告，数据区间从2011-01-20至今
# macro_china_shrzgm: 社会融资规模增量统计
```

### MCP Server实现

```python
from mcp.server import Server
from mcp.types import Tool, TextContent
import akshare as ak

server = Server("akshare-macro")

# 启动时扫描所有工具
TOOLS_CACHE = scan_akshare_macro_tools()

@server.list_tools()
async def list_tools() -> list[Tool]:
    """MCP协议：返回所有可用工具"""
    return [
        Tool(
            name=tool["name"],
            description=tool["description"],
            inputSchema={"type": "object", "properties": tool["parameters"]}
        )
        for tool in TOOLS_CACHE
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """MCP协议：执行工具调用"""
    func = getattr(ak, name, None)
    if not func:
        return [TextContent(type="text", text=f"未知工具: {name}")]

    try:
        result = func(**arguments)
        text = result.tail(20).to_string() if hasattr(result, 'to_string') else str(result)
        return [TextContent(type="text", text=text)]
    except Exception as e:
        return [TextContent(type="text", text=f"调用失败: {e}")]
```

### LangChain集成

```python
from langchain_mcp_adapters import MCPToolkit

# 连接MCP Server
toolkit = MCPToolkit(server_url="http://localhost:8000/mcp")
tools = toolkit.get_tools()  # 自动获取所有MCP工具

# 创建Agent
agent = create_agent(model=llm, tools=tools)
```

### 版本管理

需保持AKShare包最新以发现新接口：

```python
# 启动时检查版本
import requests

def check_akshare_version():
    pypi_resp = requests.get("https://pypi.org/pypi/akshare/json")
    latest = pypi_resp.json()["info"]["version"]
    current = ak.__version__
    if current != latest:
        print(f"[警告] AKShare有新版本: {current} -> {latest}")
```

---

## 方案F：Context7（补充）

### 定位

**不作为工具发现主路径**，仅用于：
- 回答"AKShare有哪些宏观接口"类问题
- 提供API文档说明和使用示例

### 局限性

| 维度 | 状态 | 说明 |
|------|------|------|
| 准确度 | 高 | 语义搜索可靠 |
| 时效性 | 存疑 | 依赖Context7索引更新时间 |
| **可执行性** | **低** | 返回文档文本，非可调用工具 |

### 使用方式

```python
from mcp_context7 import query_docs

def get_api_docs(user_query: str) -> str:
    """查询AKShare API文档（仅文档，非工具调用）"""
    result = query_docs(
        libraryId="/websites/akshare_akfamily_xyz",
        query=user_query
    )
    return result  # 返回文档文本
```

---

## LLM调用链路说明

**关键：LLM不能直接执行Python，只能生成调用指令**

```
Step 1: 工具发现（反射/向量索引）
     ↓
Step 2: 转换为Schema（name + description + parameters）
     ↓
Step 3: 注册到Agent/MCP Server
     ↓
Step 4: LLM决策（生成JSON调用指令）
     ↓
Step 5: Agent/Server执行Python函数
     ↓
Step 6: 返回结果给LLM
```

**成功调用的关键因素**：
- 描述清晰（LLM能理解何时使用）
- 参数Schema正确（从函数签名自动生成）
- 异常处理（捕获错误返回友好信息）

---

## 端到端测试

**测试问题（10个）**：
1. 2024年GDP增长率是多少？
2. 当前的通胀水平如何？
3. 最新的社会融资规模增量是多少？
4. 当前经济处于什么周期？
5. LPR利率最近有什么变化？
6. 固定资产投资增速如何？
7. 根据当前经济周期，应该配置什么资产？
8. 投资时钟当前处于哪个阶段？
9. 给出当前宏观经济的整体判断
10. 从宏观角度看，周期性行业投资机会如何？

**评分标准（满分100）**：
- **数据准确30分**：来源正确+时间正确+数值准确
- **分析专业40分**：运用理论+逻辑清晰+结合知识库
- **结论清晰30分**：判断明确+建议可行+易于理解

**完成标准**：平均评分>=80 + 平均响应<30秒

---

## 参考项目

[easy_investment_Agent_crewai](https://github.com/liangdabiao/easy_investment_Agent_crewai)：基于AKShare+CrewAI的A股分析项目

**可借鉴**：
- 返回格式化文本（非原始DataFrame）
- 统一的输入Schema
- Agent角色配置分离

---

**规范**：[项目开发规范](../项目开发规范.md)
