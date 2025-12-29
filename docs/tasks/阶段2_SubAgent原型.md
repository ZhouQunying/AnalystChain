# 阶段2：SubAgent原型

**目标**：实现宏观经济分析SubAgent
**状态**：🟡 进行中
**当前**：端到端测试

## 任务

| # | 任务 | What | Why | How | How Much | 状态 |
|---|------|------|-----|-----|----------|------|
| 1 | Tools封装 | 封装数据和知识检索工具 | Agent需要调用 | 创建src/analyst_chain/tools/akshare_tools.py+按"代码示例"实现3个函数+创建knowledge_retrieval.py实现KnowledgeRetriever+编写tests/test_akshare_tools.py+执行测试 | 5个工具函数+测试通过 | ✅ |
| 2 | Agent实现 | 创建宏观分析SubAgent | 验证架构可行性 | 创建src/analyst_chain/agents/macro_agent.py+按"代码示例"实现create_macro_agent+创建notebooks/stage2_macro_agent.ipynb+测试Agent运行 | DeepAgent+SubAgent可运行 | ✅ |
| 3 | 端到端测试 | 验证分析质量和性能 | 确保可用性 | 在stage2_macro_agent.ipynb编写10个测试问题+使用"代码示例"的批量测试代码+保存结果到data/test_results.json+按评分方法评分+生成最终报告 | 平均评分≥80+响应<30s | 🟡 |

## 现在做什么

### 第1步：验证环境（5分钟）
- 确认`config/.env`有`DEEPSEEK_API_KEY=sk-xxx`（无则配置）

### 第2步：准备测试问题（30分钟）
在`notebooks/stage2_macro_agent.ipynb`编写10个测试问题：
1. 2024年GDP增长率是多少？
2. 当前的通胀水平如何？
3. 最新的PMI数据是多少？
4. 当前经济处于什么周期？
5. 经济周期转折的信号是什么？
6. 什么指标变化会预示周期转折？
7. 根据当前经济周期，应该配置什么资产？
8. 投资时钟当前处于哪个阶段？
9. 给出当前宏观经济的整体判断
10. 从宏观角度看，周期性行业投资机会如何？

### 第3步：批量运行测试（2小时）
- 编写批量测试循环代码
- 对每个问题：运行SubAgent+记录响应时间+保存回答
- 输出到JSON文件

### 第4步：评分和分析（1小时）
对每个回答评分（满分100）：
- **数据准确30分**：来源正确+时间正确+数值准确
- **分析专业40分**：运用理论+逻辑清晰+结合知识库（最重要，因为是核心能力）
- **结论清晰30分**：判断明确+建议可行+易于理解

统计平均分、响应时间，找出问题（评分<80的回答）

### 第5步：生成报告（30分钟）
- 整理测试报告：问题+回答+评分+响应时间
- 保存为JSON文件
- 检查是否达标：平均评分≥80+响应<30s

**完成标准**：测试报告生成 + 平均响应<30秒 + 平均评分≥80

**下一步**：完成后→阶段3多Agent协作（实现政策和股票SubAgent）

---

## 代码示例

### 任务#1：Tools封装代码

**akshare_tools.py核心结构**
```python
import akshare as ak
import pandas as pd
from typing import Optional

def get_gdp_quarterly() -> Optional[pd.DataFrame]:
    """获取季度GDP数据

    Returns:
        DataFrame包含：quarter（季度）、gdp（绝对值）、gdp_yoy（同比增长率）
    """
    try:
        df = ak.macro_china_gdp()  # 调用AKShare API
        # 数据清洗和格式化
        return df
    except Exception as e:
        print(f"获取GDP数据失败: {e}")
        return None

def get_cpi_monthly() -> Optional[pd.DataFrame]:
    """获取月度CPI数据"""
    # 类似实现
    pass

def get_pmi_manufacturing() -> Optional[pd.DataFrame]:
    """获取制造业PMI数据"""
    # 类似实现
    pass
```

**knowledge_retrieval.py核心结构**
```python
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import json

class KnowledgeRetriever:
    def __init__(self, vector_db_path: str = "data/processed/knowledge/vector_db/"):
        # 初始化向量库
        self.embeddings = HuggingFaceEmbeddings(model_name="Qwen3-Embedding-0.6B")
        self.vectorstore = Chroma(persist_directory=vector_db_path, embedding_function=self.embeddings)

    def vector_search(self, query: str, k: int = 3) -> str:
        """向量检索"""
        results = self.vectorstore.similarity_search(query, k=k)
        return "\n".join([doc.page_content for doc in results])

    def get_topic_knowledge(self, topic: str) -> str:
        """JSON知识查询"""
        json_path = f"data/processed/knowledge/json/{topic}.json"
        with open(json_path, 'r') as f:
            data = json.load(f)
        return json.dumps(data, ensure_ascii=False)
```

### 任务#2：Agent实现代码

**macro_agent.py核心结构**
```python
from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI
from ..tools.akshare_tools import get_gdp_quarterly, get_cpi_monthly, get_pmi_manufacturing
from ..tools.knowledge_retrieval import KnowledgeRetriever

def create_macro_agent():
    # 1. 初始化模型
    model = ChatOpenAI(
        model="deepseek-chat",
        openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
        openai_api_base="https://api.deepseek.com",
        temperature=0.7,
        streaming=True
    )

    # 2. 初始化知识检索器
    knowledge_retriever = KnowledgeRetriever()

    # 3. 定义SubAgent配置
    macroeconomic_subagent = {
        "name": "macroeconomic_analyst",
        "description": "宏观经济分析专家",
        "system_prompt": "你是宏观经济分析专家...",
        "tools": [
            get_gdp_quarterly,
            get_cpi_monthly,
            get_pmi_manufacturing,
            knowledge_retriever.vector_search,
            knowledge_retriever.get_topic_knowledge
        ]
    }

    # 4. 创建DeepAgent
    main_agent = create_deep_agent(
        model=model,
        subagents=[macroeconomic_subagent]
    )

    return main_agent
```

### 任务#3：批量测试完整代码

```python
import time
import json
from pathlib import Path

# 10个测试问题（已在第2步定义）
test_questions = [
    "2024年GDP增长率是多少？",
    "当前的通胀水平如何？",
    "最新的PMI数据是多少？",
    "当前经济处于什么周期？",
    "经济周期转折的信号是什么？",
    "什么指标变化会预示周期转折？",
    "根据当前经济周期，应该配置什么资产？",
    "投资时钟当前处于哪个阶段？",
    "给出当前宏观经济的整体判断",
    "从宏观角度看，周期性行业投资机会如何？"
]

# 批量运行
test_results = []
for i, question in enumerate(test_questions, 1):
    print(f"\n[{i}/{len(test_questions)}] 测试问题: {question}")
    start_time = time.time()

    # 运行Agent（假设main_agent是已创建的DeepAgent实例）
    response_text = ""
    for message_chunk, metadata in main_agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="messages"
    ):
        if hasattr(message_chunk, 'content') and message_chunk.content:
            response_text += message_chunk.content

    elapsed_time = time.time() - start_time

    test_results.append({
        "question_id": i,
        "question": question,
        "response": response_text,
        "time_taken": round(elapsed_time, 2),
        "score": None  # 待手动评分
    })

    print(f"响应时间: {elapsed_time:.2f}秒")

# 保存结果到JSON
output_path = Path("data/test_results.json")
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(test_results, f, ensure_ascii=False, indent=2)

print(f"\n测试结果已保存到: {output_path}")
```

### JSON报告格式

```json
{
  "test_date": "2025-12-29",
  "summary": {
    "total_questions": 10,
    "average_score": 85.3,
    "average_response_time": 25.6,
    "pass_threshold": {
      "score": 80,
      "time": 30
    },
    "overall_pass": true
  },
  "results": [
    {
      "question_id": 1,
      "question": "2024年GDP增长率是多少？",
      "response": "根据AKShare数据，2024年Q3的GDP同比增长率为4.6%...",
      "score_breakdown": {
        "data_accuracy": 28,
        "analysis_quality": 35,
        "conclusion_clarity": 25
      },
      "total_score": 88,
      "time_taken": 20.1,
      "passed": true
    }
  ]
}
```

### 评分方法

**数据准确（30分）**：
- 数据来源正确（10分）：明确引用AKShare
- 时间准确（10分）：数据时间点正确（如2024Q3）
- 数值准确（10分）：数值与实际数据一致

**分析专业（40分）**：
- 运用理论（15分）：引用经济周期、投资时钟等理论
- 逻辑清晰（15分）：推理过程合理、因果关系明确
- 结合知识库（10分）：使用向量检索到的知识

**结论清晰（30分）**：
- 判断明确（10分）：给出清晰的结论（如"当前处于复苏期"）
- 建议可行（10分）：投资建议具体可操作
- 易于理解（10分）：语言简洁、结构清晰

---

**规范**：[AI工作规范](../AI工作规范.md)
