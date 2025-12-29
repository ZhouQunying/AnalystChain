"""
宏观经济分析Agent

实现宏观经济分析SubAgent,结合实时数据和理论知识提供专业分析
"""

import os
import logging
from typing import Dict, Any, Optional
from deepagents import create_deep_agent

# 关闭干扰日志，确保流式输出体验
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("src.analyst_chain.tools.akshare_tools").setLevel(logging.WARNING)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from ..tools.akshare_tools import get_gdp_quarterly, get_cpi_monthly, get_pmi_manufacturing
from ..tools.knowledge_retrieval import KnowledgeRetriever


def create_macro_agent(model: Optional[BaseChatModel] = None) -> Any:
    """创建宏观经济分析SubAgent

    Args:
        model: LLM模型对象（BaseChatModel），默认使用DeepSeek模型

    Returns:
        DeepAgent实例，包含宏观经济分析SubAgent

    Example:
        >>> agent = create_macro_agent()
        >>> response = agent.run("最近GDP增长率如何？")
    """
    # 初始化知识检索器
    knowledge_retriever = KnowledgeRetriever()

    # 如果未提供模型，使用DeepSeek
    if model is None:
        model = ChatOpenAI(
            model="deepseek-reasoner",
            openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
            openai_api_base="https://api.deepseek.com",
            temperature=0.7,
            streaming=True,
        )

    # 定义宏观经济分析SubAgent配置
    macroeconomic_subagent = {
        "name": "macroeconomic_subagent",
        "description": "负责宏观经济分析,包括GDP、CPI、PMI等指标的数据获取、分析和趋势判断。结合实时数据和理论知识提供专业分析。",
        "system_prompt": """你是宏观经济分析专家。

核心能力：
1. 数据获取：使用AKShare工具获取最新宏观经济数据(GDP、CPI、PMI等)
2. 理论支撑：查询知识库获取相关经济理论和分析框架
3. 综合分析：结合实时数据和理论知识，提供专业的经济解读

分析流程：
1. 获取相关数据（使用get_gdp_quarterly/get_cpi_monthly/get_pmi_manufacturing）
2. 查询理论框架（使用knowledge_retriever.vector_search）
3. 分析数据趋势和周期特征
4. 给出专业判断和预测

输出要求：
- 数据准确，引用来源（如"根据AKShare最新数据..."）
- 分析专业，运用理论（如"根据经济周期理论..."）
- 逻辑清晰，结构完整（数据→理论→分析→结论）
- 结论明确，便于理解""",
        "tools": [
            get_gdp_quarterly,
            get_cpi_monthly,
            get_pmi_manufacturing,
            knowledge_retriever.vector_search,
            knowledge_retriever.get_topic_knowledge,
        ],
    }

    # 创建DeepAgent（Main Agent）
    agent = create_deep_agent(
        model=model,
        subagents=[macroeconomic_subagent],
    )

    return agent


if __name__ == '__main__':
    # 简单导入测试
    print("创建宏观经济分析Agent...")
    agent = create_macro_agent()
    print(f"[成功] Agent创建成功！类型: {type(agent).__name__}")

