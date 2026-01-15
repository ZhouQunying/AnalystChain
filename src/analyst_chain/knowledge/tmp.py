# src/analyst_chain/knowledge/schemas.py

"""知识库Schema定义与示例"""

from typing import TypedDict, List, NotRequired, Required


# ========== Schema定义 ==========

class KeyConcept(TypedDict):
    name: Required[str]
    definition: Required[str]
    importance: NotRequired[str]


class Indicator(TypedDict):
    name: Required[str]
    calculation: NotRequired[str]
    interpretation: Required[str]


class AnalysisMethod(TypedDict):
    name: Required[str]
    steps: Required[str]
    application: NotRequired[str]


class KnowledgeJSON(TypedDict):
    topic: Required[str]
    key_concepts: List[KeyConcept]
    indicators: List[Indicator]
    analysis_methods: List[AnalysisMethod]
    summary: Required[str]


# ========== 示例（给LLM看的好案例）==========

KNOWLEDGE_EXAMPLE: KnowledgeJSON = {
    'topic': '消费——快速入门读懂经济形势',
    'key_concepts': [
        {
            'name': '社会消费品零售总额',
            'definition': '反映一定时期内人民物质文化生活水平提高情况的重要指标',
            'importance': '是观察消费变化趋势的核心指标'
        }
    ],
    'indicators': [
        {
            'name': '社会消费品零售总额同比增速',
            'calculation': '当期值与去年同期值对比的增长率',
            'interpretation': '增速超过8%为较快增长，低于5%为偏弱'
        }
    ],
    'analysis_methods': [
        {
            'name': '消费三因素分析法',
            'steps': '分析人口数量、消费意愿、消费能力三个维度',
            'application': '用于判断消费增长的驱动因素和可持续性'
        }
    ],
    'summary': '消费是经济增长的主要驱动力。'
}


# ========== 工具函数 ==========

def create_empty_knowledge(topic: str) -> KnowledgeJSON:
    """创建空知识结构（LLM提取失败时使用）"""
    return {
        'topic': topic,
        'key_concepts': [],
        'indicators': [],
        'analysis_methods': [],
        'summary': '提取失败'
    }


def validate_knowledge_json(data: dict) -> tuple[bool, str]:
    """验证JSON结构（可选）"""
    if 'topic' not in data:
        return False, '缺少topic'
    if 'summary' not in data:
        return False, '缺少summary'
    return True, ''



# Stage1 Notebook（Prompt直接写在这里）

import json
from src.analyst_chain.knowledge.schemas import (
    KnowledgeJSON,
    KNOWLEDGE_EXAMPLE,
    create_empty_knowledge,
    validate_knowledge_json
)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser


class KnowledgeExtractor:
    """知识提取器"""

    def __init__(self, llm):
        self.llm = llm

        # Prompt直接定义在这里（不抽离）
        example_json = json.dumps(KNOWLEDGE_EXAMPLE, ensure_ascii=False, indent=2)

        self.prompt = ChatPromptTemplate.from_template(f'''你是金融知识提取专家。

文档：{{content}}

参考格式：
{example_json}

只返回JSON。''')

    def extract_from_documents(self, docs, topic: str) -> KnowledgeJSON:
        content = '\n\n'.join([d.page_content for d in docs[:5]])[:15000]

        try:
            chain = self.prompt | self.llm | JsonOutputParser()
            result = chain.invoke({'content': content})
            result['topic'] = topic

            valid, msg = validate_knowledge_json(result)
            if not valid:
                return create_empty_knowledge(topic)

            return result
        except Exception as e:
            return create_empty_knowledge(topic)