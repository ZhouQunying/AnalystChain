"""知识库Schema定义与示例"""

from typing import TypedDict, List


class KeyConcept(TypedDict):
    """关键概念（单个）

    name: 概念名
    definition: 定义
    importance: 重要性
    """
    name: str
    definition: str
    importance: str


class Indicator(TypedDict):
    """指标（单个）

    name: 指标名
    calculation: 计算方法
    interpretation: 解读
    """
    name: str
    calculation: str
    interpretation: str


class AnalysisMethod(TypedDict):
    """分析方法（单个）

    name: 方法名
    steps: 步骤
    application: 应用
    """
    name: str
    steps: str
    application: str


class KnowledgeJSON(TypedDict):
    """知识库JSON结构

    topic: 主题名称
    key_concepts: 关键概念列表
    indicators: 指标列表
    analysis_methods: 分析方法列表
    summary: 总结
    """
    topic: str
    key_concepts: List[KeyConcept]
    indicators: List[Indicator]
    analysis_methods: List[AnalysisMethod]
    summary: str


# 知识库schema示例，供Prompt参考
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
