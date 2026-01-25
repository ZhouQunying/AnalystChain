"""知识库Schema定义与示例"""

from pydantic import BaseModel
from typing import List, Optional


class KeyConcept(BaseModel):
    """关键概念（单个）

    name: 概念名
    definition: 定义
    importance: 重要性
    """
    name: str
    definition: str
    importance: str


class Indicator(BaseModel):
    """指标（单个）

    name: 指标名
    calculation: 计算方法（可选，概念性指标可无）
    definition: 定义（可选，与calculation二选一）
    interpretation: 解读
    """
    name: str
    calculation: Optional[str] = None
    definition: Optional[str] = None
    interpretation: str


class AnalysisMethod(BaseModel):
    """分析方法（单个）

    name: 方法名
    steps: 步骤
    application: 应用
    """
    name: str
    steps: str
    application: str


class KnowledgeJSON(BaseModel):
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
# 设计原则：每个列表至少2个元素，展示期望的深度和丰富度
EXAMPLE_KNOWLEDGE = KnowledgeJSON(
    topic="01第一节 中国经济的三驾马车",
    key_concepts=[
        KeyConcept(
            name="三驾马车",
            definition="GDP由消费、投资、净出口三部分构成，公式：GDP = C + I + NX",
            importance="理解经济增长的核心框架"
        ),
        KeyConcept(
            name="贡献率",
            definition="某分项对GDP增长的贡献占比",
            importance="判断经济增长动力来源"
        ),
    ],
    indicators=[
        Indicator(
            name="GDP增速",
            calculation="（本期GDP - 上期GDP）/ 上期GDP × 100%",
            interpretation="正增长表示经济扩张，负增长表示收缩"
        ),
        Indicator(
            name="消费占比",
            calculation="最终消费支出 / GDP × 100%",
            interpretation="占比越高说明内需驱动型经济"
        ),
    ],
    analysis_methods=[
        AnalysisMethod(
            name="三驾马车分析法",
            steps="1. 看各分项占比；2. 看贡献率变化；3. 对比历史判断趋势",
            application="判断经济发展阶段和结构优化方向"
        ),
    ],
    summary="三驾马车是分析宏观经济的基础框架，通过观察消费、投资、净出口的占比和贡献率变化，可判断经济结构和增长动力。"
)
