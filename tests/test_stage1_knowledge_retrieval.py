"""
知识库检索工具测试脚本

测试对象：KnowledgeRetriever（知识库检索工具）

测试功能：
1. 向量检索（语义相似度搜索）
2. 主题知识查询（按主题精确查询）
3. 关键词搜索（关键词匹配）
4. 综合检索（向量+JSON组合检索）

测试目的：验证知识库是否可用、检索功能是否正常
"""

import sys
import logging
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from analyst_chain.tools.knowledge_retriever import KnowledgeRetriever

logger = logging.getLogger(__name__)


def test_vector_search():
    """测试向量检索"""
    print("=" * 80)
    print("[测试] 测试1: 向量检索")
    print("=" * 80)

    retriever = KnowledgeRetriever()

    test_queries = [
        "GDP增长率如何计算?",
        "投资时钟原理是什么?",
        "经济周期如何判断?",
        "CPI指数的含义",
        "PMI指标如何解读?"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n查询{i}: {query}")
        print("-" * 80)
        result = retriever.vector_search(query, k=2)
        print(result)


def test_topic_knowledge():
    """测试主题知识查询"""
    print("=" * 80)
    print("[测试] 测试2: 主题知识查询")
    print("=" * 80)

    retriever = KnowledgeRetriever()

    test_topics = [
        (1, "中国经济的三驾马车"),
        (8, "经济周期"),
        (9, "投资时钟")
    ]

    for topic_num, name in test_topics:
        print(f"\n查询主题{topic_num}: {name}")
        print("-" * 80)
        result = retriever.get_topic_knowledge(topic_num)
        print(result)


def test_keyword_search():
    """测试关键词搜索"""
    print("=" * 80)
    print("[测试] 测试3: 关键词搜索")
    print("=" * 80)

    retriever = KnowledgeRetriever()

    test_keywords = ["GDP", "CPI", "PMI"]

    for keyword in test_keywords:
        print(f"\n关键词: {keyword}")
        print("-" * 80)
        result = retriever.search_keyword(keyword)
        print(result)


def test_comprehensive():
    """测试综合检索"""
    print("=" * 80)
    print("[测试] 测试4: 综合检索")
    print("=" * 80)

    retriever = KnowledgeRetriever()

    query = "当前经济周期如何判断,应该配置什么资产?"
    print(f"\n查询: {query}")
    print("-" * 80)
    result = retriever.comprehensive_search(query)
    print(result)


def main():
    """运行所有测试"""
    print("=" * 80)
    print("[测试] 知识库检索工具测试")
    print("=" * 80)

    try:
        # 测试1: 向量检索
        test_vector_search()

        # 测试2: 主题知识查询
        test_topic_knowledge()

        # 测试3: 关键词搜索
        test_keyword_search()

        # 测试4: 综合检索
        test_comprehensive()

        print("=" * 80)
        print("[完成] 所有测试完成")
        print("=" * 80)

    except Exception as e:
        logger.error(f"[测试] 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

