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

import logging

from analyst_chain.tools.knowledge_retriever import KnowledgeRetriever
from analyst_chain.knowledge.constants import (
    VECTOR_DB_DIR,
    STRUCTURED_JSON_DIR,
    EMBEDDING_MODEL,
    Domain,
)


logger = logging.getLogger(__name__)

# 延迟初始化，避免导入时加载向量库
retriever = None


def _get_retriever():
    """获取检索器实例（延迟初始化）"""
    global retriever
    if retriever is None:
        retriever = KnowledgeRetriever(
            domain=Domain.MACRO_ECONOMY,
            structured_json_dir=STRUCTURED_JSON_DIR,
            vector_db_dir=VECTOR_DB_DIR,
            embedding_model=EMBEDDING_MODEL
        )
    return retriever


def test_vector_search():
    """测试向量检索"""
    print("=" * 80)
    print("[测试] 测试1: 向量检索")
    print("=" * 80)

    test_queries = [
        "GDP增长率如何计算？",
        "投资时钟原理是什么？",
        "经济周期如何判断？",
        "CPI指数的含义",
        "PMI指标如何解读？"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n查询{i}: {query}")
        print("-" * 80)
        result = _get_retriever().vector_search(query, k=2)
        print(result)


def test_topic_knowledge():
    """测试主题知识查询"""
    print("=" * 80)
    print("[测试] 测试2: 主题知识查询")
    print("=" * 80)

    test_topics = [
        (1, "中国经济的三驾马车"),
        (8, "经济周期"),
        (9, "投资时钟")
    ]

    for topic_num, name in test_topics:
        print(f"\n查询主题{topic_num}: {name}")
        print("-" * 80)
        result = _get_retriever().get_topic_knowledge(topic_num)
        print(result)


def test_keyword_search():
    """测试关键词搜索"""
    print("=" * 80)
    print("[测试] 测试3: 关键词搜索")
    print("=" * 80)

    test_keywords = ["GDP", "CPI", "PMI"]

    for keyword in test_keywords:
        print(f"\n关键词: {keyword}")
        print("-" * 80)
        result = _get_retriever().search_keyword(keyword)
        print(result)


def test_comprehensive():
    """测试综合检索"""
    print("=" * 80)
    print("[测试] 测试4: 综合检索")
    print("=" * 80)

    query = "当前经济周期如何判断，应该配置什么资产？"
    print(f"\n查询: {query}")
    print("-" * 80)
    result = _get_retriever().comprehensive_search(query)
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
        logger.error(f"[测试] 测试失败：{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

