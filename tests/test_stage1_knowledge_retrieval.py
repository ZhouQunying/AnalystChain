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

import json
import logging

from analyst_chain.tools.knowledge_retriever import KnowledgeRetriever
from analyst_chain.knowledge.constants import (
    VECTOR_DB_DIR,
    STRUCTURED_JSON_DIR,
    EMBEDDING_MODEL,
    Domain,
)
from analyst_chain.knowledge.schemas import KnowledgeJSON


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


def test_vector_search(test_queries: list = None):
    """测试向量检索"""
    print("=" * 80)
    print("[测试] 测试1：向量检索")
    print("=" * 80)

    if test_queries is None:
        test_queries = []

    for i, query in enumerate(test_queries, 1):
        print(f"\n查询{i}：{query}")
        print("-" * 80)
        result = _get_retriever().vector_search(query, k=2)
        print(result)


def test_topic_knowledge(test_topics: list = None):
    """测试主题知识查询"""
    if test_topics is None:
        test_topics = []

    print("=" * 80)
    print("[测试] 测试2：主题知识查询")
    print("=" * 80)

    for topic_num, name in test_topics:
        print(f"\n查询主题{topic_num}：{name}")
        print("-" * 80)
        result = _get_retriever().get_topic_knowledge(topic_num)
        print(result)


def test_keyword_search(test_keywords: list = None):
    """测试关键词搜索"""
    if test_keywords is None:
        test_keywords = []

    print("=" * 80)
    print("[测试] 测试3：关键词搜索")
    print("=" * 80)

    for keyword in test_keywords:
        print(f"\n关键词：{keyword}")
        print("-" * 80)
        result = _get_retriever().search_keyword(keyword)
        print(result)


def test_comprehensive(query: str = ""):
    """测试综合检索"""
    print("=" * 80)
    print("[测试] 测试4：综合检索")
    print("=" * 80)

    print(f"\n查询：{query}")
    print("-" * 80)
    result = _get_retriever().comprehensive_search(query)
    print(result)


def test_json_schema():
    """测试JSON结构正确性

    目的：验证所有生成的JSON文件结构符合KnowledgeJSON Schema
    验证内容：
    1. JSON文件可正常读取
    2. 数据结构符合Pydantic Schema定义
    """
    print("=" * 80)
    print("[测试] 测试5：JSON Schema 校验")
    print("=" * 80)

    json_dir = STRUCTURED_JSON_DIR / Domain.MACRO_ECONOMY
    json_files = list(json_dir.glob("*.json"))

    if not json_files:
        print(f"[错误] 无JSON文件：{json_dir}")
        return

    passed, failed = 0, 0
    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            KnowledgeJSON.model_validate(data)
            print(f"[通过] {json_file.name}")
            passed += 1
        except Exception as e:
            print(f"[失败] {json_file.name}：{e}")
            failed += 1

    print("-" * 80)
    print(f"[结果] 通过：{passed}，失败：{failed}")


def test_data_consistency(query: str = "", k: int = 100):
    """测试向量数据库与JSON数据一致性

    目的：验证向量库与JSON数据主题一致，确保Pipeline输出完整
    验证内容：
    1. JSON文件中的主题是否都在向量库中
    2. 向量库中的主题是否都有对应JSON文件
    """
    print("=" * 80)
    print("[测试] 测试6：数据一致性校验")
    print("=" * 80)

    json_dir = STRUCTURED_JSON_DIR / Domain.MACRO_ECONOMY
    json_files = list(json_dir.glob("*.json"))

    if not json_files:
        print(f"[错误] 无JSON文件：{json_dir}")
        return

    # 1. 从JSON获取所有topic
    json_topics = set()
    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            json_topics.add(json.load(f).get("topic", ""))

    # 2. 从向量库获取所有topic（通过metadata）
    vector_topics = set()
    results = _get_retriever().vector_search_raw(query=query, k=k)
    for doc, _ in results:
        vector_topic = doc.metadata.get("topic", "")
        if vector_topic:
            vector_topics.add(vector_topic)

    # 3. 比较
    print(f"JSON 主题数：{len(json_topics)}")
    print(f"向量库主题数：{len(vector_topics)}")

    only_in_json = json_topics - vector_topics
    only_in_vector = vector_topics - json_topics

    print("-" * 80)
    if only_in_json:
        print(f"[警告] 仅在JSON中：{only_in_json}")
    if only_in_vector:
        print(f"[警告] 仅在向量库中：{only_in_vector}")

    if not only_in_json and not only_in_vector:
        print("[通过] 数据一致")
    else:
        print("[失败] 数据不一致")


# 领域测试配置（按领域定义测试用例，便于扩展）
DOMAIN_TESTS = {
    Domain.MACRO_ECONOMY: {
        "test_queries": [
            "GDP增长率如何计算？",
            "投资时钟原理是什么？",
            "经济周期如何判断？",
            "CPI指数的含义",
            "PMI指标如何解读？"
        ],
        "test_topics": [
            (1, "中国经济的三驾马车"),
            (8, "经济周期"),
            (9, "投资时钟")
        ],
        "test_keywords": ["GDP", "CPI", "PMI"],
        "query": "当前经济周期如何判断，应该配置什么资产？",
        # k=100：确保覆盖所有主题（项目17个知识块，每个3-8个chunks）
        "consistency_params": {
            "query": "经济",
            "k": 100,
        },
    }
}


def main():
    """运行所有测试"""
    print("=" * 80)
    print("[测试] 知识库检索工具测试")
    print("=" * 80)

    # 获取当前领域的测试用例
    domain_config = DOMAIN_TESTS.get(Domain.MACRO_ECONOMY, {})
    test_queries = domain_config.get("test_queries", [])
    test_topics = domain_config.get("test_topics", [])
    test_keywords = domain_config.get("test_keywords", [])
    query = domain_config.get("query", "")
    consistency_params = domain_config.get("consistency_params", {})

    try:
        # 测试1：向量检索
        test_vector_search(test_queries)

        # 测试2：主题知识查询
        test_topic_knowledge(test_topics)

        # 测试3：关键词搜索
        test_keyword_search(test_keywords)

        # 测试4：综合检索
        test_comprehensive(query)

        # 测试5：JSON Schema校验
        test_json_schema()

        # 测试6：数据一致性校验
        test_data_consistency(**consistency_params)

        print("=" * 80)
        print("[完成] 所有测试完成")
        print("=" * 80)

    except Exception as e:
        logger.error(f"[测试] 测试失败：{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

