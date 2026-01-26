"""
知识库检索工具

提供向量检索和JSON查询功能，支持DeepAgents Tools集成。

使用场景：
- Agent工具：供SubAgent调用，获取宏观经济知识
- 测试验证：验证知识库是否可用、检索功能是否正常

主要功能：
1. 向量检索：语义相似度搜索（基于Chroma向量库）
2. 主题知识查询：按主题编号精确查询结构化JSON知识
3. 关键词搜索：在所有主题中搜索包含关键词的知识
4. 综合检索：向量检索+关键词搜索组合

示例：
    >>> from analyst_chain.knowledge.constants import (
    ...     Domain,
    ...     STRUCTURED_JSON_DIR,
    ...     VECTOR_DB_DIR,
    ...     EMBEDDING_MODEL
    ... )
    >>> from analyst_chain.tools.knowledge_retriever import KnowledgeRetriever
    >>>
    >>> # 初始化检索器
    >>> retriever = KnowledgeRetriever(
    ...     domain=Domain.MACRO_ECONOMY,
    ...     structured_json_dir=STRUCTURED_JSON_DIR,
    ...     vector_db_dir=VECTOR_DB_DIR,
    ...     embedding_model=EMBEDDING_MODEL
    ... )
    >>>
    >>> # 向量检索（文本格式，供Agent/LLM使用）
    >>> result = retriever.vector_search("GDP增长率如何计算？", k=3)
    >>>
    >>> # 向量检索（原始数据，供程序化处理）
    >>> raw_results = retriever.vector_search_raw("GDP增长率如何计算？", k=3)
    >>> for doc, score in raw_results:
    ...     print(f"相似度：{score}，内容：{doc.page_content[:50]}")
    >>>
    >>> # 主题查询
    >>> result = retriever.get_topic_knowledge(1)
"""

import json
import logging
from pathlib import Path
from langchain_chroma import Chroma
from ..utils.embeddings_utils import get_embeddings
from ..knowledge.constants import VectorMetadataKeys
from ..knowledge.schemas import KnowledgeJSON

logger = logging.getLogger(__name__)

# 常量配置
CONTENT_PREVIEW_LENGTH = 150  # 预览模式截断长度（字符数，约75字）
DEFAULT_TOP_K = 3  # 检索返回数量


class KnowledgeRetriever:
    """知识库检索工具

    提供两种检索方式：
    1. 向量检索：语义相似度搜索（基于Chroma向量库）
    2. JSON查询：按主题精确查询结构化知识

    使用场景：
    - Agent工具：供SubAgent调用，获取宏观经济知识
    - 测试验证：验证知识库是否可用、检索功能是否正常

    示例：
        >>> from analyst_chain.knowledge.constants import (
        ...     Domain, STRUCTURED_JSON_DIR, VECTOR_DB_DIR, EMBEDDING_MODEL
        ... )
        >>> retriever = KnowledgeRetriever(
        ...     domain=Domain.MACRO_ECONOMY,
        ...     structured_json_dir=STRUCTURED_JSON_DIR,
        ...     vector_db_dir=VECTOR_DB_DIR,
        ...     embedding_model=EMBEDDING_MODEL
        ... )
        >>> result = retriever.vector_search("GDP增长率如何计算？", k=3)
        >>> result = retriever.get_topic_knowledge(1)
    """

    def __init__(self,
                 domain: str,
                 structured_json_dir: Path,
                 vector_db_dir: Path,
                 embedding_model: str):
        """初始化知识库检索器

        Args:
            domain: 领域名称
            structured_json_dir: JSON知识库路径
            vector_db_dir: 向量数据库路径
            embedding_model: Embedding模型名称或路径
        """
        # 领域路径
        self.domain_json_dir = structured_json_dir / domain
        self.domain_vector_dir = vector_db_dir / domain

        # 初始化Embedding
        self.embeddings = get_embeddings(model_name=embedding_model)

        # 加载向量库
        self.vector_store = Chroma(
            collection_name="knowledge",
            persist_directory=str(self.domain_vector_dir),
            embedding_function=self.embeddings
        )

        # 加载JSON索引
        self.json_files = {}
        self._load_json_index()

    def _load_json_index(self):
        """加载JSON文件索引"""
        if not self.domain_json_dir.exists():
            return

        for json_file in self.domain_json_dir.glob("*.json"):
            try:
                topic_num = int(json_file.name.split("_")[0])
                self.json_files[topic_num] = json_file
            except (ValueError, IndexError) as e:
                logger.warning(f"跳过无效JSON文件：{json_file.name}（格式错误：{type(e).__name__}）")
                continue

    def vector_search(self, query: str, k: int = DEFAULT_TOP_K, preview_only: bool = False) -> str:
        """向量检索（Agent调用接口）

        基于Chroma向量库进行语义相似度搜索，返回格式化文本供LLM理解。

        Args:
            query: 查询问题
            k: 返回结果数量（默认3）
            preview_only: 是否只返回预览（默认False返回完整内容）
                - False: 返回完整chunk内容（约800字符/条，供LLM深度理解）
                - True: 返回150字符预览（节省token，快速浏览）

        Returns:
            格式化的检索结果字符串，包含：
            - 结果数量
            - 每个结果的相似度、来源、内容

        Example:
            >>> result = retriever.vector_search("GDP增长率如何计算？")
            >>> print(result)
            向量检索结果（共3条）：

            [结果1]
            相似度：0.856
            来源：主题1 - 中国经济的三驾马车
            内容：GDP是国内生产总值...
            ...
        """
        results = self.vector_search_raw(query, k)
        return self._format_for_llm(results, preview_only=preview_only)

    def vector_search_raw(self, query: str, k: int = DEFAULT_TOP_K) -> list:
        """向量检索（原始数据接口）

        返回原始Document对象和相似度得分，供程序化处理。

        Args:
            query: 查询问题
            k: 返回结果数量（默认3）

        Returns:
            List[Tuple[Document, float]]: (文档对象, 相似度得分) 元组列表

        Example:
            >>> results = retriever.vector_search_raw("GDP", k=1)
            >>> doc, score = results[0]
            >>> print(doc.page_content)
        """
        return self.vector_store.similarity_search_with_score(query, k=k)

    def _format_for_llm(self, results: list, preview_only: bool = False) -> str:
        """格式化向量检索结果供LLM使用

        统一的格式化逻辑，确保输出格式一致、LLM友好。

        Args:
            results: List[Tuple[Document, float]] - (文档, 相似度) 元组列表
            preview_only: 是否只返回预览（True截断150字符，False返回完整内容）

        Returns:
            格式化的文本字符串

        Example:
            完整模式（preview_only=False）::

                向量检索结果（共2条）：

                [结果1]
                相似度：0.856
                来源：主题1 - 中国经济的三驾马车
                内容：GDP是国内生产总值，由消费、投资、净出口三部分组成...（完整chunk内容，约800字符）

            预览模式（preview_only=True）::

                向量检索结果（共2条）：

                [结果1]
                相似度：0.856
                来源：主题1 - 中国经济的三驾马车
                内容：GDP是国内生产总值，由消费、投资、净出口三部分组成...（150字符预览）
        """
        if not results:
            return "未找到相关知识"

        output = f"向量检索结果（共{len(results)}条）：\n\n"
        for i, (doc, score) in enumerate(results, 1):
            output += f"[结果{i}]\n"
            output += f"相似度：{score:.3f}\n"
            # 先显示来源，再显示内容（LLM更容易理解上下文）
            if doc.metadata:
                seq = doc.metadata.get(VectorMetadataKeys.SEQUENCE, "N/A")
                topic = doc.metadata.get(VectorMetadataKeys.TOPIC, "N/A")
                output += f"来源：主题{seq} - {topic}\n"
            # 根据模式选择内容长度
            if preview_only:
                output += f"内容：{doc.page_content[:CONTENT_PREVIEW_LENGTH]}...\n"
            else:
                output += f"内容：{doc.page_content}\n"
            output += "\n"

        return output

    def get_topic_knowledge(self, topic_number: int) -> str:
        """按主题查询JSON知识

        从结构化JSON知识库中按主题编号精确查询。

        Args:
            topic_number: 主题编号

        Returns:
            格式化的主题知识字符串，包含：
            - 主题名称
            - 关键概念
            - 关键指标
            - 摘要

        Example:
            >>> result = retriever.get_topic_knowledge(1)
            >>> print(result)
            主题1：中国经济的三驾马车

            关键概念：
            - 三驾马车：消费、投资、净出口
            - GDP：国内生产总值
            ...

            关键指标：
            - GDP增长率：衡量经济增速
            ...

            摘要：
            三驾马车是拉动经济增长的核心力量...
        """
        if topic_number not in self.json_files:
            valid_range = sorted(self.json_files.keys())
            return f"错误：主题{topic_number}不存在（有效范围：{min(valid_range)}-{max(valid_range)}）"

        json_file = self.json_files[topic_number]
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            knowledge = KnowledgeJSON.model_validate(data)

        # 格式化输出
        output = f"主题{topic_number}：{knowledge.topic}\n\n"

        # 关键概念
        if knowledge.key_concepts:
            output += "关键概念：\n"
            for concept in knowledge.key_concepts:
                output += f"  - {concept.name}：{concept.definition}\n"
            output += "\n"

        # 指标
        if knowledge.indicators:
            output += "关键指标：\n"
            for indicator in knowledge.indicators:
                output += f"  - {indicator.name}：{indicator.interpretation} | {indicator.calculation}\n"
            output += "\n"

        # 摘要
        if knowledge.summary:
            output += f"摘要：\n{knowledge.summary}\n"

        return output

    def search_keyword(self, keyword: str) -> str:
        """关键词搜索

        在所有主题中搜索包含关键词的知识（搜索主题名称和JSON内容）。

        Args:
            keyword: 关键词（如"GDP"/"CPI"/"PMI"）

        Returns:
            格式化的匹配结果字符串，格式：
            - 关键词"XXX"匹配结果（共N个主题）：
            - - 主题X：主题名称

        Example:
            >>> result = retriever.search_keyword("GDP")
            >>> print(result)
            关键词'GDP'匹配结果（共3个主题）：

            - 主题1：中国经济的三驾马车
            - 主题5：PMI——快速入门读懂经济形势
            - 主题10：股市投资手册
        """
        matches = []
        for topic_num, json_file in self.json_files.items():
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                knowledge = KnowledgeJSON.model_validate(data)

            topic = knowledge.topic
            content = json.dumps(knowledge.model_dump(), ensure_ascii=False)

            if keyword in topic or keyword in content:
                matches.append({
                    "number": topic_num,
                    "topic": topic
                })

        if not matches:
            return f"未找到包含关键词'{keyword}'的主题"

        # 格式化输出
        output = f"关键词'{keyword}'匹配结果（共{len(matches)}个主题）：\n\n"
        for item in matches:
            output += f"- 主题{item['number']}：{item['topic']}\n"

        return output

    def comprehensive_search(self, query: str) -> str:
        """综合检索（向量+关键词）

        结合向量检索和关键词搜索，提供更全面的检索结果。

        Args:
            query: 查询问题

        Returns:
            格式化的综合检索结果字符串，包含：
            - 查询信息（分隔符格式）
            - 1. 语义检索结果（向量检索）
            - 2. 关键词匹配（如果查询中包含关键词）

        Example:
            >>> result = retriever.comprehensive_search("GDP增长率和CPI有什么关系？")
            >>> print(result)
            [综合检索]GDP增长率和CPI有什么关系？

            [语义检索]
            向量检索结果（共3条）：

            [结果1]
            相似度：0.856
            来源：主题1 - 中国经济的三驾马车
            内容：GDP是国内生产总值...
            ...

            [关键词匹配]
            关键词'GDP'匹配结果（共3个主题）：

            - 主题1：中国经济的三驾马车
            ...

            关键词'CPI'匹配结果（共2个主题）：

            - 主题7：物价——快速入门读懂经济形势
            ...
        """
        output = f"[综合检索]{query}\n\n"

        # 向量检索
        output += "[语义检索]\n"
        output += self.vector_search(query, k=DEFAULT_TOP_K)
        output += "\n"

        # 提取关键词（简单实现：取查询中的名词）
        keywords = [w for w in ["GDP", "CPI", "PMI", "消费", "投资", "出口",
                                "经济周期", "投资时钟", "物价", "金融"]
                   if w in query]

        if keywords:
            output += "[关键词匹配]\n"
            for kw in keywords:
                output += self.search_keyword(kw)
                output += "\n"

        return output

