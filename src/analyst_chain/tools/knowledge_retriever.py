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
    >>> retriever = KnowledgeRetriever()
    >>> result = retriever.vector_search("GDP增长率如何计算?", k=3)
    >>> result = retriever.get_topic_knowledge(1)  # 查询主题1
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 项目根目录（基于本文件位置推导：src/analyst_chain/tools/ -> 项目根目录）
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


class KnowledgeRetriever:
    """知识库检索工具

    提供两种检索方式:
    1. 向量检索: 语义相似度搜索（基于Chroma向量库）
    2. JSON查询: 按主题精确查询结构化知识

    使用场景：
    - Agent工具：供SubAgent调用，获取宏观经济知识
    - 测试验证：验证知识库是否可用、检索功能是否正常

    示例：
        >>> retriever = KnowledgeRetriever()
        >>> result = retriever.vector_search("GDP增长率如何计算?", k=3)
        >>> result = retriever.get_topic_knowledge(1)  # 查询主题1
    """

    def __init__(self,
                 vector_db_path: str = "data/processed/knowledge/vector_db/macro_economy",
                 json_dir_path: str = "data/processed/knowledge/structured/macro_economy",
                 embedding_model: Optional[str] = None):
        """初始化知识库检索器

        Args:
            vector_db_path: 向量数据库路径（相对于项目根目录）
            json_dir_path: JSON知识库路径（相对于项目根目录）
            embedding_model: Embedding模型名称或路径（可选，默认从环境变量读取）
        """
        # 转换为绝对路径
        self.vector_db_path = PROJECT_ROOT / vector_db_path
        self.json_dir_path = PROJECT_ROOT / json_dir_path

        # 获取Embedding模型（优先级：参数 > 环境变量 > 默认值）
        if embedding_model is None:
            embedding_model = os.getenv('EMBEDDING_MODEL_PATH', 'Qwen/Qwen3-Embedding-0.6B')

        # 初始化Embedding
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

        # 加载向量库
        self.vector_store = Chroma(
            collection_name="macro_economy_col",
            persist_directory=str(self.vector_db_path),
            embedding_function=self.embeddings
        )

        # 加载JSON索引
        self._load_json_index()

    def _load_json_index(self):
        """加载JSON文件索引"""
        self.json_files = {}
        if not self.json_dir_path.exists():
            return

        for json_file in self.json_dir_path.glob("*.json"):
            try:
                topic_num = int(json_file.name.split("_")[0])
                self.json_files[topic_num] = json_file
            except (ValueError, IndexError):
                continue

    def vector_search(self, query: str, k: int = 3) -> str:
        """向量检索

        基于Chroma向量库进行语义相似度搜索。

        Args:
            query: 查询问题
            k: 返回结果数量（默认3）

        Returns:
            格式化的检索结果字符串，格式：
            - 向量检索结果(共N条):
            - [结果1]
            - 内容: ...
            - 来源: 主题X - 主题名称
        """
        results = self.vector_store.similarity_search(query, k=k)

        if not results:
            return "未找到相关知识"

        # 格式化输出
        output = f"向量检索结果(共{len(results)}条):\n\n"
        for i, doc in enumerate(results, 1):
            output += f"[结果{i}]\n"
            output += f"内容: {doc.page_content[:200]}...\n"
            if doc.metadata:
                output += f"来源: 主题{doc.metadata.get('sequence', 'N/A')} - {doc.metadata.get('topic', 'N/A')}\n"
            output += "\n"

        return output

    def get_topic_knowledge(self, topic_number: int) -> str:
        """按主题查询JSON知识

        从结构化JSON知识库中按主题编号精确查询。

        Args:
            topic_number: 主题编号(1-17)

        Returns:
            格式化的主题知识字符串，包含：
            - 主题名称
            - 关键概念（前5个）
            - 关键指标（前3个）
            - 摘要（前300字符）
        """
        if topic_number not in self.json_files:
            return f"错误: 主题{topic_number}不存在(有效范围1-17)"

        json_file = self.json_files[topic_number]
        with open(json_file, 'r', encoding='utf-8') as f:
            knowledge = json.load(f)

        # 格式化输出
        output = f"主题{topic_number}: {knowledge.get('topic', 'N/A')}\n\n"

        # 关键概念
        if 'key_concepts' in knowledge:
            output += "关键概念:\n"
            for concept in knowledge['key_concepts'][:5]:  # 只显示前5个
                name = concept.get('name', 'N/A')
                definition = concept.get('definition', 'N/A')
                output += f"  - {name}: {definition}\n"
            output += "\n"

        # 指标
        if 'indicators' in knowledge:
            output += "关键指标:\n"
            for indicator in knowledge['indicators'][:3]:
                name = indicator.get('name', 'N/A')
                interpretation = indicator.get('interpretation', 'N/A')
                calculation = indicator.get('calculation', 'N/A')
                output += f"  - {name}: {interpretation} | {calculation} \n"
            output += "\n"

        # 摘要
        if 'summary' in knowledge:
            output += f"摘要:\n{knowledge['summary'][:300]}...\n"

        return output

    def search_keyword(self, keyword: str) -> str:
        """关键词搜索

        在所有主题中搜索包含关键词的知识（搜索主题名称和JSON内容）。

        Args:
            keyword: 关键词(如"GDP"/"CPI"/"PMI")

        Returns:
            格式化的匹配结果字符串，格式：
            - 关键词'XXX'匹配结果(共N个主题):
            - - 主题X: 主题名称
        """
        matches = []
        for topic_num, json_file in self.json_files.items():
            with open(json_file, 'r', encoding='utf-8') as f:
                knowledge = json.load(f)

            topic = knowledge.get('topic', '')
            content = json.dumps(knowledge, ensure_ascii=False)

            if keyword in topic or keyword in content:
                matches.append({
                    'number': topic_num,
                    'topic': topic
                })

        if not matches:
            return f"未找到包含关键词'{keyword}'的主题"

        # 格式化输出
        output = f"关键词'{keyword}'匹配结果(共{len(matches)}个主题):\n\n"
        for item in matches:
            output += f"- 主题{item['number']}: {item['topic']}\n"

        return output

    def comprehensive_search(self, query: str) -> str:
        """综合检索(向量+关键词)

        结合向量检索和关键词搜索，提供更全面的检索结果。

        Args:
            query: 查询问题

        Returns:
            格式化的综合检索结果字符串，包含：
            - 查询信息（分隔符格式）
            - 1. 语义检索结果（向量检索）
            - 2. 关键词匹配（如果查询中包含关键词）
        """
        output = "=" * 80 + "\n"
        output += f"查询: {query}\n"
        output += "=" * 80 + "\n\n"

        # 向量检索
        output += "1. 语义检索结果:\n"
        output += "-" * 80 + "\n"
        output += self.vector_search(query, k=2)
        output += "\n"

        # 提取关键词(简单实现:取查询中的名词)
        keywords = [w for w in ['GDP', 'CPI', 'PMI', '消费', '投资', '出口',
                                '经济周期', '投资时钟', '物价', '金融']
                   if w in query]

        if keywords:
            output += "2. 关键词匹配:\n"
            output += "-" * 80 + "\n"
            for kw in keywords:
                output += self.search_keyword(kw)
                output += "\n"

        return output

