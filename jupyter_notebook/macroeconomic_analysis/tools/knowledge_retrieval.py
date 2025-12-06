"""
知识库检索工具

提供向量检索和JSON查询功能,支持DeepAgents Tools集成
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


class KnowledgeRetriever:
    """知识库检索工具

    提供两种检索方式:
    1. 向量检索: 语义相似度搜索
    2. JSON查询: 按主题精确查询结构化知识
    """

    def __init__(self,
                 vector_db_path: str = "../output/vector_db/knowledge_base",
                 json_dir_path: str = "../output/structured_knowledge/knowledge_base",
                 embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"):
        """初始化知识库检索器

        Args:
            vector_db_path: 向量数据库路径
            json_dir_path: JSON知识库路径
            embedding_model: Embedding模型名称
        """
        self.vector_db_path = Path(vector_db_path)
        self.json_dir_path = Path(json_dir_path)

        # 初始化Embedding
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

        # 加载向量库
        self.vector_store = Chroma(
            collection_name="knowledge_base_col",
            persist_directory=str(self.vector_db_path),
            embedding_function=self.embeddings
        )

        # 加载JSON索引
        self._load_json_index()

    def _load_json_index(self):
        """加载JSON文件索引"""
        self.json_files = {}
        for json_file in self.json_dir_path.glob("*.json"):
            # 提取主题编号
            topic_num = int(json_file.name.split("_")[0])
            self.json_files[topic_num] = json_file

    def vector_search(self, query: str, k: int = 3) -> str:
        """向量检索

        Args:
            query: 查询问题
            k: 返回结果数量

        Returns:
            格式化的检索结果字符串
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
                output += f"来源: 主题{doc.metadata.get('seq', 'N/A')} - {doc.metadata.get('topic', 'N/A')}\n"
            output += "\n"

        return output

    def get_topic_knowledge(self, topic_number: int) -> str:
        """按主题查询JSON知识

        Args:
            topic_number: 主题编号(1-17)

        Returns:
            格式化的主题知识字符串
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
                description = indicator.get('description', 'N/A')
                output += f"  - {name}: {description}\n"
            output += "\n"

        # 摘要
        if 'summary' in knowledge:
            output += f"摘要:\n{knowledge['summary'][:300]}...\n"

        return output

    def search_keyword(self, keyword: str) -> str:
        """关键词搜索

        在所有主题中搜索包含关键词的知识

        Args:
            keyword: 关键词(如"GDP"/"CPI")

        Returns:
            包含该关键词的主题列表
        """
        matched_topics = []

        for topic_num, json_file in self.json_files.items():
            with open(json_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if keyword.lower() in content.lower():
                    knowledge = json.loads(content)
                    matched_topics.append({
                        'number': topic_num,
                        'topic': knowledge.get('topic', 'N/A')
                    })

        if not matched_topics:
            return f"未找到包含'{keyword}'的主题"

        # 格式化输出
        output = f"关键词'{keyword}'相关主题(共{len(matched_topics)}个):\n\n"
        for item in matched_topics:
            output += f"- 主题{item['number']}: {item['topic']}\n"

        return output

    def comprehensive_search(self, query: str) -> str:
        """综合检索(向量+关键词)

        Args:
            query: 查询问题

        Returns:
            综合检索结果
        """
        output = "="*80 + "\n"
        output += f"查询: {query}\n"
        output += "="*80 + "\n\n"

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


# DeepAgents Tools封装
def create_knowledge_tools(retriever: Optional[KnowledgeRetriever] = None):
    """创建知识库检索工具集(供DeepAgents使用)

    Args:
        retriever: KnowledgeRetriever实例(可选,自动创建)

    Returns:
        工具函数列表
    """
    if retriever is None:
        retriever = KnowledgeRetriever()

    def vector_search_tool(query: str) -> str:
        """向量检索工具

        使用语义相似度搜索知识库

        Args:
            query: 查询问题

        Returns:
            相关知识内容
        """
        return retriever.vector_search(query, k=3)

    def topic_knowledge_tool(topic_number: int) -> str:
        """主题知识查询工具

        按主题编号查询结构化知识

        Args:
            topic_number: 主题编号(1-17)

        Returns:
            主题的关键概念、指标和摘要
        """
        return retriever.get_topic_knowledge(topic_number)

    def keyword_search_tool(keyword: str) -> str:
        """关键词搜索工具

        在所有主题中搜索关键词

        Args:
            keyword: 关键词

        Returns:
            包含该关键词的主题列表
        """
        return retriever.search_keyword(keyword)

    return [
        vector_search_tool,
        topic_knowledge_tool,
        keyword_search_tool
    ]

