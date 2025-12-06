"""
阶段1知识库快速验证脚本

目的: 30分钟快速验证阶段1的输出质量

验证内容:
1. JSON文件生成情况
2. 向量库可用性
3. 检索结果相关性
"""

import json
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def check_json_files():
    """检查JSON文件生成情况"""
    print("\n" + "="*80)
    print("1. 检查JSON文件生成")
    print("="*80)

    # 实际输出路径: ../output/structured_knowledge/knowledge_base/
    json_dir = Path("../output/structured_knowledge/knowledge_base")

    if not json_dir.exists():
        print(f"[警告] JSON目录不存在: {json_dir.absolute()}")
        return None, False

    json_files = list(json_dir.glob("*.json"))
    print(f"[成功] 生成了 {len(json_files)} 个JSON文件")

    if json_files:
        print("\n前5个文件:")
        for f in json_files[:5]:
            print(f"  - {f.name}")
        return json_files, True
    else:
        print("[错误] 没有找到JSON文件")
        return None, False


def check_json_structure(json_files):
    """检查JSON结构和内容质量"""
    print("\n" + "="*80)
    print("2. 检查JSON结构和内容")
    print("="*80)

    if not json_files:
        print("[跳过] 没有JSON文件")
        return False

    sample_file = json_files[0]
    print(f"\n检查文件: {sample_file.name}")
    print("-" * 80)

    with open(sample_file, 'r', encoding='utf-8') as f:
        sample = json.load(f)

    print("\nJSON结构:")
    for key in sample.keys():
        value = sample[key]
        if isinstance(value, list):
            print(f"  - {key}: {len(value)} 项")
        else:
            print(f"  - {key}: {type(value).__name__}")

    print("\n示例内容:")
    print(f"  主题: {sample.get('topic', 'N/A')}")

    if 'key_concepts' in sample:
        print("\n  关键概念(前3个):")
        for concept in sample['key_concepts'][:3]:
            print(f"    - {concept}")

    if 'summary' in sample:
        print(f"\n  摘要(前100字):")
        print(f"    {sample['summary'][:100]}...")

    has_topic = 'topic' in sample
    return has_topic


def check_vector_db():
    """检查向量库"""
    print("\n" + "="*80)
    print("3. 检查向量库")
    print("="*80)

    # 实际输出路径: ../output/vector_db/
    vector_db_dir = Path("../output/vector_db")

    if not vector_db_dir.exists():
        print(f"[警告] 向量库目录不存在: {vector_db_dir.absolute()}")
        return None, False

    print(f"[成功] 向量库目录: {vector_db_dir.absolute()}")
    print("\n子目录:")
    for subdir in vector_db_dir.iterdir():
        if subdir.is_dir():
            file_count = len(list(subdir.iterdir()))
            print(f"  - {subdir.name}: {file_count} 个文件")

    return vector_db_dir, True


def load_vector_store(vector_db_dir):
    """加载向量库"""
    print("\n" + "="*80)
    print("4. 加载向量库")
    print("="*80)

    if not vector_db_dir:
        print("[跳过] 向量库目录不存在")
        return None, False

    embeddings = HuggingFaceEmbeddings(
        model_name="Qwen/Qwen3-Embedding-0.6B"
    )

    # 实际向量库路径: ../output/vector_db/knowledge_base/
    vector_db_path = vector_db_dir / "knowledge_base"

    if not vector_db_path.exists():
        print(f"[错误] 向量库不存在: {vector_db_path}")
        return None, False

    try:
        # 使用正确的collection_name (与Pipeline一致)
        vector_store = Chroma(
            collection_name="knowledge_base_col",
            persist_directory=str(vector_db_path),
            embedding_function=embeddings
        )
        count = vector_store._collection.count()
        print(f"[成功] 向量库加载成功")
        print(f"  Collection: knowledge_base_col")
        print(f"  向量数量: {count}")
        return vector_store, True
    except Exception as e:
        print(f"[错误] 加载失败: {e}")
        return None, False


def test_queries(vector_store):
    """测试查询"""
    print("\n" + "="*80)
    print("5. 测试查询")
    print("="*80)

    if not vector_store:
        print("[跳过] 向量库未加载")
        return False

    test_queries_list = [
        "GDP增长率如何计算?",
        "中国经济三驾马车是什么?",
        "消费对经济的影响",
        "CPI指数的含义",
        "PMI指标如何解读?"
    ]

    all_success = True

    for i, query in enumerate(test_queries_list, 1):
        print(f"\n{i}. 查询: {query}")
        print("-" * 80)

        try:
            results = vector_store.similarity_search(query, k=2)

            if results:
                print(f"[成功] 返回 {len(results)} 个结果")
                print(f"\n  结果1(前150字):")
                print(f"    {results[0].page_content[:150]}...")

                if len(results) > 1:
                    print(f"\n  结果2(前150字):")
                    print(f"    {results[1].page_content[:150]}...")
            else:
                print("[警告] 没有返回结果")
                all_success = False
        except Exception as e:
            print(f"[错误] 查询失败: {e}")
            all_success = False

    return all_success


def print_conclusion(checklist):
    """打印验收结论"""
    print("\n" + "="*80)
    print("6. 验收结论")
    print("="*80)

    for item, status in checklist.items():
        status_icon = "[通过]" if status else "[失败]"
        print(f"{status_icon} {item}")

    passed = all(checklist.values())

    print("\n" + "="*80)
    if passed:
        print("[通过] 验收通过: 可以进入阶段2\n")
        print("建议:")
        print("1. 开始实现宏观分析SubAgent")
        print("2. 通过SubAgent实际使用来深入验证知识库质量")
        print("3. 发现问题及时记录,迭代优化")
    else:
        print("[失败] 验收未通过: 需要修复问题\n")
        print("请检查:")
        for item, status in checklist.items():
            if not status:
                print(f"  - {item}")


def main():
    """主函数"""
    print("\n" + "="*80)
    print("阶段1知识库快速验证")
    print("="*80)

    json_files, json_ok = check_json_files()
    json_structure_ok = check_json_structure(json_files) if json_files else False
    vector_db_dir, vector_db_ok = check_vector_db()
    vector_store, vector_loaded = load_vector_store(vector_db_dir) if vector_db_ok else (None, False)
    queries_ok = test_queries(vector_store) if vector_loaded else False

    checklist = {
        "JSON文件生成": json_ok,
        "JSON结构正确": json_structure_ok,
        "向量库存在": vector_db_ok,
        "向量库可加载": vector_loaded,
        "检索有结果": queries_ok,
    }

    print_conclusion(checklist)


if __name__ == "__main__":
    main()

