"""
宏观经济Agent测试

测试宏观经济分析SubAgent的功能和输出质量
"""

import sys
from pathlib import Path

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
# 加载环境变量
load_dotenv(Path(__file__).parent.parent / "config" / ".env")

from analyst_chain.agents import create_macro_agent


# 测试问题清单（按难度分级）
TEST_QUESTIONS = {
    "basic": [
        # 基础数据查询（简单）
        "2024年GDP增长率是多少？",
        "当前的通胀水平如何？",
        "最新的PMI数据是多少？",
    ],
    "intermediate": [
        # 周期判断（中等）
        "当前经济处于什么周期？",
        "经济周期转折的信号是什么？",
        "什么指标变化会预示周期转折？",
    ],
    "advanced": [
        # 投资策略（困难）
        "根据当前经济周期，应该配置什么资产？",
        "投资时钟当前处于哪个阶段？",
    ],
    "comprehensive": [
        # 综合分析（最难）
        "给出当前宏观经济的整体判断",
        "从宏观角度看，周期性行业投资机会如何？",
    ]
}


def test_agent_creation():
    """测试Agent创建"""
    print("\n" + "=" * 60)
    print("测试1: Agent创建")
    print("=" * 60)

    agent = create_macro_agent()
    assert agent is not None, "Agent创建失败"
    print(f"[成功] Agent创建成功！类型: {type(agent).__name__}")

    return agent


def test_basic_queries(agent):
    """测试基础查询（Token级流式输出）"""
    print("\n" + "=" * 60)
    print("测试2: 基础数据查询（Token级流式输出）")
    print("=" * 60)

    for i, question in enumerate(TEST_QUESTIONS["basic"], 1):
        print(f"\n问题 {i}/3: {question}")
        print("-" * 60)
        print("\n回答：", end="", flush=True)
        try:
            # 使用stream_mode="messages"实现token级流式输出
            for message_chunk, metadata in agent.stream(
                {"messages": [{"role": "user", "content": question}]},
                stream_mode="messages"
            ):
                if message_chunk.content:
                    print(message_chunk.content, end="", flush=True)
            print("\n")
        except Exception as e:
            print(f"\n[错误] 异常: {e}")


def test_intermediate_queries(agent):
    """测试周期判断（Token级流式输出）"""
    print("\n" + "=" * 60)
    print("测试3: 周期判断分析（Token级流式输出）")
    print("=" * 60)

    for i, question in enumerate(TEST_QUESTIONS["intermediate"], 1):
        print(f"\n问题 {i}/3: {question}")
        print("-" * 60)
        print("\n回答：", end="", flush=True)
        try:
            # 使用stream_mode="messages"实现token级流式输出
            for message_chunk, metadata in agent.stream(
                {"messages": [{"role": "user", "content": question}]},
                stream_mode="messages"
            ):
                if message_chunk.content:
                    print(message_chunk.content, end="", flush=True)
            print("\n")
        except Exception as e:
            print(f"\n[错误] 异常: {e}")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("宏观经济Agent测试套件")
    print("=" * 60)

    # 测试1：创建Agent
    agent = test_agent_creation()

    # 测试2：基础查询
    test_basic_queries(agent)

    # 测试3：周期判断
    test_intermediate_queries(agent)

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
    print("\n提示：取消注释agent.run()可执行实际查询测试")


if __name__ == '__main__':
    run_all_tests()

