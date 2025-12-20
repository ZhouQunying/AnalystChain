# AnalystChain 开发指南

## 项目概述

本项目是一个基于 LangChain 框架的开发环境，集成了多种 AI 工具和数据处理功能。项目主要包括经济数据分析模块和 Jupyter Notebook 示例。

## 环境设置

本项目由 conda 管理 Python 环境，由 uv 管理 Python 包。

### 1. 使用 conda 初始化环境

```bash
conda env create -f environment.yml
conda activate analyst_chain
```

### 2. 使用 uv 安装依赖

```bash
uv pip install -r requirements.txt
uv pip install new-package
```

## 项目结构

```
.
├── config/                          # 配置文件目录
│   └── .env.example               # 环境变量配置模板
├── notebooks/                      # Jupyter Notebook 工作区
│   ├── stage1_macro_knowledge_pipeline.ipynb  # 知识库构建流程
│   ├── stage2_deep_agent.ipynb    # SubAgent 原型
│   └── learning/                  # 学习笔记
│       └── akshare_tools_outline.ipynb
├── src/                           # 源代码
│   ├── analyst_chain/            # AnalystChain 核心代码
│   │   ├── agents/               # Agent 实现（Macro Agent等）
│   │   ├── tools/                # 工具实现（AKShare、知识检索等）
│   │   └── utils/                # 工具函数
├── tests/                         # 测试代码
├── docs/                          # 项目文档
│   ├── tasks/                    # 任务文档
│   ├── learning/                 # 学习笔记
│   └── AI行为约束规范.md         # AI编码规范
├── data/                          # 数据目录
│   ├── raw/                      # 原始数据
│   ├── processed/                # 处理后的数据
│   └── outputs/                  # 输出结果
├── _archive/                      # 归档目录
│   ├── manus/                    # 早期实验脚本（GDP分析等）
│   └── notebooks/langchain/      # LangChain学习示例
├── environment.yml                # Conda 环境配置
├── requirements.txt               # Python 依赖
└── README.md                      # 项目说明文档
```

## 配置说明

在 `config/.env` 文件中配置 API 密钥：

```env
# DeepSeek API 密钥
DEEPSEEK_API_KEY=your_deepseek_api_key_here

# Serper API 密钥
SERPER_API_KEY=your_serper_api_key_here
```

## 使用说明

1. 完成环境设置
2. 复制 `config/.env.example` 为 `config/.env` 并配置 API 密钥
3. 运行 Jupyter Notebook 查看阶段任务：
   - `notebooks/stage1_macro_knowledge_pipeline.ipynb` - 知识库构建
   - `notebooks/stage2_deep_agent.ipynb` - SubAgent 原型开发
4. 查看 `docs/tasks/` 了解任务进度和详细说明

## 开发指南

**快速卡**：`docs/开发手册.md`（1页纸）

```bash
# 检查Notebook语法
python scripts/check_notebook_syntax.py <file>

# 运行测试（改什么测什么）
python tests/test_*.py

# 开发模式安装
pip install -e .
```
