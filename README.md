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
│   └── .env                        # 环境变量配置文件
├── jupyter_notebook/               # Jupyter Notebook 示例
│   ├── langchain/                  # LangChain 相关示例
│   │   ├── notebooks/              # Notebook 文件（按功能分类）
│   │   │   ├── agents/            # Agent 相关示例
│   │   │   ├── chains/            # Chain 相关示例
│   │   │   ├── graphs/            # LangGraph 相关示例
│   │   │   ├── integrations/      # 第三方集成示例
│   │   │   ├── rag/               # RAG 相关示例
│   │   │   ├── sql/               # SQL 相关示例
│   │   │   ├── storage/           # 向量存储相关示例
│   │   │   └── tools/             # 工具使用相关示例
│   │   ├── data/                  # 数据文件
│   │   │   ├── raw/               # 原始数据
│   │   │   ├── processed/         # 处理后的数据
│   │   │   ├── embeddings/       # 向量数据库
│   │   │   └── db/                # 数据库文件
│   │   ├── assets/                # 资源文件（PDF、图片等）
│   │   └── outputs/               # 输出文件（报告、结果等）
│   └── macroeconomic_analysis/    # 宏观经济分析模块
│       └── knowledge_base/         # 知识库（百度网盘导出的文档）
├── manus/                          # 其他分析模块
│   └── gdp/                       # GDP 分析模块
│       ├── scripts/               # Python 脚本
│       ├── data/                  # 数据文件
│       ├── images/                # 图片文件
│       ├── reports/               # 报告文件
│       └── docs/                  # 文档文件
├── src/                           # 源代码
│   ├── langchain_app/            # LangChain 应用代码
│   │   ├── agents/               # Agent 实现
│   │   ├── chains/               # Chain 实现
│   │   ├── tools/                # 工具实现
│   │   ├── prompts/              # Prompt 模板
│   │   └── utils/                # 工具函数
│   └── tests/                    # 测试代码
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
2. 配置 API 密钥到 `config/.env` 文件
3. 运行 Jupyter Notebook 查看示例
4. 使用 manus/gdp 模块进行经济数据分析
