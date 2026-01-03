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

### 3. 开发模式安装（可选）

如果需要运行测试或导入项目代码：

```bash
pip install -e .
```

**说明**：
- 此命令将`src/analyst_chain`安装为可导入的包
- **用途**：供`tests/`测试脚本使用
- **注意**：Notebook开发不需要此步骤（规范禁止notebook导入src/）

## 项目结构

```
.
├── _archive/                        # 归档目录
│   ├── experimental_scripts/       # 早期实验脚本
│   ├── langchain_learning/         # LangChain学习笔记
│   └── langchain_method_cheatsheet.md
├── config/                          # 配置文件目录
│   └── .env.example               # 环境变量配置模板
├── data/                            # 数据目录
│   ├── raw/                        # 原始数据（知识文件）
│   ├── processed/                  # 处理后的数据（知识库）
│   └── outputs/                    # 输出结果
├── docs/                            # 项目文档
│   ├── tasks/                      # 任务文档
│   └── AI工作规范.md               # AI编码规范
├── logs/                            # 日志文件
│   ├── agent_runs/                 # Agent运行日志
│   ├── errors/                     # 错误日志
│   └── pipeline/                   # Pipeline日志
├── notebooks/                       # Jupyter Notebook工作区
│   ├── stage1_macro_knowledge_pipeline.ipynb
│   └── stage2_macro_agent.ipynb
├── scripts/                         # 脚本工具
│   ├── check_notebook_syntax.py
│   └── README.md
├── src/                             # 源代码
│   └── analyst_chain/
│       ├── tools/                  # 工具实现
│       └── utils/                  # 工具函数
├── tests/                           # 测试代码
│   ├── test_knowledge_retrieval.py
│   └── README.md
├── .cursorrules                     # AI行为规则（AI开发必读）
├── .gitignore                       # Git忽略规则
├── environment.yml                  # Conda环境配置
├── README.md                        # 项目说明文档
├── requirements.txt                 # Python依赖
└── setup.py                         # 项目安装配置
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
   - `notebooks/stage2_macro_agent.ipynb` - SubAgent 原型（宏观经济）
4. 查看 `docs/tasks/` 了解任务进度和详细说明

## 开发指南

```bash
# 检查Notebook语法
python scripts/check_notebook_syntax.py <file>

# 运行测试（改什么测什么）
python tests/test_*.py

# 开发模式安装
pip install -e .
```

**规范**：[AI工作规范](docs/AI工作规范.md)
