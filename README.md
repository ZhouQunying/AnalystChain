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
├── config/                 # 配置文件目录
│   └── .env               # 环境变量配置文件
├── jupter_notebook/       # Jupyter Notebook 示例
│   ├── deepseek.ipynb     # DeepSeek API 使用示例
│   └── google_serper.ipynb # Google Serper API 使用示例
├── manus_gdp/             # 经济数据分析模块
└── README.md              # 项目说明文档
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
4. 使用 manus_gdp 模块进行经济数据分析