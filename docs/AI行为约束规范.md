# AI行为约束规范 - 代码示例参考

> **定位**: 代码示例参考文档（好的 vs 坏的）
>
> **核心规则**: 见 `.cursorrules`（自动加载，100%可靠）
>
> **本文档作用**: 提供具体代码示例，作为编码时的参考（不强制要求AI读取）

---

## 📚 Python代码示例

### 1. 类型提示

```python
# ❌ 没有类型提示
def get_data(query, limit):
    return []

# ✅ 完整类型提示
from typing import List, Dict, Any

def get_data(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """获取数据"""
    return []
```

---

### 2. 文档字符串

```python
# ❌ 没有文档
def process(data):
    if not data:
        raise ValueError("empty")
    return {"result": data.upper()}

# ✅ 完整文档字符串
def process(data: str) -> Dict[str, str]:
    """处理输入数据并返回结果。

    Args:
        data: 输入数据字符串，不能为空

    Returns:
        处理结果字典，包含'result'字段

    Raises:
        ValueError: 当data为空时

    Example:
        >>> result = process("test")
        >>> result['result']
        'TEST'
    """
    if not data:
        raise ValueError("data不能为空")
    return {"result": data.upper()}
```

---

### 3. 异常处理

```python
# ❌ 忽略异常
def fetch_gdp_data():
    data = akshare.macro_china_gdp()  # 可能网络失败
    return data

# ✅ 适当处理
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def fetch_gdp_data() -> Optional[pd.DataFrame]:
    """获取GDP数据，失败时返回None并记录日志。

    Returns:
        GDP数据DataFrame，失败时返回None
    """
    try:
        return akshare.macro_china_gdp()
    except Exception as e:
        logger.error(f"获取GDP数据失败: {e}")
        return None
```

---

### 4. 配置管理

```python
# ❌ 硬编码配置
class Agent:
    def __init__(self):
        self.api_key = "sk-xxxxx"  # 硬编码
        self.model = "gpt-4"
        self.temperature = 0.7

# ✅ 环境变量配置
import os
from dataclasses import dataclass

@dataclass
class AgentConfig:
    """Agent配置（从环境变量读取）"""
    api_key: str = os.getenv("API_KEY", "")
    model: str = os.getenv("MODEL_NAME", "gpt-4")
    temperature: float = float(os.getenv("TEMPERATURE", "0.7"))

class Agent:
    def __init__(self, config: AgentConfig):
        self.config = config
```

---

## 🏗️ 架构设计示例

### 单一职责原则

```python
# ❌ 违反单一职责（一个类做太多事）
class DataManager:
    def get_gdp(self) -> pd.DataFrame:
        """获取GDP数据"""
        return akshare.macro_china_gdp()

    def save_to_db(self, data: pd.DataFrame) -> None:
        """保存到数据库"""
        db.insert(data)

    def send_email(self, subject: str, body: str) -> None:
        """发送邮件通知"""
        email.send(subject, body)

# ✅ 遵循单一职责（每个类只负责一件事）
class DataFetcher:
    """数据获取器"""
    def get_gdp(self) -> pd.DataFrame:
        return akshare.macro_china_gdp()

class DataStorage:
    """数据存储器"""
    def save(self, data: pd.DataFrame) -> None:
        db.insert(data)

class NotificationService:
    """通知服务"""
    def send_email(self, subject: str, body: str) -> None:
        email.send(subject, body)
```

---

### 松耦合（依赖接口而非实现）

```python
# ❌ 紧耦合（依赖具体实现）
class MacroAgent:
    def __init__(self):
        self.fetcher = AKShareDataFetcher()  # 依赖具体类

    def analyze(self):
        data = self.fetcher.get_gdp()  # 无法替换数据源
        return self._process(data)

# ✅ 松耦合（依赖接口）
from abc import ABC, abstractmethod

class DataFetcher(ABC):
    """数据获取器接口"""
    @abstractmethod
    def get_gdp(self) -> pd.DataFrame:
        pass

class AKShareDataFetcher(DataFetcher):
    """AKShare实现"""
    def get_gdp(self) -> pd.DataFrame:
        return akshare.macro_china_gdp()

class MacroAgent:
    def __init__(self, fetcher: DataFetcher):  # 依赖接口
        self.fetcher = fetcher

    def analyze(self):
        data = self.fetcher.get_gdp()  # 可替换任意实现
        return self._process(data)

# 使用时可以轻松替换实现
agent = MacroAgent(AKShareDataFetcher())
# 或
agent = MacroAgent(MockDataFetcher())  # 测试时用Mock
```

---

## 🎯 Prompt工程示例

### 结构化Prompt模板

```python
# ❌ 糟糕的Prompt（指令不清晰）
prompt = "提取知识"

# ✅ 优秀的Prompt（结构化）
prompt_template = """
【角色定义】
你是经济学知识提取专家，擅长从文档中提取结构化信息。

【任务说明】
从以下文档中提取结构化知识，包括：
1. 关键概念及定义
2. 重要指标及计算方法
3. 分析方法及应用场景

【输出格式】
严格按照以下JSON格式输出（不要有其他内容）：
{{
  "topic": "主题名称",
  "key_concepts": [
    {{"name": "概念名", "definition": "定义", "importance": "重要性说明"}}
  ],
  "indicators": [
    {{"name": "指标名", "calculation": "计算方法", "interpretation": "解读方式"}}
  ],
  "analysis_methods": [
    {{"name": "方法名", "steps": "步骤说明", "application": "应用场景"}}
  ]
}}

【示例】（Few-shot）
输入: "GDP是衡量一个国家经济总量的指标，计算公式为GDP=C+I+G+(X-M)..."
输出:
{{
  "topic": "GDP基础",
  "key_concepts": [
    {{"name": "GDP", "definition": "国内生产总值", "importance": "核心宏观指标"}}
  ],
  "indicators": [
    {{"name": "GDP", "calculation": "C+I+G+(X-M)", "interpretation": "数值越大经济规模越大"}}
  ]
}}

【约束条件】
1. 必须输出合法JSON，不要有markdown代码块标记
2. 不要遗漏文档中的重要概念
3. 定义要准确简洁（1-2句话）
4. 如果文档中没有相关内容，对应数组返回空[]

【文档内容】
{document_text}
"""
```

---

## 🔗 相关文档

- **`.cursorrules`**: 核心规则（自动加载，必须遵守）
- **`docs/开发手册.md`**: 开发工具速查
- **`README.md`**: 项目概况

---

**本文档作用**：
- 提供具体代码示例
- 展示"好的 vs 坏的"对比
- 作为编码时的参考（不强制AI读取）
- 核心规则在`.cursorrules`（100%可靠）
