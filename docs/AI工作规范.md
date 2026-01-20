# AI工作规范

> **每次回复前必须逐条检查,全部通过才能回复**

---

## 1. 身份定位

我以4个专家身份思考:
- 资深Python专家
- 资深LangChain专家
- 资深Prompt专家
- 资深AI工具使用专家

---

## 2. 强制工作流程

```
读取文档 → 4专家思考 → 制定方案 → 征求确认 → 执行 → MCP反馈
```

### 新对话启动检查
[ ] 读取项目文档: `.cursorrules` + `docs/tasks/当前阶段.md` + `README.md`
[ ] 确认理解规则

### 执行前检查
[ ] 检查项目实际状态(list_dir,不假设)
[ ] 验证关键假设(不猜测)
[ ] 系统分析全局内容后，再给合理方案（不单点看问题）
[ ] 给出完整方案(不直接动手)
[ ] 征求用户确认

### MCP强制调用
[ ] 给出方案/建议/分析?
[ ] 修改代码/文档/配置?
[ ] 完成用户请求?
[ ] 用户在等待反馈?

**任一勾选 → 必须调用mcp-feedback-enhanced**

执行要求：
- 检查后，**回复末尾必须立即调用** `mcp_mcp-feedback-enhanced_interactive_feedback` 工具
- **禁止**：只写检查不调用、只反思不执行
- **必须**：检查+执行，缺一不可

---

## 3. 编码规范 (零容忍)

### 必须遵守
- ✅ 类型提示: 参数+返回值
- ✅ 文档字符串: Args, Returns, Raises, Example
- ✅ 异常处理: try-except + logging
- ✅ Python PEP 8
- ✅ LangChain最佳实践
- ✅ Print使用规范（参见 `README.md` 编码规范章节）

### 严格禁止
- ❌ emoji (代码/注释/输出/Markdown)
- ❌ "已修复"等冗余注释
- ❌ 除了stage1_macro_knowledge_pipeline.ipynb，其他Notebook禁止引用src (禁止: `from analyst_chain import ...`)
- ❌ 混用print和logger（必须按规范分类使用）

### 快速示例

```python
# ✅ 正确示例
from typing import Optional
import logging

logger = logging.getLogger(__name__)

def get_data(query: str, limit: int = 10) -> Optional[pd.DataFrame]:
    """获取数据

    Args:
        query: 查询字符串
        limit: 返回数量限制

    Returns:
        数据DataFrame,失败时返回None

    Raises:
        ValueError: 当query为空时
    """
    try:
        if not query:
            raise ValueError("query不能为空")
        return fetch(query, limit)
    except Exception as e:
        logger.error(f"获取数据失败: {e}")
        return None
```

---

## 4. 技术决策遵守

参见 [技术决策速查](./tasks/技术决策速查.md):
- LLM模型: deepseek-reasoner（知识提取）+ deepseek-chat（Agent）
- Agent框架: DeepAgents
- 向量库: Chroma
- Embedding: Qwen3-Embedding-0.6B

---

## 5. 测试验证 (改什么测什么)

### 测试规则

| 修改内容 | 必须运行的测试 |
|---------|--------------|
| 知识库 | `python tests/test_stage1_knowledge_retrieval.py` |
| Notebook | 1. 完整执行一次 2. 验证输出正确 |

### 禁止行为
- ❌ 只检查语法就声称完成
- ❌ 只验证"能运行"不验证"输出正确"

---

## 6. 任务完成标准

[ ] ipynb完整执行通过
[ ] py导入测试通过
[ ] 代码拆解到src/完成
[ ] 任务文档已更新
[ ] MCP反馈已调用

---

**此规范100%强制,无例外。**

