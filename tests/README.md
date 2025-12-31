# Tests 测试说明

> 项目自动化测试套件

---

## 测试脚本

### test_knowledge_retrieval.py
**测试对象**: `src/analyst_chain/tools/knowledge_retrieval.py`
**测试内容**: 向量检索+JSON查询功能
**调用时机**: 修改知识库相关代码后，AI自动调用（根据AI工作规范）
**运行**:
```bash
python tests/test_knowledge_retrieval.py
```

### test_akshare_tools.py
**测试对象**: `src/analyst_chain/tools/akshare_tools.py`
**测试内容**: GDP/CPI/PMI数据获取功能
**调用时机**: 修改`src/analyst_chain/tools/`下的代码后，AI自动调用（根据AI工作规范）
**运行**:
```bash
python tests/test_akshare_tools.py
```

### test_macro_agent.py
**测试对象**: `src/analyst_chain/agents/macro_agent.py`
**测试内容**: Agent创建+工具调用+流式输出
**调用时机**: 修改`src/analyst_chain/agents/`下的代码后，AI自动调用（根据AI工作规范）
**运行**:
```bash
python tests/test_macro_agent.py
```

---

## 测试规则

**原则**: 改什么测什么

| 修改内容 | 必须运行的测试 | 调用时机 |
|---------|---------------|---------|
| 知识库 | `python tests/test_knowledge_retrieval.py` | 修改知识库相关代码后 |
| tools/ | `python tests/test_akshare_tools.py` | 修改`src/analyst_chain/tools/`后 |
| agents/ | `python tests/test_macro_agent.py` | 修改`src/analyst_chain/agents/`后 |
| Notebook | 1. `python scripts/check_notebook_syntax.py <file>` 2. 完整执行验证 | 修改`.ipynb`文件后 |

---

## 运行所有测试

```bash
# 运行所有tests/下的测试
python -m pytest tests/

# 或逐个运行
python tests/test_akshare_tools.py
python tests/test_knowledge_retrieval.py
python tests/test_macro_agent.py
```

---

**详细测试规范**: 参见 `docs/AI工作规范.md` 第5部分

