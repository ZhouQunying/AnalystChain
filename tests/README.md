# Tests 测试说明

> 项目自动化测试套件

---

## 测试脚本

### test_stage1_knowledge_retrieval.py
**测试对象**: 知识库（Stage1产物：向量库+JSON）
**测试内容**: 验证知识库是否可用、检索功能是否正常
**调用时机**: 修改知识库相关代码后，AI自动调用（根据AI工作规范）
**运行**:
```bash
python tests/test_stage1_knowledge_retrieval.py
```

---

## 测试规则

**原则**: 改什么测什么

| 修改内容 | 必须运行的测试 | 调用时机 |
|---------|---------------|---------|
| 知识库 | `python tests/test_stage1_knowledge_retrieval.py` | 修改知识库相关代码后 |
| Notebook | 1. `python scripts/check_notebook_syntax.py <file>` 2. 完整执行验证 | 修改`.ipynb`文件后 |

---

## 运行所有测试

```bash
# 运行所有tests/下的测试
python -m pytest tests/

# 或逐个运行
python tests/test_stage1_knowledge_retrieval.py
```

---

**详细测试规范**: 参见 `docs/AI工作规范.md` 第5部分

