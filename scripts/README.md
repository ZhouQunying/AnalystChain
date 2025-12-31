# Scripts 工具说明

> Notebook开发辅助工具

---

## check_notebook_syntax.py

**用途**: 检查Notebook语法错误
**调用时机**: 修改`.ipynb`文件后，AI自动调用（根据AI工作规范）
**命令**:
```bash
python scripts/check_notebook_syntax.py <notebook_path>
```

**说明**: 自动检查notebook中所有code cell的Python语法,报告语法错误和缩进错误

---

## Notebook测试流程

修改Notebook后的完整验证流程:
1. ✅ 语法检查: `python scripts/check_notebook_syntax.py <file>`
2. ✅ 完整执行: 在Jupyter中从头到尾运行一次
3. ✅ 输出验证: 检查输出结果是否正确

**提示**: 测试规则详见 `docs/AI工作规范.md` 第5部分

