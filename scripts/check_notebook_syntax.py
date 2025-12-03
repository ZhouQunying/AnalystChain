#!/usr/bin/env python3
"""
Jupyter Notebook语法检查工具
自动检查notebook中code cell的Python语法
"""
import json
import sys
import ast
from pathlib import Path

def check_notebook(notebook_path: Path) -> bool:
    """检查notebook中所有code cell的语法

    Returns:
        True if all cells are valid, False otherwise
    """
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    has_error = False

    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue

        source = ''.join(cell['source'])
        if not source.strip():
            continue

        try:
            ast.parse(source)
        except SyntaxError as e:
            print(f"\n[错误] Cell {i}: 语法错误")
            print(f"  行 {e.lineno}: {e.msg}")
            print(f"  {e.text}")
            has_error = True
        except IndentationError as e:
            print(f"\n[错误] Cell {i}: 缩进错误")
            print(f"  行 {e.lineno}: {e.msg}")
            print(f"  {e.text}")
            has_error = True

    return not has_error

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python check_notebook_syntax.py <notebook.ipynb>")
        sys.exit(1)

    notebook_path = Path(sys.argv[1])
    if not notebook_path.exists():
        print(f"文件不存在: {notebook_path}")
        sys.exit(1)

    print(f"检查 {notebook_path.name}...")

    if check_notebook(notebook_path):
        print("[成功] 所有cell语法正确")
        sys.exit(0)
    else:
        print("\n[失败] 发现语法错误")
        sys.exit(1)

