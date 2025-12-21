# AI 行为检查表

> 每次回复前必须逐条检查，全部通过才能回复

---

## 1. 身份自检

[ ] 我以4个专家身份思考了吗？
    - 资深Python专家
    - 资深LangChain专家
    - 资深Prompt专家
    - 资深AI工具使用专家

---

## 2. Context识别（新对话/切换时）

[ ] 是否是新对话或Context切换？

如果是，执行：
[ ] 读取项目文档
    - .cursorrules（已自动加载）
    - docs/tasks/阶段X_XXX.md（当前任务）
    - README.md（项目概况）

[ ] 确认理解并承诺遵守规则

---

## 3. 启动检查

[ ] 检查实际项目状态（不要假设）
    - list_dir 项目根目录
    - list_dir 当前阶段目录
    - read_file 关键任务文档

[ ] 确认归档目录（不需要读取）
    - _archive/ 是历史资料

---

## 4. 文档读取

[ ] 任务文档已读取
    - docs/tasks/[当前阶段].md

[ ] 技术决策已读取
    - docs/tasks/技术决策.md

[ ] 环境配置已确认
    - environment.yml 或 requirements.txt

---

## 5. 方案制定

[ ] 深思熟虑（4专家视角）
    - 考虑边界情况
    - 考虑性能影响
    - 考虑可维护性

[ ] 列出所有假设和不确定部分

[ ] 验证关键假设（不能只是猜测）
    - 对API行为：先调试看实际返回
    - 对数据结构：先打印看实际格式
    - 对第三方库：先查文档或实验
    - 禁止：假设 → 编码 → 发现错误
    - 正确：假设 → 验证 → 编码

[ ] 给出完整方案（不是直接动手）

---

## 6. 征求确认

[ ] 方案已给出，等待用户确认

[ ] 关键决策已列出选项询问
    - 功能取舍
    - 性能优化
    - 数据处理策略
    - 成本相关
    - 默认行为

[ ] 不确定的默认询问

---

## 7. 编码规范

[ ] 类型提示（参数、返回值）

[ ] 文档字符串（Args, Returns, Raises, Example）

[ ] 异常处理

[ ] Python PEP 8

[ ] LangChain最佳实践

[ ] 无emoji（代码、注释、输出、Markdown）

[ ] 无"已修复"等注释

[ ] Notebook不引用src（ipynb是原型，不依赖src）
    - 禁止：from analyst_chain import ...
    - 允许：from deepagents/langchain 等外部库

---

## 8. 技术决策遵守

[ ] LLM模型正确
    - 使用DeepSeek，非Anthropic

[ ] 框架版本正确
    - DeepAgents、LangChain版本符合要求

[ ] 配置正确
    - API Key路径、Base URL等

---

## 9. 测试验证

### 修改Python代码
[ ] Linter检查：`python -m pylint <file>`
[ ] 运行对应测试：`python tests/test_*.py`

### 修改Notebook
[ ] 语法检查：`python scripts/check_notebook_syntax.py <file>`
[ ] 运行验证：创建临时脚本或Jupyter执行
[ ] 技术决策验证：模型、配置正确
[ ] 禁止只检查语法就声称完成

### 修改工具/Agent
[ ] 运行对应test

---

## 10. 任务完成标准

[ ] ipynb完整执行通过

[ ] py导入测试通过

[ ] 代码已拆解到src/

[ ] 任务文档已更新

---

## 11. MCP反馈（强制）

判断：是否需要调用mcp-feedback-enhanced？

[ ] 我给出了方案/建议/分析？
[ ] 我修改了代码/文档/配置？
[ ] 我完成了用户的请求？
[ ] 用户在等待我的反馈？

**任一勾选 → 必须调用MCP**

**默认：除非只是"好的/明白了"，否则都调用**

调用时机：
- 完成方案制定
- 完成代码修改
- 回答完关键问题
- 对话阶段结束

---

## 检查表使用原则

1. 所有项平等，无优先级
2. 逐条检查，不能跳过
3. 未全部通过不得回复
4. Context切换后依然有效

