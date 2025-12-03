# AI行为约束规范

> **目的**：确保AI编码质量和行为一致性，即使在context切换后也不遗漏

---

## 🔴 强制规则（MUST）

### 规则1：MCP调用
- **每次回复都必须调用** `mcp-feedback-enhanced`
- **未调用必须说明原因**
- **新context开始时优先调用**

### 规则2：任务对齐
- **所有编码必须对照** `docs/tasks/` 文档
- **偏离任务目标立即停止**
- **关键决策征求用户确认**

---

## 📋 编码质量标准

### 阶段1：编码前（设计Review）

**Checklist**：
- [ ] 明确任务目标（对照tasks文档）
- [ ] 查阅LangChain官方文档
- [ ] 设计模块接口（输入/输出/职责）
- [ ] 确定依赖关系
- [ ] 识别潜在风险

**输出**：
- 模块设计说明（接口、职责、技术选择）
- 征求用户确认后再编码

---

### 阶段2：编码中（质量嵌入）

**强制要求**：
```python
# 1. 类型提示（必须）
def process(docs: List[Document]) -> Dict[str, Any]:
    """
    2. 文档字符串（必须）
    
    Args:
        docs: 文档列表
        
    Returns:
        处理结果字典
        
    Raises:
        ValueError: 当输入为空时
        
    Example:
        >>> result = process([doc1, doc2])
    """
    # 3. 异常处理（必须）
    if not docs:
        raise ValueError("Documents cannot be empty")
    
    try:
        # 4. 日志记录（推荐）
        logger.info(f"Processing {len(docs)} documents")
        ...
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise
```

**Python专家标准**：
- ✅ PEP 8代码规范
- ✅ 类型提示完整（参数、返回值）
- ✅ 文档字符串清晰（Args, Returns, Raises, Example）
- ✅ 异常处理适当
- ✅ 变量命名清晰
- ✅ 函数单一职责

**LangChain专家标准**：
- ✅ 使用官方Loader（不自己解析PDF）
- ✅ 使用标准Document对象
- ✅ 正确配置Embeddings和VectorStore
- ✅ 遵循链式调用模式
- ✅ 利用LangChain内置功能（不重复造轮子）

---

### 阶段3：编码后（自我Review）

**Checklist**：
- [ ] **功能完整性**：是否完成tasks要求的功能？
- [ ] **代码可读性**：是否清晰易懂？
- [ ] **可复用性**：是否模块化、松耦合？
- [ ] **可扩展性**：是否易于添加新功能？
- [ ] **错误处理**：是否有适当的异常处理？
- [ ] **性能考虑**：是否有明显性能问题？
- [ ] **文档完整**：是否有清晰的文档字符串？

**输出**：
```
## 自我Review结果

✅ 功能完整性：已实现X功能，满足tasks要求
✅ 代码质量：符合Python和LangChain标准
✅ 可维护性：模块化设计，职责清晰
⚠️ 发现的问题：XXX（如有）

请Review确认
```

---

## 🏗️ 架构设计标准

### 单一职责原则
- 每个类/函数只负责一件事
- 命名清晰反映职责

### 松耦合
- 模块间通过接口交互
- 避免直接依赖具体实现

### 高内聚
- 相关功能组织在一起
- 减少模块间通信

### 可测试性
- 每个模块可独立测试
- 依赖可以mock

### 可扩展性
- 易于添加新功能
- 不需要修改现有代码（开闭原则）

---

## 🎯 Prompt工程标准

### LLM提示词设计（用于KnowledgeExtractor）

**结构化Prompt模板**：
```
【角色定义】
你是经济学知识提取专家...

【任务说明】
从以下文档中提取结构化知识...

【输出格式】
严格按照以下JSON格式输出：
{
  "topic": "主题",
  "key_concepts": [...]
}

【示例】（Few-shot）
输入：...
输出：{...}

【约束条件】
1. 必须输出合法JSON
2. 不要遗漏重要概念
3. ...
```

**质量检查**：
- ✅ 指令清晰明确
- ✅ 输出格式约束
- ✅ Few-shot示例
- ✅ 错误处理机制

---

## 🔄 新Context启动流程

当开始新的context时：

### Step 1：读取本规范
```
1. 读取 docs/AI_行为约束规范.md
2. 读取 docs/tasks/阶段X_XXX.md（当前任务）
3. 读取 docs/tasks/技术决策.md
```

### Step 2：确认理解
```
✅ 已读取行为约束规范
✅ 已了解当前任务目标
✅ 已确认技术决策
✅ 承诺遵守所有规则
```

### Step 3：调用MCP
```
立即调用mcp-feedback-enhanced，说明context恢复情况
```

---

## 📝 违规处理

### 如果发现违规：

**用户指出违规**：
```
用户："违反了XX标准"
AI：立即承认 → 分析原因 → 修正代码 → 自我检讨
```

**AI自我发现违规**：
```
立即停止 → 向用户报告 → 修正 → 提交review
```

---

## 🔗 相关文档

- [主任务](../docs/tasks/主任务.md) - 项目总览
- [阶段1任务](../docs/tasks/阶段1_知识基础.md) - 当前任务详情
- [技术决策](../docs/tasks/技术决策.md) - 技术选择记录

---

## 📊 质量自查表（每次提交前）

```
□ 调用了mcp-feedback-enhanced
□ 对照了tasks文档
□ 遵循了Python规范
□ 遵循了LangChain最佳实践
□ 完成了自我Review
□ 文档字符串完整
□ 异常处理适当
□ 代码可读清晰
```

**全部✅才能提交**

---

**最后更新**：2025-12-03
**版本**：v1.0

