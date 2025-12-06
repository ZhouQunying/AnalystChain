# 阶段2：单个SubAgent原型

> **目标：** 实现第一个完整的SubAgent（宏观经济分析）
> **状态：** 🟡 进行中
> **预计时间：** 3-4天
> **全局规范：** [AI行为约束规范](../AI行为约束规范.md) ⚠️ 必须遵守

---

## 📊 进度

```
进度: ░░░░░░░░░░░░ 0/4 (0%)
```

---

## 📝 任务清单

### 任务1: AKShare Tools封装 ⏸️

**预计时间：** 0.5天

- [ ] 创建tools模块目录结构
- [ ] 实现GDP数据获取工具:
  - `get_gdp_yearly()`: 年度GDP数据
  - `get_gdp_quarterly()`: 季度GDP数据
- [ ] 实现CPI数据获取工具:
  - `get_cpi_monthly()`: 月度CPI数据
  - `get_cpi_yearly()`: 年度CPI数据
- [ ] 实现PMI数据获取工具:
  - `get_pmi_monthly()`: 制造业PMI
  - `get_pmi_non_manufacturing()`: 非制造业PMI
- [ ] 工具测试与文档编写

**文件：** `jupyter_notebook/macroeconomic_analysis/tools/akshare_tools.py`

---

### 任务2: 知识库检索工具封装 ⏸️

**预计时间：** 0.5天

- [ ] 创建知识库检索工具:
  - 向量检索功能
  - JSON知识查询功能
  - 结果格式化
- [ ] 集成到DeepAgents Tools
- [ ] 测试检索准确性

**文件：** `jupyter_notebook/macroeconomic_analysis/tools/knowledge_retrieval.py`

---

### 任务3: SubAgent实现 ⏸️

**预计时间：** 1.5天

- [ ] 完善SubAgent配置:
  - description优化
  - system_prompt优化
  - tools绑定(AKShare + 知识检索)
  - model配置
- [ ] 实现主Agent:
  - 创建DeepAgent
  - 配置macroeconomic_subagent
  - 测试subagent调用
- [ ] 测试场景:
  - "最近GDP增长率如何?"
  - "当前通胀水平怎么样?"
  - "PMI数据显示经济趋势如何?"

**文件：** `jupyter_notebook/macroeconomic_analysis/notebooks/deep_agent.ipynb`

---

### 任务4: 端到端测试与优化 ⏸️

**预计时间：** 1天

- [ ] 完整分析流程测试:
  - 用户提问 → SubAgent分析 → 返回结果
  - 测试10个不同类型的问题
- [ ] 输出质量评估:
  - 数据准确性
  - 分析专业性
  - 结论清晰度
- [ ] Prompt迭代优化:
  - 根据测试结果调整system_prompt
  - 优化输出格式
- [ ] 性能评估:
  - 响应时间
  - Token消耗

**测试问题清单**:
```python
test_questions = [
    "2024年GDP增长率是多少?",
    "当前的通胀水平如何?",
    "PMI数据反映出什么经济信号?",
    "消费数据的最新趋势是什么?",
    "出口数据表现如何?",
    "当前处于什么经济周期?",
    "投资数据有何变化?",
    "货币政策对经济有什么影响?",
    "如何解读最新的经济数据?",
    "给出当前宏观经济的整体判断"
]
```

---

## 📚 相关技术决策

本阶段涉及的技术决策:
- [决策#004: Agent框架选择](./技术决策.md#决策-004agent框架选择) - DeepAgents
- [决策#005: LLM模型选择](./技术决策.md#决策-005llm模型选择用于知识提取) - deepseek-chat

新增技术决策(需记录):
- SubAgent Tools选择(AKShare API封装)
- Prompt设计策略
- 输出格式定义

---

## 🎯 阶段2成功标准

### 功能要求:
- ✅ SubAgent能独立回答宏观经济问题
- ✅ 能正确调用AKShare获取数据
- ✅ 能从知识库检索相关理论
- ✅ 能生成专业的分析报告

### 质量要求:
- ✅ 数据准确(引用正确来源)
- ✅ 分析专业(符合经济学理论)
- ✅ 结论清晰(易于理解和决策)
- ✅ 响应及时(< 30秒)

### 交付物:
- [ ] 可运行的deep_agent.ipynb
- [ ] AKShare Tools封装代码
- [ ] 知识检索工具代码
- [ ] 测试报告(10个问题的回答质量)

---

## 📝 工作日志

### 2025-12-06
- 🟡 开始阶段2
- ✅ 创建阶段2任务文档
- ✅ 完成SubAgent基础配置(description + system_prompt)
- ⏸️ 待开始: AKShare Tools封装

---

[← 返回主任务](./主任务.md)

