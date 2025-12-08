# AKShare Tools 学习指南

> **目标**: 独立实现3个数据获取函数 (GDP/CPI/PMI)
> **时间**: 4-5小时
> **文件**: `notebooks/learning/akshare_tools_outline.ipynb`

---

## 1. AKShare快速入门 (5分钟)

### 安装
```bash
pip install akshare
```

### 基本用法
```python
import akshare as ak

# 获取数据(返回pandas DataFrame)
df = ak.macro_china_gdp_yearly()

# 查看数据
print(df.head())
print(df.columns)
```

### 核心概念
- **所有函数返回**: pandas DataFrame
- **列名**: 中文(需要处理)
- **数据类型**: object(需要转换为数值)
- **索引**: 默认整数索引

---

## 2. 需要实现的3个函数

### 函数1: `get_gdp_yearly()`

**目标**: 获取年度GDP数据

**AKShare API**:
```python
ak.macro_china_gdp_yearly()
```

**返回列名**:
- `年份`: 年份
- `国内生产总值-绝对值`: GDP绝对值(亿元)
- `国内生产总值-同比增长`: GDP同比增长率(%)

**你需要做**:
1. 调用API
2. 重命名列为英文: `year`, `gdp`, `gdp_yoy`
3. 转换数据类型: `gdp`和`gdp_yoy`转为float
4. 返回DataFrame

---

### 函数2: `get_cpi_monthly()`

**目标**: 获取月度CPI数据

**AKShare API**:
```python
ak.macro_china_cpi_monthly()
```

**返回列名**:
- `月份`: 月份(格式: YYYY-MM)
- `全国当月`: CPI当月同比(%)
- `全国累计`: CPI累计同比(%)

**你需要做**:
1. 调用API
2. 重命名列为英文: `month`, `cpi_mom`, `cpi_ytd`
3. 转换数据类型: `cpi_mom`和`cpi_ytd`转为float
4. 返回DataFrame

---

### 函数3: `get_pmi_manufacturing()`

**目标**: 获取制造业PMI数据

**AKShare API**:
```python
ak.macro_china_pmi_yearly()
```

**返回列名**:
- `月份`: 月份(格式: YYYY-MM)
- `制造业PMI`: 制造业PMI指数

**你需要做**:
1. 调用API
2. 重命名列为英文: `month`, `pmi`
3. 转换数据类型: `pmi`转为float
4. 返回DataFrame

---

## 3. DataFrame操作速查

### 重命名列
```python
df.rename(columns={
    '年份': 'year',
    '国内生产总值-绝对值': 'gdp'
}, inplace=True)
```

### 转换数据类型
```python
df['gdp'] = df['gdp'].astype(float)
```

### 查看数据类型
```python
print(df.dtypes)
```

### 查看列名
```python
print(df.columns.tolist())
```

---

## 4. 异常处理(可选)

```python
def get_gdp_yearly():
    try:
        df = ak.macro_china_gdp_yearly()
        # ... 处理数据 ...
        return df
    except Exception as e:
        print(f"获取GDP数据失败: {e}")
        return None
```

---

## 5. 测试方法

```python
# 测试函数
df = get_gdp_yearly()
print(df.head())
print(df.dtypes)
print(f"数据行数: {len(df)}")
```

---

## 6. 参考链接

- [AKShare官方文档](https://akshare.akfamily.xyz/)
- [pandas DataFrame文档](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html)

---

## 7. 下一步

完成后:
1. 运行测试
2. 找我对比完整实现
3. 优化改进

