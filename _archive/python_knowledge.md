# Python知识速查

> **更新提示**：本文档遵循"简要、易记忆、系统（而非单点知识）"原则，每个知识点都包含：整体分类、知识结构图、记忆口诀，便于快速查阅和记忆。

---

## 下划线命名系统

### 整体分类（4类）

```
单下划线前缀（_name）     → 内部使用约定
双下划线前缀（__name）    → 名称改写（Name Mangling）
单下划线后缀（name_）     → 避免关键字冲突
双下划线前后（__name__） → 特殊方法/属性（Magic Methods）
```

### 详细说明

#### 1. 单下划线前缀 `_name`

**含义**：内部使用约定（非强制）

**用途**：
- 表示"内部使用，外部不要直接访问"
- `from module import *` 时不会被导入
- 约定，不强制

**示例**：
```python
class MyClass:
    def __init__(self):
        self.public = "公开"      # 公开属性
        self._internal = "内部"   # 内部属性（约定）
```

#### 2. 双下划线前缀 `__name`

**含义**：名称改写（Name Mangling，强制）

**用途**：
- Python自动改写为 `_类名__name`
- 避免子类覆盖父类私有属性
- 强制私有化

**示例**：
```python
class Parent:
    def __init__(self):
        self.__private = "私有"  # 实际变成 _Parent__private

class Child(Parent):
    def __init__(self):
        super().__init__()
        self.__private = "子类私有"  # 实际变成 _Child__private（不会覆盖父类）
```

#### 3. 单下划线后缀 `name_`

**含义**：避免关键字冲突

**用途**：
- 当变量名与Python关键字冲突时使用
- 例如：`class_`（避免与`class`冲突）、`type_`、`import_`

**示例**：
```python
def function(class_=None):  # class是关键字，用class_避免冲突
    pass
```

#### 4. 双下划线前后 `__name__`

**含义**：特殊方法/属性（Magic Methods）

**分类**：
- **特殊方法**（Magic Methods）：类中定义，由解释器自动调用
- **特殊属性**（Built-in Attributes）：模块/函数的内置属性

**重要区别**：
- `__init__`、`__str__`等是**方法**（类中定义）
- `__name__`、`__file__`等是**属性**（模块/函数的内置属性）

**示例**：
```python
class MyClass:
    def __init__(self):      # 特殊方法（构造方法）
        pass

    def __str__(self):       # 特殊方法（字符串表示）
        return "MyClass"

print(__name__)  # 特殊属性（当前模块名）
print(__file__)  # 特殊属性（当前文件路径）
```

### 对比表

| 写法 | 含义 | 强制/约定 | 用途 | 示例 |
|------|------|----------|------|------|
| `_name` | 内部使用 | 约定 | 内部属性/方法 | `_internal` |
| `__name` | 名称改写 | 强制 | 私有属性（避免覆盖） | `__private` |
| `name_` | 避免冲突 | 约定 | 与关键字冲突时 | `class_` |
| `__name__` | 特殊方法 | 内置 | 魔法方法/属性 | `__init__` |

### 知识结构图

```
Python下划线命名系统
├── 单下划线前缀（_name）
│   └── 内部使用约定（from import *不导入）
├── 双下划线前缀（__name）
│   └── 名称改写（_类名__name，强制私有）
├── 单下划线后缀（name_）
│   └── 避免关键字冲突
└── 双下划线前后（__name__）
    └── 特殊方法/属性（内置，不应自定义）
```

### 记忆口诀

- 单前：内部约定（`_name`）
- 双前：强制私有（`__name`）
- 单后：避免冲突（`name_`）
- 双前后：特殊方法（`__name__`）

### 注意事项

- `_name_`（前后都是单下划线）**没有约定**，视为普通变量（不推荐使用）

---

## Python特殊方法（Magic Methods）系统

### 整体分类（6类）

```
对象创建/销毁（__init__/__del__）
    ↓
对象表示（__str__/__repr__）
    ↓
运算符重载（__add__/__eq__等）
    ↓
容器操作（__len__/__getitem__等）
    ↓
上下文管理（__enter__/__exit__）
    ↓
属性访问（__getattr__/__setattr__）
```

### 常见特殊方法（20个）

#### 1. 对象创建/销毁（2个）

| 方法 | 作用 | 调用时机 | 示例 |
|------|------|---------|------|
| `__init__(self, ...)` | 构造方法 | 对象创建时 | `obj = MyClass()` |
| `__del__(self)` | 析构方法 | 对象销毁时 | `del obj` |

#### 2. 对象表示（2个）

| 方法 | 作用 | 返回类型 | 示例 |
|------|------|---------|------|
| `__str__(self)` | 用户友好字符串 | str | `print(obj)` → 调用`__str__` |
| `__repr__(self)` | 开发者字符串 | str | `repr(obj)` → 调用`__repr__` |

**区别**：
- `__str__`：给用户看（简洁易懂）
- `__repr__`：给开发者看（完整信息，可用来重建对象）

#### 3. 运算符重载（8个）

| 方法 | 运算符 | 作用 | 示例 |
|------|--------|------|------|
| `__add__(self, other)` | `+` | 加法 | `obj1 + obj2` |
| `__sub__(self, other)` | `-` | 减法 | `obj1 - obj2` |
| `__mul__(self, other)` | `*` | 乘法 | `obj1 * obj2` |
| `__eq__(self, other)` | `==` | 相等 | `obj1 == obj2` |
| `__lt__(self, other)` | `<` | 小于 | `obj1 < obj2` |
| `__le__(self, other)` | `<=` | 小于等于 | `obj1 <= obj2` |
| `__gt__(self, other)` | `>` | 大于 | `obj1 > obj2` |
| `__ge__(self, other)` | `>=` | 大于等于 | `obj1 >= obj2` |

#### 4. 容器操作（4个）

| 方法 | 作用 | 调用时机 | 示例 |
|------|------|---------|------|
| `__len__(self)` | 返回长度 | `len(obj)` | `len(my_list)` |
| `__getitem__(self, key)` | 获取元素 | `obj[key]` | `my_list[0]` |
| `__setitem__(self, key, value)` | 设置元素 | `obj[key] = value` | `my_list[0] = 1` |
| `__contains__(self, item)` | 成员检查 | `item in obj` | `1 in my_list` |

#### 5. 上下文管理（2个）

| 方法 | 作用 | 调用时机 | 示例 |
|------|------|---------|------|
| `__enter__(self)` | 进入上下文 | `with obj:` | `with open(...) as f:` |
| `__exit__(self, exc_type, exc_val, exc_tb)` | 退出上下文 | `with`块结束 | 自动调用 |

#### 6. 属性访问（2个）

| 方法 | 作用 | 调用时机 | 示例 |
|------|------|---------|------|
| `__getattr__(self, name)` | 获取属性（不存在时） | `obj.attr`（属性不存在） | `obj.unknown` |
| `__setattr__(self, name, value)` | 设置属性 | `obj.attr = value` | `obj.x = 1` |

### 特殊属性（内置属性，非方法）

| 属性 | 作用 | 适用对象 | 示例 |
|------|------|---------|------|
| `__name__` | 名称 | 模块/函数/类 | `__name__` → `"__main__"` |
| `__file__` | 文件路径 | 模块 | `__file__` → `"/path/to/file.py"` |
| `__doc__` | 文档字符串 | 函数/类/模块 | `func.__doc__` |
| `__module__` | 所属模块 | 类/函数 | `MyClass.__module__` |
| `__dict__` | 属性字典 | 对象/类 | `obj.__dict__` |
| `__class__` | 类对象 | 对象 | `obj.__class__` |

**重要**：这些是**属性**（不是方法），不需要调用，直接访问

### 知识结构图

```
Python特殊方法/属性系统
├── 特殊方法（Magic Methods，类中定义）
│   ├── 对象创建/销毁（__init__/__del__）
│   ├── 对象表示（__str__/__repr__）
│   ├── 运算符重载（__add__/__eq__等8个）
│   ├── 容器操作（__len__/__getitem__等4个）
│   ├── 上下文管理（__enter__/__exit__）
│   └── 属性访问（__getattr__/__setattr__）
└── 特殊属性（Built-in Attributes，内置）
    └── __name__/__file__/__doc__/__dict__/__class__等
```

### 记忆口诀

- 创建：`__init__`构造，`__del__`析构
- 表示：`__str__`用户看，`__repr__`开发者看
- 运算：`__add__`加，`__eq__`等，`__lt__`小于
- 容器：`__len__`长度，`__getitem__`索引
- 上下文：`__enter__`进入，`__exit__`退出
- 属性：`__getattr__`获取，`__setattr__`设置

---

## Python类型提示（typing）系统

### 整体分类（5类）

```
基础类型（int/str/bool/float）
    ↓
可选类型（Optional/Union）
    ↓
容器类型（List/Dict/Tuple/Set）
    ↓
特殊类型（Callable/TypeVar）
    ↓
类型别名（TypeAlias）
```

### 常见类型提示（11个）

| 类型 | 作用 | 示例 |
|------|------|------|
| `int` / `str` / `bool` / `float` | 基础类型 | `age: int = 25` |
| `Optional[T]` | 可选（T或None） | `Optional[str]` → `str \| None` |
| `Union[T1, T2]` | 联合类型 | `Union[int, str]` → `int \| str` |
| `List[T]` | 列表 | `List[str]` → `list[str]`（3.9+） |
| `Dict[K, V]` | 字典 | `Dict[str, int]` → `dict[str, int]`（3.9+） |
| `Tuple[T, ...]` | 元组 | `Tuple[int, str]` → `tuple[int, str]`（3.9+） |
| `Set[T]` | 集合 | `Set[str]` → `set[str]`（3.9+） |
| `TypedDict` | 字典类型提示 | `class User(TypedDict): name: str` |
| `Callable[[参数], 返回]` | 函数类型 | `Callable[[str, int], bool]` |
| `TypeVar('T')` | 泛型变量 | `T = TypeVar('T')` |
| `TypeAlias` | 类型别名 | `UserId: TypeAlias = int` |

### 知识结构图

```
类型提示系统
├── 基础类型（int/str/bool/float）
├── 可选类型（Optional/Union，3.10+可用|简化）
├── 容器类型（List/Dict/Tuple/Set/TypedDict，3.9+可用内置类型简化）
├── 特殊类型（Callable/TypeVar）
└── 类型别名（TypeAlias）
```

### 记忆口诀

- 基础：`int`整数，`str`字符串，`bool`布尔，`float`浮点
- 可选：`Optional[T]`可空，`Union[T1,T2]`联合
- 容器：`List[T]`列表，`Dict[K,V]`字典，`Tuple`元组，`Set`集合，`TypedDict`字典类型
- 特殊：`Callable`函数，`TypeVar`泛型
- 别名：`TypeAlias`简化复杂类型

### 适用场景

- **函数参数/返回值**：标注参数和返回值类型
- **变量标注**：标注变量类型
- **提高可读性**：让代码意图更清晰
- **IDE支持**：提供自动补全和类型检查

---

## 列表操作（extend/append/索引）

### 核心区别

**extend 拆开加，append 整个加，[i] 索引取改**

### 对比表

| 操作 | 作用 | 示例 | 结果 |
|------|------|------|------|
| `extend()` | 拆开逐个添加 | `[1,2].extend([3,4])` | `[1,2,3,4]` |
| `append()` | 整个作为一个元素 | `[1,2].append([3,4])` | `[1,2,[3,4]]` |
| `[i]` | 索引访问/修改 | `arr[0]` / `arr[0]=5` | 取/改元素 |

### 记忆口诀

```
extend 拆开加
append 整个加
[i] 取改用索引
```

---

## 字典操作（keys/values/items）

| 操作 | 结果 | 记忆 |
|------|------|------|
| `dict.keys()` | `['A', 'B']` | 只要钥匙 |
| `dict.values()` | `[1, 2]` | 只要宝藏 |
| `dict.items()` | `[('A',1), ('B',2)]` | 钥匙+宝藏打包 |

---

## 数据结构定义方式对比

### 3种方式

| 方式 | 是什么 | 验证 | 适用场景 |
|------|--------|------|----------|
| `@dataclass` | 自动生成方法的类 | ❌ 无 | 简单数据类 |
| `Pydantic` | 带验证的类 | ✅ 有 | API/配置（需验证） |
| `TypedDict` | 字典类型提示 | ❌ 无 | 字典类型提示 |

### 简单理解

**@dataclass**（类）：
- 定义后可以 `obj = FileInfo(...)` 创建对象
- 自动生成 `__init__`、`__repr__` 等方法
- 不验证数据（可以赋值错误类型）
- 适合：需要对象的地方

**Pydantic**（类）：
- 定义后可以 `obj = User(...)` 创建对象
- 自动验证数据（类型、值范围等）
- 自动序列化（转字典、JSON）
- 适合：API 请求/响应、配置文件
- 性能好（核心用 Rust 实现，Rust 是比 Python 更快的系统编程语言）

**TypedDict**（字典）：
- 不是类，只是类型提示
- 使用时仍是普通字典：`user = {"name": "张三", "age": 25}`
- 不验证数据（只是告诉类型检查器）
- 适合：需要字典的地方

### 区别

**@dataclass vs TypedDict**：
- `@dataclass`：创建的是**对象**（`obj.name`）
- `TypedDict`：创建的是**字典**（`obj["name"]`）

### 选择

- 需要验证 → Pydantic
- 不需要验证 + 用对象 → `@dataclass`
- 不需要验证 + 用字典 → `TypedDict`

---

## Enum 与 IntEnum

### 核心区别

**IntEnum 是数字可直接比较，Enum 必须用 `.value`**

### 对比表

| 特性 | IntEnum | Enum |
|------|---------|------|
| **能否比较** | ✅ 直接比较 | ❌ 需要 .value |
| **.value** | 可选（冗余） | 必须 |
| **适用** | 优先级、状态码 | 类型、配置 |

### 代码示例

```python
# 定义
from enum import IntEnum, Enum

class Priority(IntEnum):
    HIGH = 1
    LOW = 2

class FileType(Enum):
    PDF = 'pdf'
    WORD = 'word'

# 使用方式
Priority.HIGH < Priority.LOW           # ✅ True（IntEnum 直接比较）
FileType.PDF < FileType.WORD           # ❌ 报错（Enum 不能比较）
FileType.PDF.value == 'pdf'            # ✅ True（Enum 必须 .value）
```

### 记忆口诀

```
IntEnum 是数字，直接比
Enum 要 value，才能用
```

---

## Python异常处理系统

### 异常体系（3层结构）

```
BaseException（所有异常的基类）
├── SystemExit（程序退出）
├── KeyboardInterrupt（Ctrl+C中断）
└── Exception（所有普通异常）
    ├── StopIteration（迭代结束）
    ├── ArithmeticError（算术错误）
    │   ├── ZeroDivisionError（除零）
    │   └── OverflowError（溢出）
    ├── LookupError（查找错误）
    │   ├── IndexError（索引越界）
    │   └── KeyError（键不存在）
    ├── ValueError（值错误）
    ├── TypeError（类型错误）
    ├── FileNotFoundError（文件不存在）
    └── RuntimeError（运行时错误）← 环境问题
```

### 常见异常类型（8个）

| 异常类型 | 使用场景 | 示例 |
|---------|---------|------|
| `ValueError` | 参数值错误 | `int("abc")` |
| `TypeError` | 类型错误 | `"1" + 1` |
| `IndexError` | 索引越界 | `list[10]`（列表只有3个元素） |
| `KeyError` | 字典键不存在 | `dict["不存在的键"]` |
| `FileNotFoundError` | 文件不存在 | `open("不存在的文件.txt")` |
| `ZeroDivisionError` | 除零 | `1 / 0` |
| `RuntimeError` | 运行时环境问题 | 运行时环境配置错误 |
| `StopIteration` | 迭代结束 | `next(空迭代器)` |

### 知识结构图

```
异常处理系统
├── 异常体系（3层）
│   ├── BaseException（基类）
│   ├── Exception（普通异常）
│   └── 具体异常类型（ValueError、TypeError、RuntimeError等）
```

### 记忆口诀

- 异常体系：3层结构（BaseException → Exception → 具体类型）

---

## `__init__.py` 与 `__all__` 系统

### 核心规则（一句话）

**`__init__.py` 导入啥，包就能导啥；`__all__` 只管星号**

### 对比表（3种情况）

| `__init__.py` | `from pkg import item` | `from pkg.mod import item` |
|--------------|----------------------|--------------------------|
| **无文件** | ❌ | ✅ |
| **空文件** | ❌ | ✅ |
| **有导入** | ✅ | ✅ |

**`__all__` 作用**：
- ✅ 控制 `from pkg import *`（只导入 `__all__` 列表中的）
- ❌ 不影响 `from pkg import item`（指定导入不受限）

### 代码示例

#### 情况1：无 `__init__.py`

```python
# 文件内容
# （无 __init__.py 文件）

# 使用方式
from pkg.mod import item  # ✅ 唯一方式
from pkg import item      # ❌ 报错：cannot import
```

#### 情况2：空 `__init__.py`

```python
# 文件内容（pkg/__init__.py）
# （空文件）

# 使用方式
from pkg.mod import item  # ✅ 唯一方式
from pkg import item      # ❌ 报错：cannot import
```

#### 情况3：有导入

```python
# 文件内容（pkg/__init__.py）
from .mod import item

# 使用方式
from pkg import item      # ✅ 可以（init导入了item）
from pkg.mod import item  # ✅ 也可以（直接从模块导）
```

#### `__all__` 作用

```python
# 文件内容（pkg/__init__.py）
from .mod import item, other
__all__ = ['item']

# 使用方式
from pkg import item   # ✅ 可以（指定导入不受 __all__ 限制）
from pkg import other  # ✅ 可以（指定导入不受 __all__ 限制）
from pkg import *      # ⚠️ 只导入 item（__all__ 只控制星号）
```

### 知识结构图

```
__init__.py 与 __all__
├── __init__.py（控制包级导入）
│   ├── 无文件 → 只能深入导入（from pkg.mod import）
│   ├── 空文件 → 只能深入导入（from pkg.mod import）
│   └── 有导入 → 可以包级导入（from pkg import）
└── __all__（只控制星号导入）
    ├── 影响：from pkg import *
    └── 不影响：from pkg import item
```

### 记忆口诀

```
init导啥包导啥
all只管星号导
无init深入导
有init直接导
```

---

## Python包安装与导入路径

### 核心规则（一句话）

**有 `setup.py` 必须用 `pip install -e .`，装完用包名导入，不装只能 hack `sys.path`**

### 对比表（2种导入方式）

| 导入方式 | 前提条件 | sys.path 包含 | 使用场景 | 推荐度 |
|---------|---------|--------------|---------|-------|
| `from analyst_chain.x` | `pip install -e .` | `/project/src/` | 测试、生产 | ✅ 标准 |
| `from src.analyst_chain.x` | `sys.path.insert(0, '/project')` | `/project/` | 无 | ❌ hack |

### `pip install -e .` 的作用

```python
# setup.py 内容（定义包结构）
from setuptools import setup, find_packages

setup(
    name='analyst-chain',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
)

# 使用方式
# 在项目根目录执行
pip install -e .

# 效果
# 1. Python 读取 setup.py
# 2. 看到 package_dir={'': 'src'}
# 3. 将 /project/src/ 添加到 sys.path
# 4. analyst_chain 变成可导入的包
```

### 导入路径对比

```python
# 方式1：未安装（hack sys.path）
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 使用方式
from src.analyst_chain.tools import KnowledgeRetriever  # ❌ 不推荐（hack）
```

```python
# 方式2：已安装（标准方式）
# 前提：已执行 pip install -e .

# 使用方式
from analyst_chain.tools import KnowledgeRetriever  # ✅ 推荐（标准）
```

### 为什么不带 `src.`

```python
# setup.py 的关键配置
setup(
    package_dir={'': 'src'},
)

# 翻译
# analyst_chain 包的根目录是 src/
# 不是项目根目录
# 所以导入时不带 src.
```

### 对比总结

| 维度 | hack sys.path | pip install -e . |
|------|--------------|-----------------|
| **标准性** | ❌ 临时方案 | ✅ Python 标准 |
| **可维护性** | ❌ 路径硬编码 | ✅ setup.py 统一管理 |
| **IDE支持** | ❌ 补全可能失效 | ✅ 完整支持 |
| **适用场景** | 无 | 测试、开发、生产 |

### 知识结构图

```
Python包安装与导入
├── setup.py（定义包结构）
│   ├── package_dir={'': 'src'}（包在 src/ 下）
│   └── packages=find_packages(where='src')（查找 src/ 下所有包）
├── pip install -e .（开发模式安装）
│   ├── 读取 setup.py
│   ├── 将 src/ 加入 sys.path
│   └── 包名可直接导入（不带 src.）
└── 导入方式
    ├── ✅ from analyst_chain.x（标准）
    └── ❌ from src.analyst_chain.x（hack）
```

### 记忆口诀

```
setup.py 定义包结构
pip install -e . 装开发
装完导入用包名（不带 src.）
不装只能 hack path（不推荐）
```

---

## Python装饰器系统

### 整体分类（3类）

```
内置装饰器（@property/@staticmethod/@classmethod/@dataclass）
    ↓
第三方装饰器（@app.route/@login_required）
    ↓
自定义装饰器（def my_decorator(func): ...）
```

### 对比表（4个常用内置）

| 装饰器 | 第一参数 | 调用方式 | 一句话理解 |
|--------|---------|---------|-----------|
| 普通方法 | `self` | `obj.method()` | 操作某个具体对象 |
| `@property` | `self` | `obj.attr` | 方法伪装成属性（无括号） |
| `@staticmethod` | 无 | `Class.method()` | 放在类里的普通函数 |
| `@classmethod` | `cls` | `Class.method()` | 能访问类本身（常用于另一种创建方式） |

### 示例

```python
class Person:
    @property
    def info(self): return "只读属性"     # 调用：p.info（无括号）

    @staticmethod
    def is_adult(age): return age >= 18   # 调用：Person.is_adult(20)

    @classmethod
    def from_dict(cls, d): return cls()   # 调用：Person.from_dict({})
```

### 知识结构图

```
装饰器系统
├── @property → 方法变属性（无括号）
├── @staticmethod → 无self，放类里的普通函数
├── @classmethod → 有cls，访问类本身
└── @dataclass → 自动生成__init__
```

### 记忆口诀

```
property 无括号
static 无 self
classmethod 有 cls
装饰器：包装增功能
```

---

## Python `with` 语句（上下文管理器）

### 核心规则（一句话）

**`with` = 自动管理资源（打开必关闭、获取必释放）**

### 机制

| 阶段 | 动作 | 对应方法 |
|------|------|---------|
| 进入 `with` 块 | 获取资源 | `__enter__()` |
| 退出 `with` 块 | **自动**释放资源 | `__exit__()`（即使异常也执行） |

### 对比

**不用 with（手动管理）**：
```python
f = open("file.txt", "r")
try:
    data = f.read()
finally:
    f.close()  # 必须手动关闭，易忘
```

**用 with（自动管理）**：
```python
with open("file.txt", "r") as f:
    data = f.read()
# 自动调用 f.close()，即使发生异常也会关闭
```

### 常见场景

| 场景 | 示例 | 自动管理的资源 |
|------|------|---------------|
| 文件操作 | `with open() as f` | 文件句柄（自动关闭） |
| 数据库连接 | `with conn.cursor() as cur` | 游标（自动关闭） |
| 线程锁 | `with threading.Lock()` | 锁（自动释放） |
| 网络请求 | `with requests.Session() as s` | 连接池（自动关闭） |

### 记忆口诀

**`with` = "用完自动收"**

---

## Python 集合运算

### 运算对照表

| 运算 | 符号 | 方法 | 含义 | 示例 |
|------|------|------|------|------|
| 并集 | `\|` | `union()` | A或B中的元素 | `{1,2} \| {2,3}` → `{1,2,3}` |
| 交集 | `&` | `intersection()` | A且B中的元素 | `{1,2} & {2,3}` → `{2}` |
| 差集 | `-` | `difference()` | 在A不在B | `{1,2} - {2,3}` → `{1}` |
| 对称差集 | `^` | `symmetric_difference()` | 在A或B，但不同时在两者 | `{1,2} ^ {2,3}` → `{1,3}` |
| 子集 | `<=` | `issubset()` | A是否是B的子集 | `{1,2} <= {1,2,3}` → `True` |
| 超集 | `>=` | `issuperset()` | A是否包含B | `{1,2,3} >= {1,2}` → `True` |

### 数据结构语法区分

| 数据结构 | 语法 | 示例 |
|----------|------|------|
| 集合 (set) | `{元素, ...}` | `{1, 2, 3}` |
| 元组 (tuple) | `(元素, ...)` | `(1, 2, 3)` |
| 列表 (list) | `[元素, ...]` | `[1, 2, 3]` |
| 字典 (dict) | `{键: 值, ...}` | `{"a": 1}` |

**注意**：
- `{}` → **dict**（空字典，不是空集合！）
- `set()` → **set**（创建空集合必须用 `set()`）

### 记忆口诀

```
并用竖、交用与、差用减、对称用异或
花括号无冒号是集合，有冒号是字典，空花括号是字典
```

---

## Python 迭代器与生成器系统

### 整体架构（3层）

```
可迭代对象（Iterable）
    ↓ iter()
迭代器（Iterator）
    ↓ next()
生成器（Generator）= 特殊迭代器（用 yield 创建）
    ↓ async/await
异步生成器（AsyncGenerator）= 异步版本
```

### 核心概念对比

| 概念 | 定义 | 创建方式 | 遍历方式 |
|------|------|---------|---------|
| **可迭代对象** | 能被 `for` 遍历的对象 | 实现 `__iter__()` | `for item in obj` |
| **迭代器** | 记住遍历位置的对象 | 实现 `__iter__()` + `__next__()` | `next(it)` |
| **生成器** | 用 `yield` 创建的迭代器 | 函数中用 `yield` | `next(gen)` / `for` |
| **异步生成器** | 异步版生成器 | `async def` + `yield` | `async for` |

### 1. 可迭代对象（Iterable）

**一句话**：能被 `for` 遍历的对象

```python
# 常见可迭代对象
list, tuple, str, dict, set, range, file

# 判断
from collections.abc import Iterable
isinstance([1,2,3], Iterable)  # True
```

### 2. 迭代器（Iterator）

**一句话**：记住遍历位置，用 `next()` 取下一个

```python
# 创建迭代器
nums = [1, 2, 3]
it = iter(nums)  # 将可迭代对象转为迭代器

# 使用
next(it)  # 1
next(it)  # 2
next(it)  # 3
next(it)  # ❌ StopIteration（耗尽）

# 自定义迭代器
class Counter:
    def __init__(self, max):
        self.max = max
        self.n = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.n >= self.max:
            raise StopIteration
        self.n += 1
        return self.n
```

### 3. 生成器（Generator）

**一句话**：用 `yield` 创建的迭代器（惰性求值，省内存）

#### 两种创建方式

```python
# 方式1：生成器函数（用 yield）
def count_up(max):
    n = 0
    while n < max:
        yield n  # 暂停并返回值，下次从这继续
        n += 1

gen = count_up(3)
next(gen)  # 0
next(gen)  # 1
next(gen)  # 2

# 方式2：生成器表达式
squares = (x**2 for x in range(5))  # 注意是小括号
list(squares)  # [0, 1, 4, 9, 16]
```

#### yield 工作原理

```python
def gen_example():
    print("开始")
    yield 1        # 第1次 next()：执行到这，返回1，暂停
    print("继续")
    yield 2        # 第2次 next()：从暂停处继续，返回2，暂停
    print("结束")  # 第3次 next()：从暂停处继续，抛 StopIteration

g = gen_example()
next(g)  # 打印"开始"，返回1
next(g)  # 打印"继续"，返回2
next(g)  # 打印"结束"，抛 StopIteration
```

#### 生成器 vs 列表

| 维度 | 生成器 | 列表 |
|------|--------|------|
| 内存 | 惰性加载，省内存 | 一次性全加载 |
| 复用 | **只能遍历一次** | 可多次遍历 |
| 操作 | 不支持 `len()`、索引 | 支持 `len()`、索引 |
| 判空 | `if not gen` **始终 False** | 正确判空 |

#### 常见问题

```python
# 问题1：不能获取数量
files = path.glob("*.json")  # 返回生成器
len(files)  # ❌ TypeError

# 问题2：只能遍历一次
for f in files: pass
for f in files: pass  # ❌ 第二次不执行（已耗尽）

# 问题3：不能正确判空
if not files:  # ❌ 始终 False（生成器对象是 truthy）
    raise FileNotFoundError(...)

# 解决方案：转为列表
files = list(path.glob("*.json"))
```

#### 常见返回生成器的函数

| 函数 | 返回类型 |
|------|---------|
| `Path.glob()` / `Path.rglob()` | 生成器 |
| `range()` | range对象 |
| `map()` / `filter()` / `zip()` | 生成器对象 |

### 4. 异步生成器（async/await）

**一句话**：异步版本，用于 I/O 密集型场景

#### 核心关键字

| 关键字 | 作用 | 示例 |
|--------|------|------|
| `async def` | 定义异步函数/生成器 | `async def fetch(): ...` |
| `await` | 等待异步操作完成 | `result = await fetch()` |
| `async for` | 遍历异步生成器 | `async for item in agen:` |
| `yield` | 异步生成值 | `yield data` |

#### 示例

```python
import asyncio

# 异步生成器
async def async_count(max):
    for i in range(max):
        await asyncio.sleep(0.1)  # 异步等待
        yield i

# 使用
async def main():
    async for num in async_count(3):
        print(num)

asyncio.run(main())  # 0, 1, 2
```

#### 同步 vs 异步对比

| 同步 | 异步 |
|------|------|
| `def func():` | `async def func():` |
| `yield value` | `yield value` |
| `next(gen)` | `await gen.__anext__()` |
| `for x in gen:` | `async for x in gen:` |

### 知识结构图

```
迭代器与生成器系统
├── 可迭代对象（Iterable）
│   └── 能被 for 遍历（list/tuple/str/dict...）
├── 迭代器（Iterator）
│   ├── iter() 创建
│   ├── next() 取值
│   └── StopIteration 结束
├── 生成器（Generator）
│   ├── yield 创建（惰性求值）
│   ├── 生成器表达式：(x for x in ...)
│   ├── 只能遍历一次
│   └── 省内存
└── 异步生成器（AsyncGenerator）
    ├── async def + yield
    ├── async for 遍历
    └── await 等待
```

### 记忆口诀

```
iter 转迭代，next 取下一个
yield 暂停返，下次接着走
生成器省内存，只能走一遍
async 异步版，await 等结果
```

### 快速决策

```
需要遍历大量数据？
├── 数据太大一次放不下 → 生成器（yield）
├── 需要自定义遍历逻辑 → 迭代器（__iter__ + __next__）
├── 需要 len/索引/多次遍历 → list() 转换
└── I/O 密集型/并发 → async/await
```

### 使用场景

| 类型 | 使用场景 | 典型例子 |
|------|---------|---------|
| **生成器** | 数据太大一次放不下 | 大文件逐行读、分页查询、LLM流式输出 |
| **迭代器** | 自定义遍历逻辑 | 自定义数据结构遍历、框架开发 |
| **列表** | 数据量小，需多次遍历 | 普通数据处理 |

---

## 函数默认参数陷阱

### 核心规则（一句话）

**可变对象（list/dict/set）作为默认参数，必须用 `None`，函数内再创建**

### 对比表

| 参数类型 | 默认值写法 | 原因 |
|----------|-----------|------|
| 可变对象（`list`/`dict`/`set`） | `= None` | 可变对象会被所有调用共享 |
| 不可变对象（`str`/`int`/`tuple`） | `= ""`/`= 0`/`= ()` | 不可变，不会共享 |

### 代码示例

```python
# ❌ 错误写法（经典陷阱）
def bad(items=[]):
    items.append(1)
    return items

bad()  # [1]
bad()  # [1, 1]  ← 糟糕！共享了同一个list
bad()  # [1, 1, 1]

# ✅ 正确写法
def good(items=None):
    if items is None:
        items = []
    items.append(1)
    return items

good()  # [1]
good()  # [1]  ← 每次都是新list
```

### 标准模式

```python
# list参数
def func(items: list = None):
    if items is None:
        items = []

# dict参数
def func(data: dict = None):
    if data is None:
        data = {}

# set参数
def func(tags: set = None):
    if tags is None:
        tags = set()

# 不可变对象可以直接写
def func(name: str = "", count: int = 0):  # ✅ 没问题
```

### 记忆口诀

```
可变默认用None
函数内再创建
list/dict/set要小心
str/int/tuple没问题
```

---

## JSON 操作系统

### 核心规则（一句话）

**`json.load/dump` 操作文件，`json.loads/dumps` 操作字符串（s = string）**

### 对比表

| 方法 | 输入 | 输出 | 用途 |
|------|------|------|------|
| `json.load(f)` | 文件对象 | dict/list | 从文件读取JSON |
| `json.loads(s)` | 字符串 | dict/list | 从字符串解析JSON |
| `json.dump(obj, f)` | 对象+文件 | None | 写入JSON到文件 |
| `json.dumps(obj)` | 对象 | 字符串 | 对象转JSON字符串 |

### 常用参数

| 参数 | 作用 | 示例 |
|------|------|------|
| `ensure_ascii=False` | 保留中文（不转\uXXXX） | `json.dumps(obj, ensure_ascii=False)` |
| `indent=2` | 格式化缩进 | `json.dumps(obj, indent=2)` |
| `encoding="utf-8"` | 文件编码（open时指定） | `open(f, "r", encoding="utf-8")` |

### 代码示例

```python
import json

# 读文件
with open("data.json", "r", encoding="utf-8") as f:
    data = json.load(f)  # dict

# 写文件
with open("out.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

# 字符串操作
json_str = json.dumps(data, ensure_ascii=False)  # dict → str
data = json.loads(json_str)  # str → dict
```

### 注意事项

- `json.load()` 读的是**文件内容格式**，不是文件扩展名
- 只要内容是合法JSON，文件叫 `.txt`、`.dat` 都能读
- 中文必须加 `ensure_ascii=False`，否则变成 `\uXXXX`

### 知识结构图

```
JSON操作系统
├── 文件操作（load/dump）
│   ├── json.load(f) → dict（读）
│   └── json.dump(obj, f)（写）
└── 字符串操作（loads/dumps）
    ├── json.loads(s) → dict（解析）
    └── json.dumps(obj) → str（序列化）
```

### 记忆口诀

```
load/dump 操作文件
loads/dumps 操作字符串
s = string 记住就行
中文必加 ensure_ascii=False
```

---

## 后续知识扩展区

（待补充）

