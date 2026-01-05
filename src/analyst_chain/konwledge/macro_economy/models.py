from enum import IntEnum
from dataclasses import dataclass

class FilePriority(IntEnum):
    """文件优先级枚举

    优先级规则：
    - PDF笔记最优先（最详细完整，包含完整知识点）
    - Word文档次之（格式规范，信息完整）
    - 普通PDF第三（信息完整但可能不如笔记详细）
    - PPT最后（信息密度低，多为摘要）
    """
    PDF_NOTE = 1      # PDF笔记文件（文件名包含"笔记"）
    WORD_DOC = 2      # Word文档
    PDF_REGULAR = 3   # 普通PDF文件
    POWERPOINT = 4    # PowerPoint文件
    UNKNOWN = 99      # 未知类型


@dataclass
class FileInfo:
    """文件信息数据类

    统一管理文件，便于按序号分组、按优先级排序、按相似度匹配。

    Attributes:
        path: 完整路径
        original_name: 原始文件名
        cleaned_name: 清洗后的文件名（去除噪音）
        sequence: 序号（整数，用于分组）
        sequence_str: 序号字符串（用于显示）
        priority: 文件优先级（用于排序和选择最佳文件）
    """
    path: str
    original_name: str
    cleaned_name: str
    sequence: int
    sequence_str: str
    priority: FilePriority
