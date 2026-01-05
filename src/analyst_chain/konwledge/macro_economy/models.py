from enum import IntEnum

class FilePriority(IntEnum):
    """
    """
    PDF_NOTE = 1      # PDF笔记文件（文件名包含"笔记"）
    WORD_DOC = 2      # Word文档
    PDF_REGULAR = 3   # 普通PDF文件
    POWERPOINT = 4    # PowerPoint文件
    UNKNOWN = 99      # 未知类型
