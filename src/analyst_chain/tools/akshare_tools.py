"""
AKShare宏观经济数据工具

提供GDP、CPI、PMI等宏观经济指标的数据获取功能
"""

import akshare as ak
import pandas as pd
from typing import Optional
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_gdp_yearly() -> Optional[pd.DataFrame]:
    """获取年度GDP数据
    
    从AKShare获取中国年度GDP数据，包括GDP绝对值和同比增长率。
    
    Returns:
        pd.DataFrame: 包含以下列的DataFrame：
            - year (int): 年份
            - gdp (float): GDP绝对值（亿元）
            - gdp_yoy (float): GDP同比增长率（%）
        如果获取失败，返回None
    
    Example:
        >>> df = get_gdp_yearly()
        >>> print(df.head())
           year     gdp  gdp_yoy
        0  2023  1260582    5.2
        1  2022  1210207    3.0
    
    Raises:
        Exception: 数据获取或处理失败时打印错误信息
    """
    try:
        logger.info("开始获取GDP年度数据...")
        
        # 调用AKShare API
        df = ak.macro_china_gdp_yearly()
        
        # 检查数据是否为空
        if df.empty:
            logger.warning("获取的GDP数据为空")
            return None
        
        # 重命名列为英文
        df = df.rename(columns={
            '年份': 'year',
            '国内生产总值-绝对值': 'gdp',
            '国内生产总值-同比增长': 'gdp_yoy'
        })
        
        # 选择需要的列（如果存在其他列）
        df = df[['year', 'gdp', 'gdp_yoy']]
        
        # 转换数据类型
        df['year'] = df['year'].astype(int)
        df['gdp'] = pd.to_numeric(df['gdp'], errors='coerce')
        df['gdp_yoy'] = pd.to_numeric(df['gdp_yoy'], errors='coerce')
        
        # 删除包含NaN的行
        df = df.dropna()
        
        # 按年份降序排序
        df = df.sort_values('year', ascending=False).reset_index(drop=True)
        
        logger.info(f"成功获取GDP数据，共{len(df)}条记录")
        return df
        
    except Exception as e:
        logger.error(f"获取GDP数据失败: {e}")
        return None


def get_cpi_monthly() -> Optional[pd.DataFrame]:
    """获取月度CPI数据
    
    从AKShare获取中国月度CPI数据，包括当月同比和累计同比。
    
    Returns:
        pd.DataFrame: 包含以下列的DataFrame：
            - month (str): 月份（格式：YYYY-MM）
            - cpi_yoy (float): CPI当月同比（%）
            - cpi_ytd (float): CPI累计同比（%）
        如果获取失败，返回None
    
    Example:
        >>> df = get_cpi_monthly()
        >>> print(df.head())
             month  cpi_yoy  cpi_ytd
        0  2024-11      0.2      0.3
        1  2024-10     -0.3      0.3
    
    Raises:
        Exception: 数据获取或处理失败时打印错误信息
    """
    try:
        logger.info("开始获取CPI月度数据...")
        
        # 调用AKShare API
        df = ak.macro_china_cpi_monthly()
        
        # 检查数据是否为空
        if df.empty:
            logger.warning("获取的CPI数据为空")
            return None
        
        # 重命名列为英文
        df = df.rename(columns={
            '月份': 'month',
            '全国当月': 'cpi_yoy',
            '全国累计': 'cpi_ytd'
        })
        
        # 选择需要的列
        df = df[['month', 'cpi_yoy', 'cpi_ytd']]
        
        # 转换数据类型
        df['cpi_yoy'] = pd.to_numeric(df['cpi_yoy'], errors='coerce')
        df['cpi_ytd'] = pd.to_numeric(df['cpi_ytd'], errors='coerce')
        
        # 删除包含NaN的行
        df = df.dropna()
        
        # 按月份降序排序
        df = df.sort_values('month', ascending=False).reset_index(drop=True)
        
        logger.info(f"成功获取CPI数据，共{len(df)}条记录")
        return df
        
    except Exception as e:
        logger.error(f"获取CPI数据失败: {e}")
        return None


def get_pmi_manufacturing() -> Optional[pd.DataFrame]:
    """获取制造业PMI数据
    
    从AKShare获取中国官方制造业PMI指数。
    
    Returns:
        pd.DataFrame: 包含以下列的DataFrame：
            - month (str): 月份（格式：YYYY-MM）
            - pmi (float): 制造业PMI指数
        如果获取失败，返回None
    
    Example:
        >>> df = get_pmi_manufacturing()
        >>> print(df.head())
             month    pmi
        0  2024-11  50.3
        1  2024-10  50.1
    
    Note:
        PMI指数说明：
        - PMI > 50: 制造业扩张
        - PMI = 50: 持平
        - PMI < 50: 制造业收缩
    
    Raises:
        Exception: 数据获取或处理失败时打印错误信息
    """
    try:
        logger.info("开始获取PMI数据...")
        
        # 调用AKShare API
        df = ak.macro_china_pmi_yearly()
        
        # 检查数据是否为空
        if df.empty:
            logger.warning("获取的PMI数据为空")
            return None
        
        # 重命名列为英文
        df = df.rename(columns={
            '月份': 'month',
            '制造业PMI': 'pmi'
        })
        
        # 选择需要的列
        df = df[['month', 'pmi']]
        
        # 转换数据类型
        df['pmi'] = pd.to_numeric(df['pmi'], errors='coerce')
        
        # 删除包含NaN的行
        df = df.dropna()
        
        # 按月份降序排序
        df = df.sort_values('month', ascending=False).reset_index(drop=True)
        
        logger.info(f"成功获取PMI数据，共{len(df)}条记录")
        return df
        
    except Exception as e:
        logger.error(f"获取PMI数据失败: {e}")
        return None


if __name__ == '__main__':
    # 测试各个函数
    print("=" * 50)
    print("测试GDP数据获取")
    print("=" * 50)
    gdp_df = get_gdp_yearly()
    if gdp_df is not None:
        print(gdp_df.head())
        print(f"\n数据类型:\n{gdp_df.dtypes}\n")
    
    print("=" * 50)
    print("测试CPI数据获取")
    print("=" * 50)
    cpi_df = get_cpi_monthly()
    if cpi_df is not None:
        print(cpi_df.head())
        print(f"\n数据类型:\n{cpi_df.dtypes}\n")
    
    print("=" * 50)
    print("测试PMI数据获取")
    print("=" * 50)
    pmi_df = get_pmi_manufacturing()
    if pmi_df is not None:
        print(pmi_df.head())
        print(f"\n数据类型:\n{pmi_df.dtypes}\n")
    
    print("=" * 50)
    print("所有测试完成！")
    print("=" * 50)

