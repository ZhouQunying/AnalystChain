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


def get_gdp_quarterly() -> Optional[pd.DataFrame]:
    """获取季度GDP数据

    从AKShare获取中国季度GDP数据，包括GDP绝对值和同比增长率。
    使用历史真实数据API，非预测数据。

    Returns:
        pd.DataFrame: 包含以下列的DataFrame：
            - quarter (str): 季度（格式：YYYY年第N季度）
            - gdp (float): GDP绝对值（亿元）
            - gdp_yoy (float): GDP同比增长率（%）
        如果获取失败，返回None

    Example:
        >>> df = get_gdp_quarterly()
        >>> print(df.head())
                  quarter       gdp  gdp_yoy
        0  2025年第1-3季度  1015036.1      5.2
        1  2025年第1-2季度   660535.8      5.3

    Raises:
        Exception: 数据获取或处理失败时打印错误信息
    """
    try:
        logger.info("开始获取GDP季度数据...")

        # 调用AKShare API（历史数据，非预测）
        df = ak.macro_china_gdp()

        # 检查数据是否为空
        if df.empty:
            logger.warning("获取的GDP数据为空")
            return None

        # 重命名列为英文
        df = df.rename(columns={
            '季度': 'quarter',
            '国内生产总值-绝对值': 'gdp',
            '国内生产总值-同比增长': 'gdp_yoy'
        })

        # 选择需要的列
        df = df[['quarter', 'gdp', 'gdp_yoy']]

        # 转换数据类型
        df['gdp'] = pd.to_numeric(df['gdp'], errors='coerce')
        df['gdp_yoy'] = pd.to_numeric(df['gdp_yoy'], errors='coerce')

        # 删除包含NaN的行
        df = df.dropna()

        # 按季度降序排序（最新数据在前）
        df = df.reset_index(drop=True)

        logger.info(f"成功获取GDP数据，共{len(df)}条记录")
        return df

    except Exception as e:
        logger.error(f"获取GDP数据失败: {e}")
        return None


def get_cpi_monthly() -> Optional[pd.DataFrame]:
    """获取月度CPI数据

    从AKShare获取中国月度CPI数据，包括同比增长率和环比增长率。
    使用历史真实数据API，非预测数据。

    Returns:
        pd.DataFrame: 包含以下列的DataFrame：
            - month (str): 月份（格式：YYYY年MM月份）
            - cpi_yoy (float): CPI同比增长率（%）
            - cpi_mom (float): CPI环比增长率（%）
            - cpi_ytd (float): CPI累计同比（%）
        如果获取失败，返回None

    Example:
        >>> df = get_cpi_monthly()
        >>> print(df.head())
                month  cpi_yoy  cpi_mom  cpi_ytd
        0  2025年11月份      0.7     -0.1     -0.1
        1  2025年10月份      0.2      0.2      0.0

    Raises:
        Exception: 数据获取或处理失败时打印错误信息
    """
    try:
        logger.info("开始获取CPI月度数据...")

        # 调用AKShare API（历史数据，非预测）
        df = ak.macro_china_cpi()

        # 检查数据是否为空
        if df.empty:
            logger.warning("获取的CPI数据为空")
            return None

        # 重命名列为英文
        df = df.rename(columns={
            '月份': 'month',
            '全国-同比增长': 'cpi_yoy',
            '全国-环比增长': 'cpi_mom',
            '全国-累计': 'cpi_ytd'
        })

        # 选择需要的列
        df = df[['month', 'cpi_yoy', 'cpi_mom', 'cpi_ytd']]

        # 转换数据类型
        df['cpi_yoy'] = pd.to_numeric(df['cpi_yoy'], errors='coerce')
        df['cpi_mom'] = pd.to_numeric(df['cpi_mom'], errors='coerce')
        df['cpi_ytd'] = pd.to_numeric(df['cpi_ytd'], errors='coerce')

        # 删除包含NaN的行
        df = df.dropna()

        # 数据已按月份降序排序（最新数据在前）
        df = df.reset_index(drop=True)

        logger.info(f"成功获取CPI数据，共{len(df)}条记录")
        return df

    except Exception as e:
        logger.error(f"获取CPI数据失败: {e}")
        return None


def get_pmi_manufacturing() -> Optional[pd.DataFrame]:
    """获取制造业PMI数据

    从AKShare获取中国官方制造业PMI指数（月度数据）。
    使用历史真实数据API，非预测数据。

    Returns:
        pd.DataFrame: 包含以下列的DataFrame：
            - month (str): 月份（格式：YYYY年MM月份）
            - pmi (float): 制造业PMI指数
            - pmi_yoy (float): PMI同比增长率（%）
        如果获取失败，返回None

    Example:
        >>> df = get_pmi_manufacturing()
        >>> print(df.head())
                month   pmi  pmi_yoy
        0  2025年11月份  49.2    -2.19
        1  2025年10月份  49.0    -2.20

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

        # 调用AKShare API（历史数据，非预测）
        df = ak.macro_china_pmi()

        # 检查数据是否为空
        if df.empty:
            logger.warning("获取的PMI数据为空")
            return None

        # 重命名列为英文
        df = df.rename(columns={
            '月份': 'month',
            '制造业-指数': 'pmi',
            '制造业-同比增长': 'pmi_yoy'
        })

        # 选择需要的列
        df = df[['month', 'pmi', 'pmi_yoy']]

        # 转换数据类型
        df['pmi'] = pd.to_numeric(df['pmi'], errors='coerce')
        df['pmi_yoy'] = pd.to_numeric(df['pmi_yoy'], errors='coerce')

        # 删除包含NaN的行
        df = df.dropna()

        # 数据已按月份降序排序（最新数据在前）
        df = df.reset_index(drop=True)

        logger.info(f"成功获取PMI数据，共{len(df)}条记录")
        return df

    except Exception as e:
        logger.error(f"获取PMI数据失败: {e}")
        return None


if __name__ == '__main__':
    # 测试各个函数
    print("=" * 50)
    print("测试GDP季度数据获取")
    print("=" * 50)
    gdp_df = get_gdp_quarterly()
    if gdp_df is not None:
        print(gdp_df.head())
        print(f"\n数据形状: {gdp_df.shape}")
        print(f"数据类型:\n{gdp_df.dtypes}\n")

    print("=" * 50)
    print("测试CPI月度数据获取")
    print("=" * 50)
    cpi_df = get_cpi_monthly()
    if cpi_df is not None:
        print(cpi_df.head())
        print(f"\n数据形状: {cpi_df.shape}")
        print(f"数据类型:\n{cpi_df.dtypes}\n")

    print("=" * 50)
    print("测试PMI数据获取")
    print("=" * 50)
    pmi_df = get_pmi_manufacturing()
    if pmi_df is not None:
        print(pmi_df.head())
        print(f"\n数据形状: {pmi_df.shape}")
        print(f"数据类型:\n{pmi_df.dtypes}\n")

    print("=" * 50)
    print("所有测试完成！")
    print("=" * 50)

