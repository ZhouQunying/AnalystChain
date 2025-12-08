"""
AKShare工具测试脚本

测试akshare_tools.py中的3个核心函数
"""

import sys
from pathlib import Path

# 添加tools目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from akshare_tools import get_gdp_yearly, get_cpi_monthly, get_pmi_manufacturing


def test_get_gdp_yearly():
    """测试GDP数据获取"""
    print("\n" + "=" * 60)
    print("测试1: get_gdp_yearly()")
    print("=" * 60)
    
    df = get_gdp_yearly()
    
    if df is None:
        print("❌ 测试失败：返回None")
        return False
    
    # 检查DataFrame不为空
    assert not df.empty, "DataFrame不应为空"
    print(f"✅ 数据行数: {len(df)}")
    
    # 检查列名
    expected_columns = ['year', 'gdp', 'gdp_yoy']
    assert list(df.columns) == expected_columns, f"列名应为{expected_columns}"
    print(f"✅ 列名正确: {list(df.columns)}")
    
    # 检查数据类型
    assert df['year'].dtype == 'int64', "year应为int类型"
    assert df['gdp'].dtype == 'float64', "gdp应为float类型"
    assert df['gdp_yoy'].dtype == 'float64', "gdp_yoy应为float类型"
    print("✅ 数据类型正确")
    
    # 检查数据范围
    assert df['year'].min() > 1900, "年份应大于1900"
    assert df['gdp'].min() > 0, "GDP应大于0"
    print("✅ 数据范围合理")
    
    # 显示最新数据
    print(f"\n最新5条数据:")
    print(df.head())
    
    print("\n✅ GDP数据测试通过！")
    return True


def test_get_cpi_monthly():
    """测试CPI数据获取"""
    print("\n" + "=" * 60)
    print("测试2: get_cpi_monthly()")
    print("=" * 60)
    
    df = get_cpi_monthly()
    
    if df is None:
        print("❌ 测试失败：返回None")
        return False
    
    # 检查DataFrame不为空
    assert not df.empty, "DataFrame不应为空"
    print(f"✅ 数据行数: {len(df)}")
    
    # 检查列名
    expected_columns = ['month', 'cpi_yoy', 'cpi_ytd']
    assert list(df.columns) == expected_columns, f"列名应为{expected_columns}"
    print(f"✅ 列名正确: {list(df.columns)}")
    
    # 检查数据类型
    assert df['month'].dtype == 'object', "month应为字符串类型"
    assert df['cpi_yoy'].dtype == 'float64', "cpi_yoy应为float类型"
    assert df['cpi_ytd'].dtype == 'float64', "cpi_ytd应为float类型"
    print("✅ 数据类型正确")
    
    # 检查月份格式（YYYY-MM 或 YYYYMM）
    first_month = df['month'].iloc[0]
    assert len(str(first_month)) >= 6, "月份格式应为YYYY-MM或YYYYMM"
    print(f"✅ 月份格式: {first_month}")
    
    # 显示最新数据
    print(f"\n最新5条数据:")
    print(df.head())
    
    print("\n✅ CPI数据测试通过！")
    return True


def test_get_pmi_manufacturing():
    """测试PMI数据获取"""
    print("\n" + "=" * 60)
    print("测试3: get_pmi_manufacturing()")
    print("=" * 60)
    
    df = get_pmi_manufacturing()
    
    if df is None:
        print("❌ 测试失败：返回None")
        return False
    
    # 检查DataFrame不为空
    assert not df.empty, "DataFrame不应为空"
    print(f"✅ 数据行数: {len(df)}")
    
    # 检查列名
    expected_columns = ['month', 'pmi']
    assert list(df.columns) == expected_columns, f"列名应为{expected_columns}"
    print(f"✅ 列名正确: {list(df.columns)}")
    
    # 检查数据类型
    assert df['month'].dtype == 'object', "month应为字符串类型"
    assert df['pmi'].dtype == 'float64', "pmi应为float类型"
    print("✅ 数据类型正确")
    
    # 检查PMI范围（通常在0-100之间）
    assert df['pmi'].min() > 0 and df['pmi'].max() < 100, "PMI应在0-100之间"
    print(f"✅ PMI范围合理: {df['pmi'].min():.1f} - {df['pmi'].max():.1f}")
    
    # 显示最新数据
    print(f"\n最新5条数据:")
    print(df.head())
    
    print("\n✅ PMI数据测试通过！")
    return True


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("开始AKShare工具测试")
    print("=" * 60)
    
    results = []
    
    # 测试1: GDP
    try:
        results.append(('GDP', test_get_gdp_yearly()))
    except Exception as e:
        print(f"❌ GDP测试异常: {e}")
        results.append(('GDP', False))
    
    # 测试2: CPI
    try:
        results.append(('CPI', test_get_cpi_monthly()))
    except Exception as e:
        print(f"❌ CPI测试异常: {e}")
        results.append(('CPI', False))
    
    # 测试3: PMI
    try:
        results.append(('PMI', test_get_pmi_manufacturing()))
    except Exception as e:
        print(f"❌ PMI测试异常: {e}")
        results.append(('PMI', False))
    
    # 总结测试结果
    print("\n" + "=" * 60)
    print("测试结果总结")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name:10s}: {status}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！akshare_tools.py工作正常")
        return True
    else:
        print(f"\n⚠️  有{total - passed}个测试失败，请检查")
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)

