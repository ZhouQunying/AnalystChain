import pandas as pd

def calculate_growth_rates(df, value_col, date_col):
    df = df.sort_values(by=date_col)
    df[f'{value_col}_同比'] = df[value_col].pct_change(periods=1) * 100
    return df

def calculate_gdp_structure(df):
    # 由于GDP三驾马车数据获取困难，暂时跳过此计算
    return df

def detect_inflection_points(series):
    # 简单的趋势拐点检测：当连续三个值趋势发生变化时
    diff = series.diff().dropna()
    inflection_points = []
    for i in range(1, len(diff) - 1):
        if (diff.iloc[i] > 0 and diff.iloc[i-1] < 0 and diff.iloc[i+1] > 0) or \
           (diff.iloc[i] < 0 and diff.iloc[i-1] > 0 and diff.iloc[i+1] < 0):
            inflection_points.append(series.index[i+1])
    return inflection_points

if __name__ == '__main__':
    # 读取数据
    gdp_yearly_df = pd.read_csv('gdp_yearly.csv')
    cpi_df = pd.read_csv('cpi.csv')
    pmi_df = pd.read_csv('pmi.csv')

    # 转换日期列为 Period 类型
    gdp_yearly_df['日期'] = pd.PeriodIndex(gdp_yearly_df['日期'], freq='Y')
    cpi_df['日期'] = pd.PeriodIndex(cpi_df['日期'], freq='M')
    pmi_df['日期'] = pd.PeriodIndex(pmi_df['日期'], freq='M')

    # 计算同比
    gdp_yearly_df = calculate_growth_rates(gdp_yearly_df, 'GDP_亿元', '日期')
    cpi_df = calculate_growth_rates(cpi_df, 'CPI_同比', '日期')
    pmi_df = calculate_growth_rates(pmi_df, 'PMI', '日期')

    # 检测趋势拐点
    gdp_inflection_points = detect_inflection_points(gdp_yearly_df['GDP_亿元'])
    cpi_inflection_points = detect_inflection_points(cpi_df['CPI_同比'])
    pmi_inflection_points = detect_inflection_points(pmi_df['PMI'])

    print("数据处理完成。")
    print("GDP 拐点: ", gdp_inflection_points)
    print("CPI 拐点: ", cpi_inflection_points)
    print("PMI 拐点: ", pmi_inflection_points)

    # 保存处理后的数据
    gdp_yearly_df.to_csv('gdp_yearly_processed.csv', index=False)
    cpi_df.to_csv('cpi_processed.csv', index=False)
    pmi_df.to_csv('pmi_processed.csv', index=False)


