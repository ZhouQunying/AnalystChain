
import akshare as ak
import pandas as pd

def get_gdp_data():
    # 获取中国年度GDP数据
    gdp_yearly = ak.macro_china_gdp_yearly()
    gdp_yearly.rename(columns={
        '年份': '日期',
        '今值': 'GDP_亿元',
        '第一产业': '第一产业_亿元',
        '第二产业': '第二产业_亿元',
        '第三产业': '第三产业_亿元'
    }, inplace=True)
    gdp_yearly['日期'] = pd.to_datetime(gdp_yearly['日期'], format='%Y')
    gdp_yearly['日期'] = gdp_yearly['日期'].dt.to_period('Y')
    return gdp_yearly

def get_cpi_data():
    # 获取中国CPI年度数据
    cpi_yearly = ak.macro_china_cpi_yearly()
    cpi_yearly.rename(columns={
        '月份': '日期',
        '今值': 'CPI_同比'
    }, inplace=True)
    cpi_yearly['日期'] = pd.to_datetime(cpi_yearly['日期'], format='%Y%m')
    cpi_yearly['日期'] = cpi_yearly['日期'].dt.to_period('M')
    return cpi_yearly

def get_pmi_data():
    # 获取中国官方制造业PMI数据
    pmi_data = ak.macro_china_pmi_yearly()
    pmi_data.rename(columns={
        '月份': '日期',
        '今值': 'PMI'
    }, inplace=True)
    pmi_data['日期'] = pd.to_datetime(pmi_data['日期'], format='%Y%m')
    pmi_data['日期'] = pmi_data['日期'].dt.to_period('M')
    return pmi_data

if __name__ == '__main__':
    gdp_yearly_df = get_gdp_data()
    cpi_df = get_cpi_data()
    pmi_df = get_pmi_data()

    # 将所有数据保存到CSV文件
    gdp_yearly_df.to_csv('../data/gdp_yearly.csv', index=False)
    cpi_df.to_csv('../data/cpi.csv', index=False)
    pmi_df.to_csv('../data/pmi.csv', index=False)

    print('数据获取完成并保存到CSV文件。')


