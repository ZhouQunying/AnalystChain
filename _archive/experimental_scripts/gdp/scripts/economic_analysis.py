
import pandas as pd

def analyze_economic_cycle(gdp_df, cpi_df, pmi_df):
    # 获取最新数据点
    latest_gdp_growth = gdp_df["GDP_亿元_同比"].iloc[-1]
    latest_cpi = cpi_df["CPI_同比"].iloc[-1]
    latest_pmi = pmi_df["PMI"].iloc[-1]

    economic_stage = "未知"
    recommendations = {
        "国家层面": "",
        "企业层面": "",
        "投资者层面": "",
        "风控层面": ""
    }

    # 简化的经济周期判断逻辑
    # 扩张阶段：GDP增长，CPI温和上涨，PMI高于50
    if latest_gdp_growth > 0 and latest_cpi > 0 and latest_pmi > 50:
        economic_stage = "扩张阶段"
        recommendations["国家层面"] = "继续实施积极的财政政策和稳健的货币政策，关注通胀压力，适时进行微调。"
        recommendations["企业层面"] = "扩大产能，增加投资，抓住市场机遇。"
        recommendations["投资者层面"] = "积极配置股票等风险资产，关注成长性行业。"
        recommendations["风控层面"] = "保持适度杠杆，关注市场过热风险。"
    # 繁荣阶段：GDP高增长，CPI快速上涨，PMI高位运行
    elif latest_gdp_growth > 5 and latest_cpi > 3 and latest_pmi > 55:
        economic_stage = "繁荣阶段"
        recommendations["国家层面"] = "适度收紧货币政策，控制通胀，防止经济过热。"
        recommendations["企业层面"] = "优化产能结构，提高效率，警惕过度扩张。"
        recommendations["投资者层面"] = "逐步降低风险资产配置，关注价值股和防御性资产。"
        recommendations["风控层面"] = "严格控制杠杆，防范资产泡沫和信用风险。"
    # 衰退阶段：GDP负增长或低增长，CPI下降，PMI低于50
    elif latest_gdp_growth < 0 or (latest_gdp_growth < 2 and latest_pmi < 50):
        economic_stage = "衰退阶段"
        recommendations["国家层面"] = "实施宽松的财政政策和货币政策，刺激经济增长，稳定就业。"
        recommendations["企业层面"] = "削减成本，优化现金流，保持核心竞争力。"
        recommendations["投资者层面"] = "规避风险资产，关注债券、黄金等避险资产。"
        recommendations["风控层面"] = "加强风险管理，关注企业违约风险。"
    # 萧条阶段：GDP持续负增长，CPI持续下降，PMI持续低迷
    elif latest_gdp_growth < -2 and latest_cpi < -1 and latest_pmi < 45:
        economic_stage = "萧条阶段"
        recommendations["国家层面"] = "加大财政刺激力度，推出大规模基建项目，实行超宽松货币政策。"
        recommendations["企业层面"] = "保存实力，等待复苏，寻找新的增长点。"
        recommendations["投资者层面"] = "持有现金，等待市场底部，分批建仓。"
        recommendations["风控层面"] = "全面收缩，严控风险，避免破产。"
    # 复苏阶段：GDP增长，CPI低位回升，PMI回升至50附近
    elif latest_gdp_growth > 0 and latest_cpi < 1 and latest_pmi > 48 and latest_pmi < 52:
        economic_stage = "复苏阶段"
        recommendations["国家层面"] = "继续保持宽松政策，但需关注经济结构调整和新增长动能培育。"
        recommendations["企业层面"] = "逐步恢复生产，谨慎扩张，关注市场需求变化。"
        recommendations["投资者层面"] = "逐步增加风险资产配置，关注周期性行业。"
        recommendations["风控层面"] = "适度放松，但仍需警惕潜在风险。"

    return economic_stage, recommendations

if __name__ == '__main__':
    gdp_df = pd.read_csv('../data/gdp_yearly_processed.csv')
    cpi_df = pd.read_csv('../data/cpi_processed.csv')
    pmi_df = pd.read_csv('../data/pmi_processed.csv')

    economic_stage, recommendations = analyze_economic_cycle(gdp_df, cpi_df, pmi_df)

    with open('../reports/economic_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write(f"当前中国经济所处阶段：{economic_stage}\n\n")
        f.write("策略建议：\n")
        for level, advice in recommendations.items():
            f.write(f"- {level}：{advice}\n")

    print("经济分析报告生成完成：economic_analysis_report.txt")


