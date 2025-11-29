import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei"]
plt.rcParams["axes.unicode_minus"] = False

# 设置配色方案
colors = ["#4080FF", "#57A9FB", "#37D4CF", "#23C343", "#FBE842", "#FF9A2E", "#A9AEB8"]

def plot_gdp(df):
    plt.figure(figsize=(12, 6))
    plt.plot(df["日期"].astype(str), df["GDP_亿元"], marker="o", color=colors[0], label="GDP（亿元）")
    plt.title("中国年度GDP总量变化趋势")
    plt.xlabel("年份")
    plt.ylabel("GDP（亿元）")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.savefig("../images/gdp_trend.png")
    plt.close()

def plot_cpi(df):
    plt.figure(figsize=(12, 6))
    plt.plot(df["日期"].astype(str), df["CPI_同比"], marker="o", color=colors[1], label="CPI同比（%）")
    plt.title("中国CPI同比变化趋势")
    plt.xlabel("月份")
    plt.ylabel("CPI同比（%）")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.savefig("../images/cpi_trend.png")
    plt.close()

def plot_pmi(df):
    plt.figure(figsize=(12, 6))
    plt.plot(df["日期"].astype(str), df["PMI"], marker="o", color=colors[2], label="PMI")
    plt.title("中国制造业PMI变化趋势")
    plt.xlabel("月份")
    plt.ylabel("PMI")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.savefig("../images/pmi_trend.png")
    plt.close()

if __name__ == "__main__":
    gdp_df = pd.read_csv("../data/gdp_yearly_processed.csv")
    cpi_df = pd.read_csv("../data/cpi_processed.csv")
    pmi_df = pd.read_csv("../data/pmi_processed.csv")

    plot_gdp(gdp_df)
    plot_cpi(cpi_df)
    plot_pmi(pmi_df)

    print("图表生成完成。")


