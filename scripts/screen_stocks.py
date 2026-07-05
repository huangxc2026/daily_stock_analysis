#!/usr/bin/env python3
"""
A股量化筛选脚本 - 从沪深300+中证500中筛选最值得关注的股票
用于 daily_stock_analysis 工作流的前置筛选步骤
"""

import os
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import akshare as ak
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── 配置 ──────────────────────────────────────────────
TOP_N = int(os.environ.get("SCREEN_TOP_N", "10"))       # 输出前N只
WORKERS = int(os.environ.get("SCREEN_WORKERS", "10"))    # 并行线程数
HISTORY_DAYS = 60                                         # 拉取历史天数(确保30+交易日)
INDICES = ["000300", "000905"]                            # 沪深300 + 中证500


def get_index_components():
    """获取沪深300+中证500成分股代码列表"""
    all_codes = set()
    for idx in INDICES:
        try:
            df = ak.index_stock_cons_csindex(symbol=idx)
            codes = df["成分券代码"].tolist()
            all_codes.update(codes)
            print(f"  {idx}: 获取到 {len(codes)} 只成分股")
        except Exception as e:
            print(f"  {idx}: 获取失败 - {e}")
            # 备用方法
            try:
                if idx == "000300":
                    df = ak.index_stock_cons(symbol="sh000300")
                else:
                    df = ak.index_stock_cons(symbol="sh000905")
                codes = df["品种代码"].tolist()
                all_codes.update(codes)
                print(f"  {idx}: 备用方法获取到 {len(codes)} 只")
            except Exception as e2:
                print(f"  {idx}: 备用方法也失败 - {e2}")
    print(f"合计 {len(all_codes)} 只不重复成分股")
    return sorted(all_codes)


def fetch_stock_history(code, end_date, start_date):
    """获取单只股票的历史K线数据"""
    try:
        df = ak.stock_zh_a_hist(
            symbol=code,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust="qfq",  # 前复权
        )
        if df is not None and len(df) >= 20:
            return code, df
    except Exception:
        pass
    return code, None


def calculate_rsi(series, period=14):
    """计算RSI指标"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calculate_macd(close, fast=12, slow=26, signal=9):
    """计算MACD指标"""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    macd_hist = (dif - dea) * 2
    return dif, dea, macd_hist


def score_stock(df):
    """
    对单只股票进行技术面评分（满分100）
    指标：均线趋势(25) + 放量突破(25) + MACD(20) + RSI(15) + 短期动量(15)
    """
    if df is None or len(df) < 20:
        return -1

    close = df["收盘"].astype(float)
    volume = df["成交量"].astype(float)
    score = 0

    # ── 1. 均线趋势 (25分) ──
    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    latest_close = close.iloc[-1]
    latest_ma5 = ma5.iloc[-1]
    latest_ma20 = ma20.iloc[-1]

    if latest_close > latest_ma5 > latest_ma20:
        score += 25  # 完美多头排列
    elif latest_close > latest_ma20:
        score += 15  # 价格在20日均线上方
    elif latest_close > latest_ma5:
        score += 10  # 短期回暖
    else:
        score += 0

    # ── 2. 放量突破 (25分) ──
    vol_5avg = volume.rolling(5).mean().iloc[-2]  # 前5日均量(不含今天)
    vol_today = volume.iloc[-1]
    if vol_5avg > 0:
        vol_ratio = vol_today / vol_5avg
        if vol_ratio >= 2.0:
            score += 25  # 显著放量
        elif vol_ratio >= 1.5:
            score += 20
        elif vol_ratio >= 1.2:
            score += 15
        elif vol_ratio >= 1.0:
            score += 8
        else:
            score += 0

    # ── 3. MACD (20分) ──
    dif, dea, _ = calculate_macd(close)
    # 近3日是否金叉
    if len(dif) >= 4:
        golden_cross = False
        for i in range(-3, 0):
            if dif.iloc[i - 1] < dea.iloc[i - 1] and dif.iloc[i] >= dea.iloc[i]:
                golden_cross = True
                break
        if golden_cross:
            score += 20  # 近3日金叉
        elif dif.iloc[-1] > dea.iloc[-1] and dif.iloc[-1] > 0:
            score += 12  # DIF在DEA上方且为正
        elif dif.iloc[-1] > dea.iloc[-1]:
            score += 6

    # ── 4. RSI (15分) ──
    rsi = calculate_rsi(close, 14)
    latest_rsi = rsi.iloc[-1] if not rsi.empty else 50
    if 40 <= latest_rsi <= 60:
        score += 15  # RSI适中，不过热不过冷
    elif 30 <= latest_rsi < 40 or 60 < latest_rsi <= 70:
        score += 10
    elif latest_rsi < 30:
        score += 5   # 超卖，可能有反弹
    else:
        score += 0   # 超买

    # ── 5. 短期动量 (15分) ──
    if len(close) >= 6:
        pct_5d = (close.iloc[-1] / close.iloc[-6] - 1) * 100
        if 0 < pct_5d <= 5:
            score += 15  # 温和上涨
        elif 5 < pct_5d <= 10:
            score += 10  # 较强上涨
        elif -3 < pct_5d <= 0:
            score += 8   # 小幅回调
        elif pct_5d > 10:
            score += 3   # 短期涨幅过大，追高风险
        else:
            score += 0

    return score


def main():
    print("=" * 60)
    print(f"A股量化筛选 | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"目标: 从沪深300+中证500中筛选 TOP {TOP_N}")
    print("=" * 60)

    # 1. 获取成分股
    print("\n[1/4] 获取指数成分股...")
    codes = get_index_components()
    if not codes:
        print("错误: 无法获取成分股列表")
        sys.exit(1)

    # 2. 并行拉取历史数据
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=HISTORY_DAYS)).strftime("%Y%m%d")
    print(f"\n[2/4] 拉取K线数据 ({start_date} ~ {end_date})...")

    stock_data = {}
    failed = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {
            executor.submit(fetch_stock_history, code, end_date, start_date): code
            for code in codes
        }
        done_count = 0
        for future in as_completed(futures):
            done_count += 1
            code, df = future.result()
            if df is not None:
                stock_data[code] = df
            else:
                failed += 1
            if done_count % 100 == 0:
                print(f"  进度: {done_count}/{len(codes)} (成功: {len(stock_data)})")

    print(f"  完成: 成功 {len(stock_data)} 只, 失败 {failed} 只")

    # 3. 评分排名
    print(f"\n[3/4] 技术面评分...")
    results = []
    for code, df in stock_data.items():
        s = score_stock(df)
        if s >= 0:
            # 获取股票名称
            name = ""
            try:
                name = df.attrs.get("name", "")
            except Exception:
                pass
            latest_close = float(df["收盘"].iloc[-1])
            pct_change = float(df["涨跌幅"].iloc[-1]) if "涨跌幅" in df.columns else 0
            results.append({
                "code": code,
                "name": name,
                "close": latest_close,
                "pct_change": pct_change,
                "score": s,
            })

    results.sort(key=lambda x: x["score"], reverse=True)

    # 4. 输出结果
    print(f"\n[4/4] 筛选结果 TOP {TOP_N}:")
    print("-" * 55)
    print(f"{'排名':>4}  {'代码':<8} {'现价':>8} {'涨跌':>7} {'评分':>4}")
    print("-" * 55)

    top_stocks = results[:TOP_N]
    for i, s in enumerate(top_stocks, 1):
        print(f"  {i:>2}.  {s['code']:<8} {s['close']:>8.2f}  {s['pct_change']:>+6.2f}%  {s['score']:>3}")

    print("-" * 55)

    # 输出代码列表
    screened_codes = ",".join(s["code"] for s in top_stocks)
    print(f"\n筛选代码: {screened_codes}")

    # 写入文件（供 workflow 读取）
    output_file = os.environ.get("SCREEN_OUTPUT_FILE", "screened_stocks.txt")
    with open(output_file, "w") as f:
        f.write(screened_codes)
    print(f"已写入: {output_file}")

    # GitHub Actions output
    gh_output = os.environ.get("GITHUB_OUTPUT")
    if gh_output:
        with open(gh_output, "a") as f:
            f.write(f"screened_list={screened_codes}\n")
        print("已写入 GITHUB_OUTPUT")

    return screened_codes


if __name__ == "__main__":
    main()
