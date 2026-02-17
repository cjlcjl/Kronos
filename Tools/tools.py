import json

import pandas as pd
import pandas_market_calendars as mcal
import torch
from safetensors.torch import load_file

from model import Kronos, KronosTokenizer


def get_basic():
    # 1. 设置本地路径 (请确保路径正确)
    tokenizer_path = r"D:\Work\Kronos_Data\tokenizer-base"
    model_path = r"D:\Work\Kronos_Data\base"

    # 2. 加载分词器 (解决那16个参数报错的关键)
    with open(f"{tokenizer_path}/config.json", 'r') as f:
        t_config = json.load(f)
    # 用 ** 解包 JSON 里的参数，填补代码里缺少的 16 个参数
    tokenizer = KronosTokenizer(**t_config)
    tokenizer.load_state_dict(load_file(f"{tokenizer_path}/model.safetensors"))

    # 3. 加载预测模型
    with open(f"{model_path}/config.json", 'r') as f:
        m_config = json.load(f)
    model = Kronos(**m_config)
    model.load_state_dict(load_file(f"{model_path}/model.safetensors"))

    # 4. 移动到显卡 (因为你是 CUDA: True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer.to(device)
    model.to(device)

    # print("✅ 恭喜！模型和分词器已成功离线加载到显卡！")

    return tokenizer, model


def get_china_future_timestamps(start_time, periods):
    # 1. 确保 start_time 是 Timestamp 格式
    start_time = pd.to_datetime(start_time)

    # 【核心修复】如果 CSV 时间没时区，我们假设它是北京时间，给它打上 +08:00 的标签
    if start_time.tz is None:
        start_time = start_time.tz_localize('Asia/Shanghai')

    # 2. 获取上证交易所日历
    sse = mcal.get_calendar('XSHG')

    # 3. 这里的 start_date 需要是日期字符串
    schedule = sse.schedule(start_date=start_time.date(),
                            end_date=(start_time + pd.Timedelta(days=30)).date())

    # 4. 生成 5 分钟交易序列 (返回的是 UTC 时区的时间)
    all_times = mcal.date_range(schedule, frequency='5min')

    # 【核心修复】将生成的序列全部转为北京时间进行比较
    all_times_sh = all_times.tz_convert('Asia/Shanghai')

    # 5. 过滤掉过去的时间，只取未来的
    future_times = [t for t in all_times_sh if t > start_time]

    # 6. 取够 periods 个数，并为了画图方便，去掉时区标签（转回 naive 格式）
    res = pd.to_datetime(future_times[:periods]).tz_localize(None)

    return res


def get_china_future_timestampsV2(start_time, periods, freq='5min'):
    """
    支持多种频率的未来交易时间生成
    :param start_time: 起始时间
    :param periods: 预测步数
    :param freq: 频率，支持 '5min', '60min' (或 'H'), '1D' (或 'D')
    """
    start_time = pd.to_datetime(start_time)
    if start_time.tz is None:
        start_time = start_time.tz_localize('Asia/Shanghai')

    sse = mcal.get_calendar('XSHG')

    # 动态计算 end_date，防止 periods 很大时 30 天不够用
    # 日线需要的天数更多，5分钟线需要的天数较少
    days_buffer = periods * 2 if 'D' in freq.upper() else (periods // 4) + 30
    end_date = (start_time + pd.Timedelta(days=days_buffer)).date()

    schedule = sse.schedule(start_date=start_time.date(), end_date=end_date)

    # 1. 如果是日线频率
    if freq.upper() in ['1D', 'D']:
        # 直接获取交易日序列，取其开盘时间或只取日期
        all_times = schedule.index
        # schedule.index 默认是 DatetimeIndex (UTC)
        all_times_sh = all_times.tz_localize('UTC').tz_convert('Asia/Shanghai')

    # 2. 如果是分钟/小时频率
    else:
        # 使用 mcal.date_range 生成盘中细分序列
        # 注意：pandas_market_calendars 的 'H' 可能会包含中午休市时间，
        # 如果需要精准去掉 A 股 11:30-13:00，建议用 '60min'
        all_times = mcal.date_range(schedule, frequency=freq)
        all_times_sh = all_times.tz_convert('Asia/Shanghai')

    # 过滤掉过去的时间
    future_times = [t for t in all_times_sh if t > start_time]

    # 取够个数并去掉时区
    res = pd.to_datetime(future_times[:periods]).tz_localize(None)

    return res