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
