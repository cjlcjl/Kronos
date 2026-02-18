import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from Tools.tools import get_basic, get_china_future_timestampsV2
from model import KronosPredictor


def evaluate_prediction(kline_df, pred_df, lookback=512):
    # 1. 数据切片
    # 历史最后一点的价格（作为计算涨跌幅的基准）
    last_close = kline_df['close'].iloc[lookback - 1]

    # 获取真实发生的未来价格序列
    actual_series = kline_df['close'].iloc[lookback:]
    # 获取预测的价格序列
    pred_series = pred_df['close']

    # 1. 计算终点涨跌幅(Gap)
    pred_change = (pred_series.iloc[-1] - last_close) / last_close
    actual_change = (actual_series.iloc[-1] - last_close) / last_close

    # 2. 计算偏差
    error_gap = pred_change - actual_change

    # 3. 计算平均偏离度 (MAPE)
    y_true = actual_series.values
    y_pred = pred_series.values
    # 过滤掉分母为 0 的情况（防止出现 inf）
    mask = y_true != 0
    # 公式: mean( |(真实 - 预测) / 真实| ) * 100
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    # 4. 计算方向是否正确 (Hit)
    is_correct = (pred_change * actual_change) > 0

    # 预测评估

    result = {
        '预测涨幅': f'{pred_change:.2%}',
        '实际涨幅': f'{actual_change:.2%}',
        '预测偏差(Gap)': f'{error_gap:.2%}',
        '平均偏离度(MAPE)': f'{mape:.2f}%',
        '预测方向是否正确': is_correct,
    }

    return result


def plot_prediction(kline_df, pred_df):
    # 1. 处理历史数据的时间戳
    # 确保转为 Datetime 格式
    h_ts = pd.to_datetime(kline_df['timestamps'])

    # 【核心修复】判断是 Series 还是 Index，并统一转为字符串列表
    if hasattr(h_ts, 'dt'):
        history_dates = h_ts.dt.strftime('%m-%d %H:%M').tolist()
    else:
        history_dates = h_ts.strftime('%m-%d %H:%M').tolist()

    # 2. 处理预测数据的时间戳 (pred_df.index 通常是 Index)
    f_ts = pd.to_datetime(pred_df.index)
    if hasattr(f_ts, 'dt'):
        future_dates = f_ts.dt.strftime('%m-%d %H:%M').tolist()
    else:
        future_dates = f_ts.strftime('%m-%d %H:%M').tolist()

    # 3. 构建“连线桥梁”
    # 取历史最后一个点的坐标
    last_date_label = history_dates[-1]
    last_close = kline_df['close'].iloc[-1]
    last_vol = kline_df['volume'].iloc[-1]

    # 预测数据点：桥梁点 + 未来点
    pred_x = [last_date_label] + future_dates
    pred_y_close = [last_close] + list(pred_df['close'].values)
    pred_y_vol = [last_vol] + list(pred_df['volume'].values)

    # 4. 创建子图 (2行1列)
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("价格走势预测", "成交量预测"),
        row_heights=[0.7, 0.3]
    )

    # --- 价格图层 ---
    fig.add_trace(go.Scatter(
        x=history_dates, y=kline_df['close'],
        mode='lines', name='历史价格', line=dict(color='#1f77b4', width=2)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=pred_x, y=pred_y_close,
        mode='lines', name='预测价格', line=dict(color='#ef553b', width=2, dash='dash')
    ), row=1, col=1)

    # --- 成交量图层 ---
    fig.add_trace(go.Bar(
        x=history_dates, y=kline_df['volume'],
        name='历史成交量', marker_color='#1f77b4', opacity=0.6
    ), row=2, col=1)

    fig.add_trace(go.Bar(
        x=pred_x, y=pred_y_vol,
        name='预测成交量', marker_color='#ef553b', opacity=0.6
    ), row=2, col=1)

    # 5. 【核心配置】强制 Categorical 模式，消除所有非交易时段空隙
    fig.update_xaxes(
        type='category',
        tickangle=45,
        nticks=20,  # 限制显示的刻度数量，防止挤在一起
        row=1, col=1
    )
    fig.update_xaxes(type='category', row=2, col=1)

    # 6. 总布局设置
    fig.update_layout(
        height=800,
        title_text="Kronos AI 5分钟线全连通预测",
        hovermode='x unified',
        template='plotly_white',
        showlegend=True
    )

    fig.show()


def plot_backtest_refined(kline_df, pred_df, period='1D', lookback=512):
    """
    专门用于回测展示：kline_df 包含历史和真实未来，pred_df 为预测值
    """
    import plotly.io as pio
    pio.renderers.default = 'browser'

    # 1. 强制按时间升序，防止连线乱跳
    kline_df = kline_df.sort_values('timestamps', ascending=True).copy()

    # 2. 自动切分数据
    # 前 512 条是模型看到的历史
    history_part = kline_df.iloc[:lookback]
    # 512 条之后的是真实的后续走势（Ground Truth）
    actual_part = kline_df.iloc[lookback:]

    # 3. 动态日期格式（加入 %Y 彻底解决你遇到的跨年乱码问题）
    date_format = '%Y-%m-%d' if 'D' in period.upper() else '%Y-%m-%d %H:%M'

    # 4. 准备时间轴标签
    h_ts = pd.to_datetime(history_part['timestamps'])
    history_dates = h_ts.dt.strftime(date_format).tolist()

    a_ts = pd.to_datetime(actual_part['timestamps'])
    actual_dates = a_ts.dt.strftime(date_format).tolist()

    # 预测数据的周期通常由 pred_df.index 决定
    f_ts = pd.to_datetime(pred_df.index)
    future_dates = f_ts.strftime(date_format).tolist()

    # 5. 构建“三线连接桥梁”
    # 所有的后续线（真实和预测）都从历史的最后一个点开始出发
    last_h_date = history_dates[-1]
    last_h_close = history_part['close'].iloc[-1]
    last_h_vol = history_part['volume'].iloc[-1]

    # 真实曲线连线
    actual_x = [last_h_date] + actual_dates
    actual_y_close = [last_h_close] + actual_part['close'].tolist()
    actual_y_vol = [last_h_vol] + actual_part['volume'].tolist()

    # 预测曲线连线
    pred_x = [last_h_date] + future_dates
    pred_y_close = [last_h_close] + pred_df['close'].tolist()
    pred_y_vol = [last_h_vol] + pred_df['volume'].tolist()

    # 6. 创建双子图
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("价格回测 (蓝色:历史 | 绿色:真实 | 红色:预测)", "成交量对比"),
        row_heights=[0.7, 0.3]
    )

    # --- 价格图层 ---
    fig.add_trace(
        go.Scatter(x=history_dates, y=history_part['close'], name='历史价格', line=dict(color='#1f77b4', width=2)),
        row=1, col=1)
    fig.add_trace(go.Scatter(x=actual_x, y=actual_y_close, name='真实走势', line=dict(color='#2ca02c', width=2)), row=1,
                  col=1)
    fig.add_trace(
        go.Scatter(x=pred_x, y=pred_y_close, name='AI 预测', line=dict(color='#ef553b', width=2, dash='dash')), row=1,
        col=1)

    # --- 成交量图层 ---
    fig.add_trace(go.Bar(x=history_dates, y=history_part['volume'], name='历史量', marker_color='#1f77b4', opacity=0.5),
                  row=2, col=1)
    fig.add_trace(go.Bar(x=actual_x, y=actual_y_vol, name='真实量', marker_color='#2ca02c', opacity=0.5), row=2, col=1)
    fig.add_trace(go.Bar(x=pred_x, y=pred_y_vol, name='预测量', marker_color='#ef553b', opacity=0.5), row=2, col=1)

    # 7. 核心配置：Categorical 模式 + 布局
    fig.update_xaxes(type='category', tickangle=45, nticks=15)
    fig.update_layout(height=800, title_text=f"Kronos 回测系统 - 周期: {period}", hovermode='x unified',
                      template='plotly_white')

    fig.show()


# 回测
def stock_back_test(df=None, pred_len=20, period='1D', show=False):
    tokenizer, model = get_basic()

    # 2. Instantiate Predictor
    predictor = KronosPredictor(model, tokenizer, max_context=512)

    # 3. Prepare Data
    df['timestamps'] = pd.to_datetime(df['timestamps'])

    lookback = 512

    x_df = df.loc[:lookback - 1, ['open', 'high', 'low', 'close', 'volume', 'amount']]
    x_timestamp = df.loc[:lookback - 1, 'timestamps']
    # y_timestamp = df.loc[lookback:lookback + pred_len - 1, 'timestamps']

    # --- 在你的主程序中这样调用 ---
    # last_time = pd.to_datetime(df['timestamps'].iloc[-1])
    last_time = pd.to_datetime(df.loc[:lookback]['timestamps'].iloc[-1])
    future_index = get_china_future_timestampsV2(last_time, pred_len, freq=period)

    # 3. 【关键修复】将 Index 转换为 Series，这样它就有了 .dt 属性
    y_timestamp = pd.Series(future_index)

    # 4. Make Prediction
    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=pred_len,
        T=1.0,  # (Temperature)：温度。数值越高，预测结果越“放飞自我”；数值越低，预测越保守。
        top_p=0.9,  # 核采样。只考虑概率最高的前 90% 的可能性，过滤掉极端的乱跳。
        sample_count=8,  # 只生成一种预测方案。如果你设为 5，它会给你出 5 条不同的连线。
        verbose=True
    )

    # 5. Visualize Results
    print("Forecasted Data Head:")
    print(pred_df.head())

    # Combine historical and forecasted data for plotting
    kline_df = df.loc[:lookback + pred_len - 1]

    # visualize
    if show:
        plot_backtest_refined(kline_df, pred_df, period=period, lookback=lookback)

    return evaluate_prediction(kline_df, pred_df, lookback)


if __name__ == '__main__':
    stock_back_test()
