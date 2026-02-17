import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from Tools.tools import get_basic
from model import KronosPredictor


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


# 回测
def back_test(df=None, pred_len=20, show=False):
    tokenizer, model = get_basic()

    # 2. Instantiate Predictor
    predictor = KronosPredictor(model, tokenizer, max_context=512)

    # 3. Prepare Data
    df['timestamps'] = pd.to_datetime(df['timestamps'])

    lookback = 512

    x_df = df.loc[:lookback - 1, ['open', 'high', 'low', 'close', 'volume', 'amount']]
    x_timestamp = df.loc[:lookback - 1, 'timestamps']
    y_timestamp = df.loc[lookback:lookback + pred_len - 1, 'timestamps']

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
        plot_prediction(kline_df, pred_df)
    else:
        return pred_df['close'].iloc[-1] > df['close'].iloc[-1]


if __name__ == '__main__':
    back_test()
