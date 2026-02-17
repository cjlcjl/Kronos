import pandas as pd
import plotly.graph_objects as go

from Tools.tools import get_china_future_timestamps, get_basic
from model import KronosPredictor


def plot_prediction_interactive(history_df, pred_df, future_timestamps):
    # 1. 数据准备与格式化
    # 关键点：将时间转为字符串格式，这样 Plotly 才会把它们当成紧挨着的“标签”
    h_ts = pd.to_datetime(history_df['timestamps'])
    # 格式化为 月-日 时:分
    history_dates = h_ts.dt.strftime('%m-%d %H:%M').tolist()

    f_ts = pd.to_datetime(future_timestamps)
    future_dates = f_ts.strftime('%m-%d %H:%M').tolist()

    # 准备价格数据
    history_close = history_df['close']
    last_hist_date_label = history_dates[-1]
    last_hist_price = history_close.iloc[-1]

    # 为了连线不断开，把历史最后一个点作为预测的起点
    plot_pred_dates = [last_hist_date_label] + future_dates
    plot_pred_prices = [last_hist_price] + list(pred_df['close'].values)

    # 2. 创建画布
    fig = go.Figure()

    # 3. 添加历史曲线（蓝色）
    fig.add_trace(go.Scatter(
        x=history_dates,
        y=history_close,
        mode='lines',
        name='历史走势',
        line=dict(color='#1f77b4', width=2),
        hovertemplate='时间: %{x}<br>价格: %{y:.2f}<extra>历史</extra>'
    ))

    # 4. 添加预测曲线（红色虚线）
    fig.add_trace(go.Scatter(
        x=plot_pred_dates,
        y=plot_pred_prices,
        mode='lines',
        name='AI 预测',
        line=dict(color='#ff7f0e', width=2, dash='dash'),
        hovertemplate='时间: %{x}<br>价格: %{y:.2f}<extra>预测</extra>'
    ))

    # 5. 设置布局和关键的“分类轴”配置
    fig.update_layout(
        title='Kronos AI 股价预测（完全连续版）',
        xaxis_title='时间 (已剔除休市时段)',
        yaxis_title='价格',
        hovermode='x unified',
        template='plotly_white',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    # 【核心修复】强制 X 轴为分类模式，消除 15:00 到 09:30 的物理间隔
    fig.update_xaxes(
        type='category',
        tickangle=45,
        nticks=20  # 限制显示的标签数量，防止 X 轴文字重叠
    )

    # 6. 显示图表
    fig.show()


# 预测
def predict_future(df=None, pred_len=20, show=False):
    tokenizer, model = get_basic()

    # 2. Instantiate Predictor
    predictor = KronosPredictor(model, tokenizer, max_context=512)

    # 3. Prepare Data
    # df = pd.read_csv("./data/XSHG_5min_600977.csv")
    df = pd.read_csv("../examples/data/XSHG_5min_600977_V2.csv") if df is None else df
    df['timestamps'] = pd.to_datetime(df['timestamps'])

    lookback = 512

    x_df = df.loc[:lookback - 1, ['open', 'high', 'low', 'close', 'volume', 'amount']]
    x_timestamp = df.loc[:lookback - 1, 'timestamps']

    # --- 在你的主程序中这样调用 ---
    last_time = pd.to_datetime(df['timestamps'].iloc[-1])
    future_index = get_china_future_timestamps(last_time, pred_len)

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
    # print("Forecasted Finished")

    # 这里的 df 就是你的 513 条历史数据
    kline_df = df

    # 这里的 y_timestamp 是刚才我们用 pd.date_range 生成的那个未来时间
    # 如果你之前转换成了 Series，这里取 .values 或者直接传 Index 都可以
    # 建议直接传刚才生成的 future_index
    if show:
        plot_prediction_interactive(kline_df, pred_df, future_index)
    else:
        return pred_df['close'].iloc[-1] > df['close'].iloc[-1]


if __name__ == '__main__':
    result = predict_future(show=False)

    print(result)
