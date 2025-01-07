import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import config
import math

input_file = config.input_file

# CSVファイルパス
file_paths = {
    "ファインチューニング前のCKA行列とファインチューニング前のモデルの精度を用いたKRRの散布図": f"outputs/{input_file}/krr_results/predicted_results.csv",
    "ファインチューニング後のCKA行列とファインチューニング後のモデルの精度を用いたKRRの散布図": f"outputs/{input_file}/krr_results/finetune_predicted_results.csv",
    "ファインチューニング前のCKA行列とファインチューニング前後のモデルの精度の差分を用いたKRRの散布図": f"outputs/{input_file}/krr_results/diff_predicted_results.csv",
    "ファインチューニング前のCKA行列とファインチューニング後のモデルの精度を用いたKRRの散布図": f"outputs/{input_file}/krr_results/predicted_results_before_cka_and_finetune_accuracy.csv",
}

def round_to_nearest(value, step):
    """
    指定したステップに基づいて値を丸める関数
    """
    return math.floor(value / step) * step if value < 0 else math.ceil(value / step) * step

def plot_scatter_with_error_lines(data):
    """
    散布図を作成し、誤差に基づいて色を変え、縦線・横線を追加し、
    メモリを0.2刻みで0を含むように調整する関数
    """
    # 誤差を計算
    data["Error"] = abs(data["True Value"] - data["Predicted Value"])

    # 散布図を作成
    fig = px.scatter(
        data,
        x="Predicted Value",
        y="True Value",
        hover_name="Model Name",  # カーソルを合わせるとモデル名を表示
        color="Error",  # 誤差に応じて色分け
        color_continuous_scale="Viridis",  # 色のスケールを指定
        labels={
            "Predicted Value": "予測精度",
            "True Value": "実際の精度",
            "Error": "誤差",
        },
    )

    # 縦線と横線を追加 (y=xの直線)
    min_val = min(data["Predicted Value"].min(), data["True Value"].min())
    max_val = max(data["Predicted Value"].max(), data["True Value"].max())

    # メモリ範囲を調整 (0を含む0.2刻み)
    grid_min = round_to_nearest(min_val, 0.2)
    grid_max = round_to_nearest(max_val, 0.2)
    if grid_min > 0:
        grid_min -= 0.2  # 0を含むように調整
    if grid_max < 0:
        grid_max += 0.2  # 0を含むように調整

    fig.add_trace(
        go.Scatter(
            x=[grid_min, grid_max],
            y=[grid_min, grid_max],
            mode="lines",
            line=dict(color="gray", dash="dash"),
            name="y=x (理想的な一致)",
        )
    )

    # グラフのスケールを統一し、メモリを設定
    fig.update_layout(
        xaxis=dict(
            scaleanchor="y",  # X軸とY軸のスケールを固定
            range=[grid_min, grid_max],  # スケール範囲を設定
            showgrid=True,  # グリッドラインを表示
            gridcolor="lightgray",  # グリッドの色
            gridwidth=0.5,  # グリッドの線幅
            tick0=0,  # メモリの開始位置
            dtick=0.2,  # メモリの区切り幅
        ),
        yaxis=dict(
            scaleanchor="x",  # Y軸とX軸のスケールを固定
            range=[grid_min, grid_max],  # スケール範囲を設定
            showgrid=True,  # グリッドラインを表示
            gridcolor="lightgray",  # グリッドの色
            gridwidth=0.5,  # グリッドの線幅
            tick0=0,  # メモリの開始位置
            dtick=0.2,  # メモリの区切り幅
        ),
        autosize=False,
        width=600,  # グラフの幅
        height=600,  # グラフの高さ（正方形）
        plot_bgcolor="white",  # 背景色を白に設定
    )

    return fig

def main():
    st.title("KRRによる精度の予測値と実際の精度の可視化")

    # 各CSVファイルを個別に処理
    for title, path in file_paths.items():
        st.subheader(title)

        try:
            # CSVファイルを読み込み
            data = pd.read_csv(path)

            # 必要な列が存在するか確認
            required_columns = {"Split", "Model Name", "True Value", "Predicted Value"}
            if not required_columns.issubset(data.columns):
                st.error(f"ファイル {path} に必要な列が含まれていません。必要な列: {', '.join(required_columns)}")
                continue

            # 散布図を作成して表示
            fig = plot_scatter_with_error_lines(data)
            st.plotly_chart(fig, use_container_width=False)
        except FileNotFoundError:
            st.error(f"ファイル {path} が見つかりません。パスを確認してください。")
        except Exception as e:
            st.error(f"ファイル {path} の処理中にエラーが発生しました: {e}")

if __name__ == "__main__":
    main()
