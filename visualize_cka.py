import os
import pandas as pd
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from umap import UMAP
import config

# 必要なパラメータを設定
perplexity = 10
input_file = config.input_file

# 処理するファイルリスト
matrix_files = [
    ("cka_matrix", f"outputs/{input_file}/cka_matrix/cka_matrix.csv"),
    ("cka_matrix_finetune", f"outputs/{input_file}/cka_matrix/cka_matrix_finetune.csv")
]

# CSVファイルの読み込み
def load_data(file_path):
    # CKAマトリックスの読み込み
    cka_df = pd.read_csv(file_path, header=None)
    cka_matrix_np = cka_df.values[1:, 1:].astype(float)  # データ部分（最初の行と列を削除）
    labels = cka_df.values[1:, 0]                      # ラベル（最初の列）
    labels = [label.replace('.csv', '') for label in labels]  # .csvを除外

    # 改善データの読み込み
    improvements_df = pd.read_csv('improvements.csv')
    improvements_dict = dict(zip(improvements_df['Model'], improvements_df['Improvement']))

    return cka_matrix_np, labels, improvements_dict

# カラーマッピング関数
def get_color_values(improvements, max_improvement, min_improvement):
    colors = []
    for val in improvements:
        if pd.isna(val):
            colors.append('rgba(255, 255, 255, 1)')  # 白
        elif val > 0:
            opacity = min(val / max_improvement, 1)
            colors.append(f'rgba(0, 0, 255, {opacity})')  # 青
        elif val < 0:
            opacity = min(-val / abs(min_improvement), 1)
            colors.append(f'rgba(255, 0, 0, {opacity})')  # 赤
        else:
            colors.append('rgba(255, 255, 255, 1)')  # 白
    return colors

# 各ファイルを処理
for matrix_name, matrix_path in matrix_files:
    matrix_path = matrix_path.format(input_file)

    if not os.path.exists(matrix_path):
        print(f"File not found: {matrix_path}")
        continue

    print(f"Processing {matrix_name}...")

    # データの読み込み
    cka_matrix_np, labels, improvements_dict = load_data(matrix_path)

    # t-SNEによる次元削減（2次元）
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=1000)
    transformed_data_tsne = tsne.fit_transform(cka_matrix_np)

    # UMAPによる次元削減（2次元）
    umap_model = UMAP(n_components=2, random_state=42)
    transformed_data_umap = umap_model.fit_transform(cka_matrix_np)

    # t-SNEの結果をデータフレームに変換
    tsne_df = pd.DataFrame(transformed_data_tsne, columns=['Dim 1', 'Dim 2'])
    tsne_df['Label'] = labels  # ラベルを追加
    tsne_df['Improvement'] = tsne_df['Label'].map(improvements_dict)

    # UMAPの結果をデータフレームに変換
    umap_df = pd.DataFrame(transformed_data_umap, columns=['Dim 1', 'Dim 2'])
    umap_df['Label'] = labels  # ラベルを追加
    umap_df['Improvement'] = umap_df['Label'].map(improvements_dict)

    # カラーバリューを定義（NaNは0に設定して白に近づける）
    max_improvement = max(tsne_df['Improvement'].max(), umap_df['Improvement'].max())
    min_improvement = min(tsne_df['Improvement'].min(), umap_df['Improvement'].min())

    tsne_df['Improvement_plot'] = tsne_df['Improvement'].fillna(0)
    umap_df['Improvement_plot'] = umap_df['Improvement'].fillna(0)

    # 色の計算
    tsne_colors = get_color_values(tsne_df['Improvement'], max_improvement, min_improvement)
    umap_colors = get_color_values(umap_df['Improvement'], max_improvement, min_improvement)

    # t-SNEのプロットを作成
    fig_tsne = go.Figure()
    fig_tsne.add_trace(go.Scatter(
        x=tsne_df['Dim 1'],
        y=tsne_df['Dim 2'],
        mode='markers',
        marker=dict(
            size=10,
            color=tsne_colors,  # カスタム色を使用
            showscale=False,    # デフォルトのカラースケールを無効化
            line=dict(width=1, color='black')
        ),
        text=tsne_df['Label'],
        hoverinfo='text'
    ))

    fig_tsne.update_layout(
        title=f't-SNEによる{matrix_name}の可視化',
        xaxis=dict(
            zeroline=False,
            showline=True,
            linecolor='black',
            mirror=True,
            showgrid=False,
            automargin=True
        ),
        yaxis=dict(
            zeroline=False,
            showline=True,
            linecolor='black',
            mirror=True,
            showgrid=False,
            automargin=True
        ),
        plot_bgcolor='white',
        width=800,
        height=600,
        margin=dict(l=50, r=50, b=50, t=50)
    )

    # UMAPのプロットを作成
    fig_umap = go.Figure()
    fig_umap.add_trace(go.Scatter(
        x=umap_df['Dim 1'],
        y=umap_df['Dim 2'],
        mode='markers',
        marker=dict(
            size=10,
            color=umap_colors,  # カスタム色を使用
            showscale=False,    # デフォルトのカラースケールを無効化
            line=dict(width=1, color='black')
        ),
        text=umap_df['Label'],
        hoverinfo='text'
    ))
    fig_umap.update_layout(
        title=f'UMAPによる{matrix_name}の可視化',
        xaxis=dict(
            zeroline=False,
            showline=True,
            linecolor='black',
            mirror=True,
            showgrid=False,
            automargin=True
        ),
        yaxis=dict(
            zeroline=False,
            showline=True,
            linecolor='black',
            mirror=True,
            showgrid=False,
            automargin=True
        ),
        plot_bgcolor='white',
        width=800,
        height=600,
        margin=dict(l=50, r=50, b=50, t=50)
    )

    # 結果を保存するディレクトリを作成
    output_dir = f"outputs/{input_file}/visualization_results/{matrix_name}"
    os.makedirs(output_dir, exist_ok=True)

    # t-SNEのプロットを画像ファイルとして保存
    fig_tsne.write_image(f"{output_dir}/tsne_visualization.png")

    # UMAPのプロットを画像ファイルとして保存
    fig_umap.write_image(f"{output_dir}/umap_visualization.png")

    print(f"{matrix_name}のt-SNEとUMAPの可視化結果を {output_dir} に保存しました。")
