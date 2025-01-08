import pandas as pd
import umap
import plotly.express as px
import streamlit as st
import config

# 必要なパラメータ設定
perplexity = 10
input_file = config.input_file
matrix_files = [
    ("cka_matrix", f"outputs/{input_file}/cka_matrix/cka_matrix.csv", "outputs/evaluation_accuracy.csv"),
    ("cka_matrix_finetune", f"outputs/{input_file}/cka_matrix/cka_matrix_finetune.csv", "outputs/evaluation_accuracy_finetune.csv")
]

# Streamlitアプリケーション
st.title("CKA行列のUMAP次元削減: 教師あり & 教師なし")

# 各ファイルについて処理
for matrix_name, file_path, accuracy_path in matrix_files:
    # Accuracyデータの読み込み
    accuracy_df = pd.read_csv(accuracy_path)
    accuracy_df = accuracy_df.set_index('Model_ID')  # Model_IDをインデックスに設定

    # モデル名の '/' を '_' に置き換え
    accuracy_df.index = accuracy_df.index.str.replace('/', '_', regex=False)

    # カーネル行列の読み込み
    matrix_df = pd.read_csv(file_path, index_col=0)
    
    # モデル名から".csv"を除く
    matrix_df.index = matrix_df.index.str.replace(".csv", "", regex=False)
    
    # accuracy_dfをmatrix_dfのインデックス順に並び替える
    reordered_accuracy_df = accuracy_df.reindex(matrix_df.index)
    
    # 並び替え後に欠損値があるか確認
    if reordered_accuracy_df.isnull().any().any():
        st.error(f"Model IDs in {matrix_name} do not match those in {accuracy_path}.")
        st.write("CKA Matrix Model IDs:", matrix_df.index.tolist())
        st.write("Accuracy Model IDs:", accuracy_df.index.tolist())
        continue
    
    # ラベルとしてAccuracyを取得
    y = reordered_accuracy_df['Accuracy'].values
    
    # データ整合性チェック
    if len(matrix_df) != len(y):
        st.error(f"Matrix size {len(matrix_df)} does not match Accuracy size {len(y)} for {matrix_name}.")
        continue
    
    # カーネル行列の値を取得
    matrix = matrix_df.values  # 数値データとして取得
    
    # --- 教師ありUMAP ---
    reducer_supervised = umap.UMAP(n_neighbors=perplexity, metric='euclidean', random_state=42)
    embedding_supervised = reducer_supervised.fit_transform(matrix, y=y)  # 教師あり次元削減
    
    # DataFrameに変換（モデル名、UMAP埋め込み結果、Accuracyを含む）
    supervised_df = pd.DataFrame(embedding_supervised, columns=['UMAP1', 'UMAP2'], index=matrix_df.index)
    supervised_df['Accuracy'] = y
    supervised_df['Model'] = supervised_df.index  # ツールチップ用にモデル名を追加
    
    # Plotlyを使用して教師ありプロットを作成
    fig_supervised = px.scatter(
        supervised_df,
        x='UMAP1',
        y='UMAP2',
        color='Accuracy',
        hover_data={'UMAP1': False, 'UMAP2': False, 'Model': True, 'Accuracy': True},  # ツールチップに表示するデータ
        title=f"{matrix_name}の教師あり次元削減",
        labels={'Accuracy': 'Accuracy'},
        color_continuous_scale='Viridis'
    )
    
    # Streamlitに教師ありプロットを表示
    st.plotly_chart(fig_supervised)

    # --- 教師なしUMAP ---
    reducer_unsupervised = umap.UMAP(n_neighbors=perplexity, metric='euclidean', random_state=42)
    embedding_unsupervised = reducer_unsupervised.fit_transform(matrix)  # 教師なし次元削減
    
    # DataFrameに変換（モデル名、UMAP埋め込み結果、Accuracyを含む）
    unsupervised_df = pd.DataFrame(embedding_unsupervised, columns=['UMAP1', 'UMAP2'], index=matrix_df.index)
    unsupervised_df['Accuracy'] = y
    unsupervised_df['Model'] = unsupervised_df.index  # ツールチップ用にモデル名を追加
    
    # Plotlyを使用して教師なしプロットを作成
    fig_unsupervised = px.scatter(
        unsupervised_df,
        x='UMAP1',
        y='UMAP2',
        color='Accuracy',
        hover_data={'UMAP1': False, 'UMAP2': False, 'Model': True, 'Accuracy': True},  # ツールチップに表示するデータ
        title=f"{matrix_name}の教師なし次元削減",
        labels={'Accuracy': 'Accuracy'},
        color_continuous_scale='Viridis'
    )
    
    # Streamlitに教師なしプロットを表示
    st.plotly_chart(fig_unsupervised)