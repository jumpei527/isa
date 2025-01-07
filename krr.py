import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import config
import os

input_file = config.input_file

# パスの組み合わせリスト
path_configs = [
    {
        "cka_matrix_path": f"outputs/{input_file}/cka_matrix/cka_matrix.csv",
        "accuracy_path": "outputs/evaluation_accuracy.csv",
        "output_path": f"outputs/{input_file}/krr_results/results.txt",
    },
    {
        "cka_matrix_path": f"outputs/{input_file}/cka_matrix/cka_matrix_finetune.csv",
        "accuracy_path": "outputs/evaluation_accuracy_finetune.csv",
        "output_path": f"outputs/{input_file}/krr_results/finetune_results.txt",
    },
    {
        "cka_matrix_path": f"outputs/{input_file}/cka_matrix/cka_matrix.csv",
        "accuracy_path": "outputs/evaluation_accuracy_diff.csv",
        "output_path": f"outputs/{input_file}/krr_results/diff_results.txt",
    },
    {
        "cka_matrix_path": f"outputs/{input_file}/cka_matrix/cka_matrix.csv",
        "accuracy_path": "outputs/evaluation_accuracy_finetune.csv",
        "output_path": f"outputs/{input_file}/krr_results/results_before_cka_and_finetune_accuracy.txt",
    },
]

# 全ての組み合わせで処理
for config in path_configs:
    cka_matrix_path = config["cka_matrix_path"]
    accuracy_path = config["accuracy_path"]
    output_path = config["output_path"]

    # ディレクトリ作成
    dir_path = os.path.dirname(output_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # カーネル行列の読み込み
    K = pd.read_csv(cka_matrix_path, index_col=0).values

    # Accuracy列を読み込み
    accuracy_df = pd.read_csv(accuracy_path)
    y = accuracy_df['Accuracy'].values

    # データ数
    n_samples = K.shape[0]
    assert len(y) == n_samples, "データ数が一致しません。カーネル行列とAccuracyのサイズを確認してください。"

    # 正則化パラメータの候補
    param_grid = {'alpha': [0.1, 1, 10, 100]}

    # 繰り返し設定
    n_splits = 10
    test_size = 0.2

    # 精度記録用
    mse_list = []

    # 出力ファイルオープン
    with open(output_path, "w", encoding="utf-8") as f_out:
        f_out.write("=== Kernel Ridge Regression (KRR) 結果 ===\n\n")

        # 繰り返し実行
        for i in range(n_splits):
            indices = np.arange(n_samples)
            train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=i)

            # 学習データとテストデータのカーネル行列とターゲットベクトル
            K_train = K[np.ix_(train_idx, train_idx)]
            K_test = K[np.ix_(test_idx, train_idx)]
            y_train, y_test = y[train_idx], y[test_idx]

            # カーネルリッジ回帰モデル
            model = KernelRidge(kernel='precomputed')

            # グリッドサーチ設定
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=0)
            grid_search.fit(K_train, y_train)

            # 最適モデルで予測
            best_model = grid_search.best_estimator_
            y_test_pred = best_model.predict(K_test)

            # MSEを記録
            mse = mean_squared_error(y_test, y_test_pred)
            mse_list.append(mse)

            # ファイルへ書き込み
            f_out.write(f"Split {i+1}/{n_splits}  |  MSE = {mse:.4f}\n")

        # 平均と標準偏差の計算
        mse_mean = np.mean(mse_list)
        mse_std = np.std(mse_list)

        f_out.write("\n--- 最終結果 ---\n")
        f_out.write(f"平均MSE         : {mse_mean:.4f}\n")
        f_out.write(f"MSEの標準偏差   : {mse_std:.4f}\n")

    print(f"結果を '{output_path}' に保存しました。")
