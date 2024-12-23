import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import config

input_file = config.input_file

# CSVファイルのパス
cka_matrix_path = f"outputs/{input_file}/cka_matrix/cka_matrix.csv"
accuracy_path = "outputs/evaluation_accuracy.csv"

# カーネル行列の読み込み
K = pd.read_csv(cka_matrix_path, index_col=0).values  # 1行目・1列目をスキップし数値のみ

# Accuracy列を読み込み（数値データとして）
accuracy_df = pd.read_csv(accuracy_path)
y = accuracy_df['Accuracy'].values  # Accuracy列を取得

# データ数
n_samples = K.shape[0]
print(n_samples)
assert len(y) == n_samples, "データ数が一致しません。カーネル行列とAccuracyのサイズを確認してください。"

# 正則化パラメータの候補
param_grid = {'alpha': [0.1, 1, 10, 100]}  # alphaは正則化パラメータλに対応

# 繰り返し設定
n_splits = 10  # 学習・テスト分割の回数
test_size = 0.2

# 精度記録用
mse_list = []

# 繰り返し実行
for i in range(n_splits):
    print(f"Split {i+1}/{n_splits}")
    
    # インデックスを明示的に設定
    indices = np.arange(n_samples)
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=i)

    # 確認: インデックスが範囲外でないこと
    assert np.max(train_idx) < n_samples, "train_idxが範囲を超えています"
    assert np.max(test_idx) < n_samples, "test_idxが範囲を超えています"
    
    # 学習データとテストデータのカーネル行列とターゲットベクトル
    K_train = K[np.ix_(train_idx, train_idx)]
    K_test = K[np.ix_(test_idx, train_idx)]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # カーネルリッジ回帰モデルの定義
    model = KernelRidge(kernel='precomputed')
    
    # グリッドサーチの設定
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=0)
    grid_search.fit(K_train, y_train)
    
    # 最適モデルで予測
    best_model = grid_search.best_estimator_
    y_test_pred = best_model.predict(K_test)
    
    # MSEを記録
    mse = mean_squared_error(y_test, y_test_pred)
    mse_list.append(mse)
    print(f"  MSE = {mse:.4f}")

# 平均と標準偏差の計算
mse_mean = np.mean(mse_list)
mse_std = np.std(mse_list)

print("\n--- 結果 ---")
print(f"平均MSE: {mse_mean:.4f}")
print(f"MSEの標準偏差: {mse_std:.4f}")
