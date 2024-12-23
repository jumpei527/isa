# モデルごとの類似度行列からCKAを計算

import os
import numpy as np
import pandas as pd
import config

# カーネル行列のセンタリングを行う関数
def centering(K):
    n = K.shape[0]
    unit = np.ones([n, n])
    I = np.eye(n)
    H = I - unit / n
    return np.dot(np.dot(H, K), H)  # Double centering

# CKAの計算
def linear_CKA(X, Y):
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)

    # X, Yのカーネル行列
    K = np.dot(X, X.T)
    L = np.dot(Y, Y.T)

    # カーネル行列をセンタリング
    centered_K = centering(K)
    centered_L = centering(L)

    # 分子部分：tr(K H L H)
    numerator = np.trace(np.dot(centered_K, centered_L))

    # 分母部分：sqrt(tr(K H K H) * tr(L H L H))
    denominator = np.sqrt(np.trace(np.dot(centered_K, centered_K)) * np.trace(np.dot(centered_L, centered_L)))

    if denominator == 0:
        return 0

    # CKAの最終値
    return numerator / denominator

# メイン処理
if __name__ == '__main__':

    input_file = config.input_file

    # outputsフォルダ内の全てのCSVファイルを取得
    csv_files = [f for f in os.listdir(f"outputs/{input_file}/similarity_matrix") if f.endswith('.csv')]
    num_files = len(csv_files)


    # 類似度行列を初期化
    similarity_matrix = np.zeros((num_files, num_files))

    # 全ての組み合わせに対して線形CKAを計算
    for i in range(num_files):
        for j in range(num_files):
            if i <= j:  # 上三角行列のみ計算
                X = pd.read_csv(os.path.join(f'outputs/{input_file}/similarity_matrix', csv_files[i]), header=None).values[1:, 1:]
                Y = pd.read_csv(os.path.join(f'outputs/{input_file}/similarity_matrix', csv_files[j]), header=None).values[1:, 1:]
                similarity_matrix[i, j] = linear_CKA(X, Y)
                similarity_matrix[j, i] = similarity_matrix[i, j]  # 対称行列

    os.makedirs(f"outputs/{input_file}/cka_matrix/", exist_ok=True)

    # 類似度行列をCSVファイルにエクスポート
    similarity_df = pd.DataFrame(similarity_matrix, index=csv_files, columns=csv_files)
    similarity_df.to_csv(f"outputs/{input_file}/cka_matrix/cka_matrix.csv")
