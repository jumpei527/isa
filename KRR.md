
**カーネルリッジ回帰の方法について**

---

### **1. はじめに**

CKAで決めた$71 \times 71$のカーネル行列と，それに対応する精度ベクトル$\mathbf{y}$が得られているとする．ここで$\mathbf{y}$についてはfine tune前の精度や後の精度，あるいはその差分の３パターンを試す．

今回は生データが手に入らず，カーネル行列の形でデータが得られているので，モデルの予測性能の予測モデルとしてはカーネルベースの方法が適している．

**リッジ回帰**：線形回帰モデルに正則化（ペナルティ項）を加えることで過学習を防ぎモデルの汎化性能を向上させる手法．**カーネルリッジ回帰**はこのリッジ回帰を**カーネル法**と組み合わせたもので，非線形な関係性を捉えることが可能．


---

### **2. リッジ回帰の復習**

まずリッジ回帰速習．

**線形回帰モデル：**

与えられたデータセット $\{(\mathbf{x}_i, y_i)\}_{i=1}^n$に対して，目的は説明変数 $ \mathbf{x}_i \in \mathbb{R}^d $ と目的変数 $ y_i \in \mathbb{R} $ の関係をモデル化することで，線形回帰モデルは以下のように表される：

$$
   y_i = \mathbf{w}^\top \mathbf{x}_i + b + \epsilon_i
$$

ここで、$ \mathbf{w} $ は重みベクトル，$ b $ はバイアス項，$ \epsilon_i $ は誤差項．

リッジ回帰では以下の目的関数を最小化する：

$$
J(\mathbf{w}, b) = \sum_{i=1}^n \left( y_i - \mathbf{w}^\top \mathbf{x}_i - b \right)^2 + \lambda \|\mathbf{w}\|^2
$$

ここで$ \lambda > 0 $ は正則化パラメータでありモデルの複雑さを制御する．

---

### **3. カーネル法**

**カーネル法**はデータを高次元空間に写像することで非線形な関係性を線形な方法で捉える手法．直接高次元空間への写像を行うのではなくカーネル関数を用いて内積を計算する．

**カーネル関数：**

カーネル関数 $ K(\mathbf{x}, \mathbf{x}') $ は以下の性質を満たす関数である．

$$
K(\mathbf{x}, \mathbf{x}') = \phi(\mathbf{x})^\top \phi(\mathbf{x}')
$$

ここで$ \phi: \mathbb{R}^d \rightarrow \mathcal{H} $ は特徴空間への写像，$ \mathcal{H} $ はヒルベルト空間である．代表的なカーネル関数には以下がある：

- **線形カーネル：** $ K(\mathbf{x}, \mathbf{x}') = \mathbf{x}^\top \mathbf{x}' $
- **多項式カーネル：** $ K(\mathbf{x}, \mathbf{x}') = (\mathbf{x}^\top \mathbf{x}' + c)^p $
- **ガウシアン（RBF）カーネル：** $ K(\mathbf{x}, \mathbf{x}') = \exp\left(-\frac{\|\mathbf{x} - \mathbf{x}'\|^2}{2\sigma^2}\right) $

今は，CKAを使っていて，カーネル行列を事前計算済みカーネル（precomputed kernel）として扱う．`sklearn`の`KernelRidge`クラスではカーネルを`'precomputed'`として指定する．

---

### **4. カーネルリッジ回帰の定式化**

カーネルリッジ回帰はリッジ回帰にカーネル法を組み合わせたもので，非線形な回帰問題に対応する．

**特徴空間でのモデル：**

特徴空間 $ \mathcal{H} $ におけるモデルは以下のようになる：

$$
f(\mathbf{x}) = \mathbf{w}^\top \phi(\mathbf{x}) + b
$$

**カーネルリッジ回帰の目的関数：**

カーネルリッジ回帰では以下の目的関数を最小化する．

$$
J(\mathbf{w}, b) = \sum_{i=1}^n \left( y_i - \mathbf{w}^\top \phi(\mathbf{x}_i) - b \right)^2 + \lambda \|\mathbf{w}\|^2
$$

**最適化：**

目的関数を最小化するために偏微分：

1. **重みベクトルの最適化：**

$$
\mathbf{w} = (\Phi^\top \Phi + \lambda I)^{-1} \Phi^\top \mathbf{y}
$$

$ \Phi $ はデータ行列で$ \Phi_{i,j} = \phi_j(\mathbf{x}_i) $ ．

2. **バイアス項の最適化：**

バイアス項 $ b $ はデータの平均を用いて次で計算：

$$
b = \bar{y} - \mathbf{w}^\top \bar{\phi}
$$

ここで$ \bar{y} $ は目的変数の平均，$ \bar{\phi} $ は特徴ベクトルの平均．

重みベクトル $ \mathbf{w} $ をカーネル関数を用いて表現する：

$$
\mathbf{w} = \sum_{i=1}^n \alpha_i \phi(\mathbf{x}_i)
$$

ここで$ \alpha_i $ は係数．これを目的関数に代入すると

$$
f(\mathbf{x}) = \sum_{i=1}^n \alpha_i K(\mathbf{x}_i, \mathbf{x}) + b
$$

となる．

最適な係数 $ \boldsymbol{\alpha} $は次の方程式の解である:

$$
   \tag{a}
(\mathbf{K} + \lambda I) \boldsymbol{\alpha} = \mathbf{y}
$$

ここで$ \mathbf{K} $ はカーネル行列であり，$ \mathbf{K}_{i,j} = K(\mathbf{x}_i, \mathbf{x}_j) $．

---

### **5. カーネルリッジ回帰のアルゴリズム**

カーネルリッジ回帰は次の手順で行う：

1. **カーネル行列の計算：**

   データセット $ \{\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_n\} $ に対してカーネル行列 $ \mathbf{K} $ を計算する（所与の場合はスキップ）

   $$
   \mathbf{K}_{i,j} = K(\mathbf{x}_i, \mathbf{x}_j)
   $$

2. **係数の計算：**

   方程式(a)を解いて係数 $ \boldsymbol{\alpha} $ を求める

   $$
   \boldsymbol{\alpha} = (\mathbf{K} + \lambda I)^{-1} \mathbf{y}
   $$

3. **予測：**

   新しいデータ点 $ \mathbf{x} $ に対する予測値 $ \hat{y} $ は以下：

   $$
   \hat{y} = \sum_{i=1}^n \alpha_i K(\mathbf{x}_i, \mathbf{x}) + b
   $$

---

### **6. ハイパーパラメータの選択**

カーネルリッジ回帰には主に以下のハイパーパラメータがある：

- **正則化パラメータ $ \lambda $：**

- **カーネルの選択とそのパラメータ：**
 - 色々あるが，今回はカーネル行列が所与なのでとりあえず考えない．後で，CKA以外の方法や，CLSトークンのcosine以外の設計を考えるときにはハイパーパラメータが入りうる


データを学習データとテストデータに分けて，学習データの5-fold CVでハイパーパラメータを最適化してKRRを実行し，テストデータで評価する


以下，サンプルコード

 ```python
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error

# サンプル数
n_samples = 71

# ランダムなカーネル行列の生成（対称で正定）
np.random.seed(123)
## K, yは所与の行列，ベクトル

# サンプルのインデックスを生成
indices = np.arange(n_samples)

# 学習データと一テストデータに分割
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=123)

# カーネル行列のサブセットを作成
K_train = K[np.ix_(train_idx, train_idx)]
K_test = K[np.ix_(test_idx, train_idx)]

# ターゲットベクトルのサブセットを作成
y_train = y[train_idx]
y_test = y[test_idx]

# 正則化パラメータの候補
param_grid = {
    'lambda': [0.1, 1, 10, 100]
}

# カーネルリッジ回帰モデルの定義（カーネルは事前計算済み）
model = KernelRidge(kernel='precomputed')

# グリッドサーチの設定
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1)

# グリッドサーチの実行
grid_search.fit(K_train, y_train)

# 最適なハイパーパラメータの表示
print(f"Best parameters: {grid_search.best_params_}")

# 最適なモデルの取得
best_model = grid_search.best_estimator_

# テストデータでの予測
y_test_pred = best_model.predict(K_test)

# テストデータでの性能評価
mse_test = mean_squared_error(y_test, y_test_pred)
print(f"Test Mean Squared Error: {mse_test:.4f}")
```




できればこの学習・テストデータへの分割を5回とか10回ランダムに行って，予測精度の平均とSDも出してほしい．





**参考文献**

- **Bishop, C. M.** (2006). *Pattern Recognition and Machine Learning*. Springer.
- **Schölkopf, B., & Smola, A. J.** (2002). *Learning with Kernels: Support Vector Machines, Regularization, Optimization, and Beyond*. MIT Press.
- **赤穂昭太郎** (2008). *カーネル多変量解析*. 岩波書店.
