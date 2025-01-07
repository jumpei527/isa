## モデルの類似度の測定
**モデルをダウンロードする**
   ```bash
   python download_models.py
   ```

**モデルをファインチューニング**
   ```bash
   python peft_finetune.py
   ```

**モデルの精度を評価**
   ```bash
   python evaluate_accuracy.py
   ```

**使うテキストをconfig.pyで設定(qa1が医療用、qa2が一般的な文章)**

**類似度行列を作成(ファインチューニング前のモデル)**
   ```bash
   python calculate_similarity_matrix.py
   ```

**CKA行列を作成(ファインチューニング前のモデル)**
   ```bash
   python calculate_cka.py
   ```

**類似度行列を作成(ファインチューニング後のモデル)**
   ```bash
   python calculate_similarity_matrix_finetune.py
   ```

**CKA行列を作成(ファインチューニング後のモデル)**
   ```bash
   python calculate_cka_finetune.py
   ```

**CKA行列からマッピング**
   ```bash
   python visualize_cka.py
   ```

**ファインチューニング前と後の精度の差分を出力**
   ```bash
   python calculate_accuracy_diff.py
   ```

**CKA行列からカーネルリッジ回帰**
   ```bash
   python krr.py
   ```

**カーネルリッジ回帰の結果を散布図で可視化**
   ```bash
   streamlit run visualize_krr.py
   ```
