import csv

# CSVファイルのパス
base_csv = "outputs/evaluation_accuracy.csv"
finetune_csv = "outputs/evaluation_accuracy_finetune.csv"
output_csv = "outputs/evaluation_accuracy_diff.csv"

# (1) ベース精度を辞書にロード
base_dict = {}
with open(base_csv, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        model_id = row["Model_ID"]
        accuracy = float(row["Accuracy"])
        base_dict[model_id] = accuracy

# (2) ファインチューニング後の精度を辞書にロード
finetune_dict = {}
with open(finetune_csv, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        # 例: '/' を '_' に置換するなど表記ゆれの対応が必要な場合
        model_id = row["Model_ID"].replace("/", "_")
        accuracy = float(row["Accuracy"])
        finetune_dict[model_id] = accuracy

# (3) 差分 = (finetune - base) を計算
diff_results = []
for model_id_base, acc_base in base_dict.items():
    if model_id_base in finetune_dict:
        acc_finetune = finetune_dict[model_id_base]
        acc_diff = acc_finetune - acc_base
        # 小数点以下2桁に丸める
        acc_diff_rounded = round(acc_diff, 2)
        diff_results.append((model_id_base, acc_diff_rounded))

# (4) CSVとして出力
with open(output_csv, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Model_ID", "Accuracy_diff"])
    for model_id, diff in diff_results:
        # f"{diff:.2f}" のようにフォーマットして書き込んでもOK
        writer.writerow([model_id, f"{diff:.2f}"])

print(f"差分計算が完了しました。'{output_csv}' に出力しています。")
