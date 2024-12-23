from huggingface_hub import HfApi
from transformers.utils import logging
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import os
from tqdm import tqdm

# ロギングレベルを設定
logging.set_verbosity_error()

# APIクライアントの初期化
api = HfApi()

# ダウンロードするモデル一覧を取得
with open("download_model_list.txt", "r", encoding="utf-8") as f:
    model_list = [line.strip() for line in f if line.strip()]

# ダウンロード先のディレクトリを設定
download_dir = "./bert_qa_models"

if not os.path.exists(download_dir):
    os.makedirs(download_dir)

# モデルのダウンロード
for model_id in tqdm(model_list, desc="Downloading models"):
    try:
        # モデルカードを取得して、アーキテクチャを確認
        model_card = api.model_info(model_id)
        architectures = model_card.config.get('architectures', [])
        
        # "Bert" を含むアーキテクチャ名がある場合のみ続行
        if not any('Bert' in arch for arch in architectures):
            continue

        # モデルのローカルディレクトリを設定
        model_dir = os.path.join(download_dir, model_id.replace('/', '_'))

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # トークナイザーのダウンロード
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.save_pretrained(model_dir)

        # モデルのダウンロード
        model = AutoModelForQuestionAnswering.from_pretrained(model_id)
        model.save_pretrained(model_dir)

    except Exception as e:
        print(f"Failed to download {model_id}: {e}")
        continue
