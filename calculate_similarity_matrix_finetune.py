import os
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from peft import PeftModel
import torch
import pandas as pd
import config

def list_directory_names(directory):
    """
    指定されたディレクトリ内のサブディレクトリ名をリストで返す。
    """
    try:
        entries = os.listdir(directory)
        directories = [d for d in entries if os.path.isdir(os.path.join(directory, d))]
        return directories
    except FileNotFoundError:
        print(f"The directory '{directory}' does not exist.")
        return []
    except PermissionError:
        print(f"Permission denied to access the directory '{directory}'.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def load_input_texts(file_path):
    """
    指定されたJSONファイルから入力テキストを読み込む。
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            import json
            return json.load(file)
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
        return []
    except json.JSONDecodeError:
        print(f"Error parsing '{file_path}'. Ensure it's a valid JSON file.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

input_file = config.input_file

# qa1.txtファイルから入力文章を読み込む
input_texts = load_input_texts(f"input_texts/{input_file}.txt")

if not input_texts:
    print("Input texts could not be loaded. Exiting...")
    exit()

# 使用するデバイスを設定（GPUが使用可能な場合はcuda、そうでなければCPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# モデルIDを読み込む
with open("finetune_models/test_download_model.txt", "r") as f:
    model_ids = [line.strip() for line in f if line.strip() and not line.startswith("#")]

for model_id in model_ids:
    try:
        print(f"Evaluating model: {model_id}")

        # トークナイザとモデルのパスを指定
        tokenizer_path = f"./bert_qa_models/{'_'.join(model_id.split('/'))}"  # トークナイザのパス
        model_base_path = f"./finetune_models/peft_lora_qa/{model_id}"  # モデルのベースパス
        print(tokenizer_path)
        print(model_base_path)

        # ベースモデルとPEFTモデルのロード
        base_model = AutoModelForQuestionAnswering.from_pretrained(
            model_base_path, output_hidden_states=True
        )
        model = PeftModel.from_pretrained(base_model, model_base_path)  # PEFTモデルをラップ
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)  # トークナイザを指定のパスからロード

        # モデルをデバイスに移動
        model.to(device)

        # モデルを評価モードに設定
        model.eval()

        # トークナイズ
        encoded_inputs = tokenizer(input_texts, return_tensors='pt', padding=True).to(device)

        # モデルの実行
        with torch.no_grad():
            outputs = model(**encoded_inputs)

        # 隠れ層の出力を取得（outputs.hidden_states[-1] が最後の隠れ層）
        hidden_states = outputs.hidden_states

        # 最後の隠れ層のCLSトークンを取得
        cls_vectors = hidden_states[-1][:, 0, :]  # Shape: (len(input_texts), hidden_size)

        # 各ベクトルを正規化
        normalized_vectors = torch.nn.functional.normalize(cls_vectors, p=2, dim=1)

        # 類似度行列の計算
        similarity_matrix = torch.matmul(normalized_vectors, normalized_vectors.transpose(0, 1))

        # NumPyに変換
        similarity_matrix_np = similarity_matrix.cpu().numpy()

        # Pandas DataFrameに変換
        similarity_df = pd.DataFrame(
            similarity_matrix_np,
            index=[f"Q{i+1}" for i in range(len(input_texts))],
            columns=[f"Q{j+1}" for j in range(len(input_texts))]
        )

        # 類似度行列をCSVとして保存
        output_dir = f'outputs/{input_file}/similarity_matrix_finetune/{model_id.replace("/", "_")}'
        os.makedirs(os.path.dirname(output_dir), exist_ok=True)

        similarity_df.to_csv(f'{output_dir}.csv', index=True)

        print(f"Model {model_id} processed successfully.")

    except Exception as e:
        print(f"An error occurred while processing model {model_id}: {e}")
