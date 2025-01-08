import os
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
import pandas as pd
import config

def list_directory_names(directory):
    """
    Lists all directory names in the specified directory.
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
    指定されたファイルから入力文章のリストを読み込む。
    :param file_path: テキストファイルのパス
    :return: 文章のリスト
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

# bert_qa_modelsディレクトリ内の全てのディレクトリ名を取得
model_directories = list_directory_names('bert_qa_models')

# qa1.txtファイルから入力文章を読み込む
input_texts = load_input_texts(f"input_texts/{input_file}.txt")

if not input_texts:
    print("Input texts could not be loaded. Exiting...")
    exit()

# 使用するデバイスを設定（GPUが使用可能な場合はcuda、そうでなければCPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 出力先のディレクトリを確認し、存在しない場合は作成
output_dir = f'outputs/{input_file}/cls_vectors'
os.makedirs(output_dir, exist_ok=True)

for model_dir in model_directories:
    print(f"Processing model in directory: {model_dir}")

    # AutoTokenizerとAutoModelForQuestionAnsweringを使用して、トークナイザーとモデルを自動で選択
    tokenizer = AutoTokenizer.from_pretrained(f'bert_qa_models/{model_dir}')
    model = AutoModelForQuestionAnswering.from_pretrained(f'bert_qa_models/{model_dir}', output_hidden_states=True)

    # モデルを指定したデバイスに転送
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

    # CLSトークンベクトルをCSV形式で保存
    cls_vectors_df = pd.DataFrame(cls_vectors.cpu().numpy())
    cls_vectors_df.to_csv(f'{output_dir}/{model_dir}_cls_vectors.csv', index=False, header=False)

    print(f"CLS token vectors saved for model {model_dir} at {output_dir}/{model_dir}_cls_vectors.csv")
