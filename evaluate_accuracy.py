import torch
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import json
import random
import pandas as pd


# Parameters
Num_eval = 100  # Number of evaluation examples
device = 'cuda:0'

# Load evaluation data
def load_squad_data(file_path):
    with open(file_path, "r") as f:
        squad_data = json.load(f)

    data_pairs = []
    for article in squad_data["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                question = qa["question"]
                is_impossible = qa.get("is_impossible", False)
                if not is_impossible:
                    answers = qa["answers"]
                    answer_starts = [answer["answer_start"] for answer in answers]
                    answer_texts = [answer["text"] for answer in answers]
                    data_pairs.append({
                        "context": context,
                        "question": question,
                        "answers": {
                            "answer_start": answer_starts,
                            "text": answer_texts,
                        },
                    })
    return data_pairs

# Load evaluation data pairs
eval_data_pairs = load_squad_data("SQuAD2.0/dev-v2.0.json")
eval_data_pairs = random.sample(eval_data_pairs, Num_eval)

# Load the model IDs
with open("download_model_list.txt", "r") as f:
    model_ids = [line.strip().replace("/", "_") for line in f if line.strip() and not line.startswith("#")]

# Prepare a DataFrame to store results
results_df = pd.DataFrame(columns=["Model_ID", "Accuracy"])

for model_id in model_ids:
    try:
        print(f"Evaluating model: {model_id}")

        # Paths and directories
        model_path = f"bert_qa_models/{model_id}"

        # Load base model and tokenizer
        model = AutoModelForQuestionAnswering.from_pretrained(model_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Create QA pipeline
        qa_pipeline = pipeline(
            "question-answering",
            model=model,
            tokenizer=tokenizer,
            device=0 if device == 'cuda' else -1,
        )

        # Evaluate the model
        correct = 0
        for idx, entry in enumerate(eval_data_pairs):
            context = entry['context']
            question = entry['question']
            answers = entry['answers']['text']

            try:
                result = qa_pipeline(question=question, context=context)
            except Exception as e:
                print(f"Error during QA pipeline at index {idx}: {e}")
                predicted_answer = ""
            else:
                predicted_answer = result['answer']

            is_correct = any(answer.lower() in predicted_answer.lower() for answer in answers)
            if is_correct:
                correct += 1

        # Calculate accuracy and save to DataFrame
        accuracy = correct / Num_eval
        results_df = pd.concat([results_df, pd.DataFrame([[model_id, accuracy]], columns=["Model_ID", "Accuracy"])], ignore_index=True)
        print(f"Model: {model_id}, Accuracy: {accuracy:.2f}")

    except Exception as e:
        print(f"Error evaluating model {model_id}: {e}\n")
        continue

    finally:
        # Free up GPU memory
        if 'model' in locals():
            del model
        if 'tokenizer' in locals():
            del tokenizer
        if 'qa_pipeline' in locals():
            del qa_pipeline
        torch.cuda.empty_cache()

    print("\n")

# Save results to CSV
results_df.to_csv("outputs/evaluation_accuracy.csv", index=False)
print("Evaluation complete. Results saved to outputs_evaluation_accuracy.csv.")
