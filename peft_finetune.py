# Import necessary libraries
from datasets import Dataset
import random
import torch
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    pipeline,
    default_data_collator,
)
from peft import get_peft_model, LoraConfig, TaskType
import json
import os


# Parameters
Num_train = 30000  # Number of training examples
Num_eval = 100  # Number of evaluation examples
device = 'cuda:0' 

# Set random seed for reproducibility
random.seed(42)

# Load and prepare training data
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


# Prepare validation features
def prepare_validation_features(examples):
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        sequence_ids = tokenized_examples.sequence_ids(i)
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(i)
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == 1 else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]
    return tokenized_examples


## Main routine

# Load training data
train_data_pairs = load_squad_data("SQuAD2.0/train-v2.0.json")
# Limit to Num_train training examples
train_data_pairs = random.sample(train_data_pairs, min(Num_train, len(train_data_pairs)))
train_dataset = Dataset.from_list(train_data_pairs)
del train_data_pairs

# Load evaluation data
eval_data_pairs = load_squad_data("SQuAD2.0/dev-v2.0.json")
# Sample evaluation data
eval_data_pairs = random.sample(eval_data_pairs, Num_eval)
eval_dataset = Dataset.from_list(eval_data_pairs)


# Load the first `Num_models` model IDs, skipping lines that start with '#'
with open("download_model_list.txt", "r") as f:
    model_ids = [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Open files to save evaluation results and model answers
with open("evaluation_results_finetune.txt", "a") as results_file, open("predicted_answers_finetune.txt", "w") as answers_file:
    for model_id in model_ids:
        try:
            print(f"Training and evaluating model: {model_id}")

            # Load model and tokenizer
            model = AutoModelForQuestionAnswering.from_pretrained(model_id).to(device)
            tokenizer = AutoTokenizer.from_pretrained(model_id)

            model_path = f"./finetune/peft_lora_qa/{model_id}"
            if os.path.exists(model_path):
                print(f"Model {model_id} already fine-tuned. Skipping training...")
            else:
                # Configure PEFT with LoRA
                peft_config = LoraConfig(
                    task_type="QUESTION_ANSWERING",  # Use the correct task type for question answering
                    r=8,                  # Rank of the LoRA decomposition
                    lora_alpha=32,        # Scaling factor
                    lora_dropout=0.1,     # Dropout rate for LoRA layers
                    target_modules=['intermediate.dense', 'output.dense'],  # Target modules for LoRA
                )
                model = get_peft_model(model, peft_config)

                # Move model to device
                model.to(device)

                max_length = 384  # The maximum length of a feature (question and context)
                doc_stride = 128  # The stride when splitting up a long document

                def prepare_train_features(examples):
                    tokenized_examples = tokenizer(
                        examples["question"],
                        examples["context"],
                        truncation="only_second",
                        max_length=max_length,
                        stride=doc_stride,
                        return_overflowing_tokens=True,
                        return_offsets_mapping=True,
                        padding="max_length",
                    )
                    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
                    offset_mapping = tokenized_examples.pop("offset_mapping")

                    tokenized_examples["start_positions"] = []
                    tokenized_examples["end_positions"] = []

                    for i, offsets in enumerate(offset_mapping):
                        input_ids = tokenized_examples["input_ids"][i]
                        cls_index = input_ids.index(tokenizer.cls_token_id)
                        sequence_ids = tokenized_examples.sequence_ids(i)
                        sample_index = sample_mapping[i]
                        answers = examples["answers"][sample_index]
                        if len(answers["answer_start"]) == 0:
                            tokenized_examples["start_positions"].append(cls_index)
                            tokenized_examples["end_positions"].append(cls_index)
                        else:
                            start_char = answers["answer_start"][0]
                            end_char = start_char + len(answers["text"][0])
                            token_start_index = 0
                            token_end_index = len(input_ids) - 1

                            while sequence_ids[token_start_index] != 1:
                                token_start_index += 1
                            while sequence_ids[token_end_index] != 1:
                                token_end_index -= 1

                            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                                tokenized_examples["start_positions"].append(cls_index)
                                tokenized_examples["end_positions"].append(cls_index)
                            else:
                                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                                    token_start_index += 1
                                tokenized_examples["start_positions"].append(token_start_index - 1)
                                while offsets[token_end_index][1] >= end_char:
                                    token_end_index -= 1
                                tokenized_examples["end_positions"].append(token_end_index + 1)
                    return tokenized_examples

                # Tokenize training data
                train_dataset_tokenized = train_dataset.map(
                    prepare_train_features,
                    batched=True,
                    remove_columns=train_dataset.column_names,
                )


                # Tokenize evaluation data
                eval_dataset_tokenized = eval_dataset.map(
                    prepare_validation_features,
                    batched=True,
                    remove_columns=eval_dataset.column_names,
                )

                # Set up training arguments
                training_args = TrainingArguments(
                    output_dir=f'./finetune/results/{model_id}',
                    num_train_epochs=3,
                    per_device_train_batch_size=16,
                    per_device_eval_batch_size=16,
                    learning_rate=3e-4,
                    logging_steps=50,
                    evaluation_strategy="no",
                    save_steps=1000,
                    logging_dir=f'./finetune/logs_{model_id}',
                    fp16=torch.cuda.is_available(),
                )

                # Create Trainer
                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_dataset_tokenized,
                    tokenizer=tokenizer,
                    data_collator=default_data_collator,
                )

                # Train the model
                print(f"Training the model {model_id}...")
                trainer.train()

                # Save the fine-tuned model with LoRA
                model.save_pretrained(f"./finetune/peft_lora_qa/{model_id}")

            # Create QA pipeline for evaluation
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

                # Handle cases where context or question is too long
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

                # Write detailed results to predicted_answers.txt
                answers_file.write(f"Model: {model_id}\n")
                answers_file.write(f"Context: {context}\n")
                answers_file.write(f"Question: {question}\n")
                answers_file.write(f"Correct Answers: {answers}\n")
                answers_file.write(f"Model Prediction: {predicted_answer}\n")
                answers_file.write(f"Correct: {is_correct}\n")
                answers_file.write("\n" + "-"*50 + "\n\n")

            # Write accuracy to evaluation_results.txt
            accuracy = correct / Num_eval
            results_file.write(f"Model: {model_id} (peft_lora_qa_squad2), Accuracy: {accuracy:.2f}\n\n")
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
            if 'trainer' in locals():
                del trainer
            if 'qa_pipeline' in locals():
                del qa_pipeline
            torch.cuda.empty_cache()

        print("\n")

print("Evaluation complete. Results saved to evaluation_results.txt and predicted_answers.txt.")