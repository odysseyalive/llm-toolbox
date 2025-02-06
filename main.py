import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from safetensors.torch import save_file
from peft import get_peft_model, LoraConfig, TaskType
import torch
from torch.cuda.amp import GradScaler

scaler = GradScaler()


def simple_tokenize(batch, text_field, tokenizer):
    text = batch.get(text_field)
    if text is None:
        raise ValueError(
            f"Dataset does not have the field '{text_field}'. Available fields: {list(batch.keys())}"
        )
    return tokenizer(text, truncation=True, padding="max_length", max_length=512)


def instruction_prompt_tokenize(batch, tokenizer):
    if "question" not in batch or "response" not in batch:
        raise ValueError(
            "Missing required fields for instruction prompt tokenization: 'question' and/or 'response'"
        )

    merged = []
    for i in range(len(batch["question"])):
        question = batch["question"][i] or ""
        response = batch["response"][i] or ""
        system_prompt = (
            batch.get("system_prompt", [""] * len(batch["question"]))[i] or ""
        )

        if system_prompt.strip():
            prompt = f"{system_prompt.strip()}\nQuestion: {question.strip()}\nResponse: {response.strip()}"
        else:
            prompt = f"Question: {question.strip()}\nResponse: {response.strip()}"
        merged.append(prompt)

    return tokenizer(merged, truncation=True, padding="max_length", max_length=512)


def train_model(args):
    print(f"Starting training using model {args.model_id}")
    if args.use_lora:
        print("Using LoRA for fine-tuning.")
    device = "cuda" if args.use_cuda else "cpu"
    print(f"Using device: {device}")

    print(f"Loading dataset {args.dataset_id}")
    dataset = load_dataset(args.dataset_id)
    train_dataset = dataset["train"] if "train" in dataset else dataset

    num_examples = len(train_dataset)
    print(f"Generating train split: {num_examples} examples")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    print("Available columns in dataset:", train_dataset.column_names)

    if args.instruction_prompt:
        train_dataset = train_dataset.map(
            lambda batch: instruction_prompt_tokenize(batch, tokenizer),
            batched=True,
            remove_columns=train_dataset.column_names,
        )
    else:
        train_dataset = train_dataset.map(
            lambda batch: simple_tokenize(batch, args.dataset_text_field, tokenizer),
            batched=True,
            remove_columns=train_dataset.column_names,
        )

    print("Tokenization complete.")

    model = AutoModelForCausalLM.from_pretrained(args.model_id)
    model.to(device)

    if args.use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"],
        )
        model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_dir=args.logging_dir,
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    print("Starting the training process...")
    trainer.train()
    print("Training complete.")

    print("Saving the trained model...")
    model_path = "trained_model.safetensors"
    save_file(model.state_dict(), model_path)
    print(f"Model saved successfully to {model_path}.")


def main():
    parser = argparse.ArgumentParser(description="LLM Toolbox Training Script")
    parser.add_argument("command", choices=["train"], help="Command to run")
    parser.add_argument(
        "--model-id",
        required=True,
        help="ID of the model to use (e.g. models/deepseek-r1-1.5b)",
    )
    parser.add_argument(
        "--dataset-id",
        required=True,
        help="ID of the dataset to use (e.g. datasets/OpenOrca)",
    )
    parser.add_argument(
        "--num-train-epochs", type=int, default=5, help="Number of training epochs"
    )
    parser.add_argument(
        "--use-cuda", action="store_true", help="Flag to allow usage of CUDA"
    )
    parser.add_argument(
        "--use-lora",
        action="store_true",
        help="Flag to indicate usage of LoRA for fine-tuning",
    )
    parser.add_argument(
        "--dataset-text-field",
        type=str,
        default="text",
        help="Dataset column name for simple tokenization (default: 'text')",
    )
    parser.add_argument(
        "--instruction-prompt",
        action="store_true",
        help="Enable instruction-like tokenization using the 'question' and 'response' fields",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Directory to save the model and results",
    )
    parser.add_argument(
        "--batch-size", type=int, default=2, help="Batch size for training"
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=10000,
        help="Number of steps between model saves",
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=2,
        help="Limit the total amount of checkpoints",
    )
    parser.add_argument(
        "--logging-dir", type=str, default="./logs", help="Directory to save the logs"
    )

    args = parser.parse_args()

    if args.command == "train":
        train_model(args)


if __name__ == "__main__":
    main()
