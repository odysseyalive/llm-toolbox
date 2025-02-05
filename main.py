#!/usr/bin/env python3
import os
import sys
import argparse
import torch
import subprocess
import traceback
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

# Set a custom exception hook to log errors to error.txt
def log_exception(exctype, value, tb):
    with open("error.txt", "w") as f:
        f.write("An unhandled exception occurred:\n")
        f.write("Type: " + str(exctype) + "\n")
        f.write("Value: " + str(value) + "\n")
        f.write("Traceback:\n")
        f.write("".join(traceback.format_tb(tb)))
    # Also print the error to stderr.
    sys.__excepthook__(exctype, value, tb)

sys.excepthook = log_exception

# Try to import PEFT for LoRA fine-tuning; if not available, LoRA functionality will be disabled.
try:
    from peft import get_peft_model, LoraConfig, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

def tokenize_function(tokenizer, max_seq_length):
    def tokenize(example):
        # Adjust the key "text" if your dataset uses a different field name.
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
        )
    return tokenize

def train_model(args):
    print(f"Starting training using model {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id)

    # Optionally apply LoRA adaptation if --use-lora is specified.
    if args.use_lora:
        if not PEFT_AVAILABLE:
            error_msg = ("PEFT library not installed. "
                         "Install it using 'pip install peft' to use LoRA functionality.")
            print(error_msg)
            raise RuntimeError(error_msg)
        print("Using LoRA for fine-tuning.")
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],  # Adjust target modules as needed for your model.
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
    else:
        print("Training the full model without LoRA.")

    # Determine the device based on the --use-cuda flag.
    if args.use_cuda:
        if not torch.cuda.is_available():
            print("Warning: --use-cuda flag is set but CUDA is not available. Falling back to CPU.")
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    model.to(device)

    # Load the dataset from Hugging Face; default is Open-Orca.
    dataset = load_dataset(args.dataset_id)
    if "train" not in dataset:
        dataset = dataset["train"].train_test_split(train_size=args.train_val_split)
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]
    else:
        train_dataset = dataset["train"]
        eval_dataset = dataset.get("validation", None)

    # Tokenize dataset using the provided tokenizer.
    print("Tokenizing dataset...")
    tokenize_fn = tokenize_function(tokenizer, args.max_seq_length)
    train_dataset = train_dataset.map(tokenize_fn, batched=True, remove_columns=train_dataset.column_names)
    if eval_dataset:
        eval_dataset = eval_dataset.map(tokenize_fn, batched=True, remove_columns=eval_dataset.column_names)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        evaluation_strategy="steps" if eval_dataset else "no",
        eval_steps=1000,
        logging_steps=50,
        save_steps=1000,
        fp16=args.fp16,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        save_total_limit=3,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Start training.
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"Training complete. Model saved to {args.output_dir}")

def convert_to_gguf(args):
    print(f"Starting conversion of {args.input_model} to GGUF format as {args.output_model}")

    # Assume that an external conversion tool is available named "deepseek-gguf-converter".
    # This tool should convert a safetensors model to a GGUF model.
    conversion_tool = "deepseek-gguf-converter"  # Ensure this is installed and in your PATH.
    
    # Check if the conversion tool is available by calling its help command.
    try:
        subprocess.run([conversion_tool, "--help"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except FileNotFoundError:
        error_msg = (f"Conversion tool '{conversion_tool}' not found. "
                     "Please install it and ensure it is in your PATH.")
        print(error_msg)
        raise RuntimeError(error_msg)
    except subprocess.CalledProcessError as e:
        error_msg = f"Conversion tool '{conversion_tool}' encountered an error: {e}"
        print(error_msg)
        raise RuntimeError(error_msg)

    # Build the conversion command with the required arguments.
    command = [
        conversion_tool,
        "--safetensors", args.input_model,
        "--gguf", args.output_model
    ]
    print("Running conversion command:", " ".join(command))
    try:
        subprocess.run(command, check=True)
        print("Conversion complete. GGUF model saved as", args.output_model)
    except subprocess.CalledProcessError as e:
        error_msg = f"Conversion failed: {e}"
        print(error_msg)
        raise RuntimeError(error_msg)

def main():
    parser = argparse.ArgumentParser(
        description="Script to fine-tune DeepSeek R1 1.5B (optionally with LoRA) and convert to GGUF format."
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Sub-command help")

    # Train subcommand.
    train_parser = subparsers.add_parser("train", help="Train the model (optionally with LoRA)")
    train_parser.add_argument("--model-id", type=str, default="DeepSeek/deepseek-r1-1.5b",
                              help="Hugging Face model id for the base model")
    train_parser.add_argument("--dataset-id", type=str, default="Open-Orca/OpenOrca",
                              help="Hugging Face dataset id (default: Open-Orca/OpenOrca)")
    train_parser.add_argument("--output-dir", type=str, default="./finetuned_model",
                              help="Directory to save the fine-tuned model")
    train_parser.add_argument("--max-seq-length", type=int, default=1024,
                              help="Maximum sequence length for tokenization")
    train_parser.add_argument("--train-val-split", type=float, default=0.9,
                              help="Ratio for splitting train and validation if dataset has no explicit splits")
    train_parser.add_argument("--num-train-epochs", type=int, default=3,
                              help="Number of training epochs")
    train_parser.add_argument("--per-device-train-batch-size", type=int, default=2,
                              help="Training batch size per device")
    train_parser.add_argument("--learning-rate", type=float, default=5e-5,
                              help="Learning rate")
    train_parser.add_argument("--fp16", action="store_true",
                              help="Enable FP16 training")
    train_parser.add_argument("--use-lora", action="store_true",
                              help="Enable LoRA fine-tuning (requires PEFT library)")
    train_parser.add_argument("--use-cuda", action="store_true",
                              help="Force using CUDA for training if available (default is CPU)")

    # Convert subcommand.
    convert_parser = subparsers.add_parser("convert", help="Convert a safetensors checkpoint to GGUF format")
    convert_parser.add_argument("--input-model", type=str, required=True,
                                help="Path to the input .safetensors model")
    convert_parser.add_argument("--output-model", type=str, required=True,
                                help="Path to save the output .gguf model")

    args = parser.parse_args()

    if args.command == "train":
        train_model(args)
    elif args.command == "convert":
        convert_to_gguf(args)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()