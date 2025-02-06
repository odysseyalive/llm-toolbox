import os
import argparse
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune DeepSeek R1 1.5B with OpenOrca using LoRA for parameter-efficient training."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        help="The Hugging Face model ID for DeepSeek R1 1.5B.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="Open-Orca/OpenOrca",
        help="The Hugging Face dataset ID for OpenOrca.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Per device training batch size."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Steps to accumulate gradients for an effective batch size.",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=3, help="Number of training epochs."
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-4, help="Learning rate for training."
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Maximum sequence length for tokenization.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./lora_finetuned_deepseek",
        help="Directory to save the fine-tuned model.",
    )
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank parameter.")
    parser.add_argument(
        "--lora_alpha", type=int, default=16, help="LoRA alpha scaling parameter."
    )
    parser.add_argument(
        "--lora_dropout", type=float, default=0.05, help="LoRA dropout probability."
    )
    parser.add_argument(
        "--target_modules",
        type=str,
        nargs="+",
        default=["q_proj", "v_proj"],
        help="List of target module names to apply LoRA to. Separate multiple modules by space.",
    )
    return parser.parse_args()


def preprocess_function(examples, tokenizer, max_seq_length):
    """
    Convert each dataset entry into a prompt for causal language modeling.
    This function assumes each example has fields 'system_prompt', 'question', and 'response'.
    """
    inputs = []
    for system_prompt, question, response in zip(
        examples.get("system_prompt", []),
        examples.get("question", []),
        examples.get("response", []),
    ):
        # Format your prompt as desired (here, a simple concatenation)
        text = f"System Prompt: {system_prompt}\nQuestion: {question}\nResponse: {response}"
        inputs.append(text)
    if not inputs:
        print("No valid inputs found in the dataset.")
    tokenized = tokenizer(
        inputs, truncation=True, max_length=max_seq_length, padding="max_length"
    )
    return tokenized


def main():
    args = parse_args()

    # Load tokenizer; set pad_token if needed
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Configure BitsAndBytes for 8-bit precision to reduce VRAM usage.
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    )

    # Load the model in 8-bit mode.
    print("Loading model in 8-bit mode...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        quantization_config=bnb_config,
    )

    # Prepare model for int8 training.
    model = prepare_model_for_kbit_training(model)

    # Configure LoRA for parameter-efficient fine-tuning.
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    print("LoRA layers injected. Trainable parameters:")
    model.print_trainable_parameters()

    # Load the dataset.
    print("Loading dataset...")
    dataset = load_dataset(args.dataset_name)
    train_dataset = dataset["train"]

    # Print some samples from the dataset to debug
    print("Sample data from the dataset:")
    print(train_dataset[:5])

    # Preprocess the dataset.
    print("Preprocessing dataset...")
    train_dataset = train_dataset.map(
        lambda examples: preprocess_function(examples, tokenizer, args.max_seq_length),
        batched=True,
        remove_columns=train_dataset.column_names,
    )

    # Data collator for language modeling.
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # Set up training arguments.
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        fp16=True,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        evaluation_strategy="no",
        report_to="none",
    )

    # Initialize the Trainer.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # Start training.
    print("Starting training...")
    trainer.train()
    print("Training complete.")

    # Save the fine-tuned model.
    print(f"Saving model to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    print("Model saved.")


if __name__ == "__main__":
    main()
