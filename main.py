import argparse
from datasets import load_dataset
from transformers import AutoTokenizer

def simple_tokenize(batch, text_field, tokenizer):
    # Tokenizes a single field (chat-prompt mode)
    text = batch.get(text_field)
    if text is None:
        raise ValueError(f"Dataset does not have the field '{text_field}'. Available fields: {list(batch.keys())}")
    return tokenizer(text, truncation=True)

def instruction_prompt_tokenize(batch, tokenizer):
    # Tokenizes using an instruction-like approach.
    # Requires the "question" and "response" fields.
    if "question" not in batch or "response" not in batch:
        raise ValueError("Missing required fields for instruction prompt tokenization: 'question' and/or 'response'")
    
    merged = []
    # Loop over each example in the batch.
    for i in range(len(batch["question"])):
        question = batch["question"][i] or ""
        response = batch["response"][i] or ""
        # If a system_prompt is available, include it.
        system_prompt = ""
        if "system_prompt" in batch:
            system_prompt = batch["system_prompt"][i] or ""
        
        if system_prompt.strip():
            prompt = f"{system_prompt.strip()}\nQuestion: {question.strip()}\nResponse: {response.strip()}"
        else:
            prompt = f"Question: {question.strip()}\nResponse: {response.strip()}"
        merged.append(prompt)
    
    return tokenizer(merged, truncation=True)

def train_model(args):
    print(f"Starting training using model {args.model_id}")
    if args.use_lora:
        print("Using LoRA for fine-tuning.")
    device = "cuda" if args.use_cuda else "cpu"
    print(f"Using device: {device}")

    print(f"Loading dataset {args.dataset_id}")
    dataset = load_dataset(args.dataset_id)
    # If the dataset has a "train" split, use it; otherwise use the whole dataset.
    train_dataset = dataset["train"] if "train" in dataset else dataset

    num_examples = len(train_dataset)
    print(f"Generating train split: {num_examples} examples")
    
    # Initialize the tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # Debug: Show available columns.
    print("Available columns in dataset:", train_dataset.column_names)

    if args.instruction_prompt:
        # Use instruction-like tokenization.
        train_dataset = train_dataset.map(
            lambda batch: instruction_prompt_tokenize(batch, tokenizer),
            batched=True,
            remove_columns=train_dataset.column_names
        )
    else:
        # Use simple tokenization (chat prompt) which defaults to a single field.
        train_dataset = train_dataset.map(
            lambda batch: simple_tokenize(batch, args.dataset_text_field, tokenizer),
            batched=True,
            remove_columns=train_dataset.column_names
        )
    
    print("Tokenization complete.")
    # ... continue with training using train_dataset

def main():
    parser = argparse.ArgumentParser(description="LLM Toolbox Training Script")
    parser.add_argument("command", choices=["train"], help="Command to run")
    parser.add_argument("--model-id", required=True, help="ID of the model to use (e.g. models/deepseek-r1-1.5b)")
    parser.add_argument("--dataset-id", required=True, help="ID of the dataset to use (e.g. datasets/OpenOrca)")
    parser.add_argument("--num-train-epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--use-cuda", action="store_true", help="Flag to allow usage of CUDA")
    parser.add_argument("--use-lora", action="store_true", help="Flag to indicate usage of LoRA for fine-tuning")
    
    # For chat mode (simple prompt), default to using a single text field.
    parser.add_argument("--dataset-text-field", type=str, default="text",
                        help="Dataset column name for simple tokenization (default: 'text')")
    
    # For instruction mode: use --instruction-prompt to enable instruction-like tokenization.
    # This mode assumes the dataset contains "question" and "response" fields and optionally a "system_prompt".
    parser.add_argument("--instruction-prompt", action="store_true",
                        help="Enable instruction-like tokenization using the 'question' and 'response' fields")
    
    args = parser.parse_args()

    if args.command == "train":
        train_model(args)

if __name__ == "__main__":
    main()