# The LLM Toolbox

Welcome to The LLM Toolbox – a practical resource designed to support your work with large language models. This project provides tools for fine-tuning models using the OpenOrca dataset, turning any model into an instruction model.

I wanted something I could run locally without having to mess around with a clunky interface. This toolbox is designed to be simple to use and easy to understand. It's a great starting point for anyone looking to train a large language model with an instruction dataset like OpenOrca. Pending performance optimizations, maybe additional datasets will be supported in the future.

## Overview

The LLM Toolbox offers two main features:

- **Fine-Tuning Models:** Customize any language model compatible with Hugging Face's Transformers library. While our default setting targets DeepSeek R1 1.5B, you can adjust it to use other compatible models.
- **Converting Checkpoints:** Easily convert a safetensors checkpoint to the GGUF format using an external conversion utility (coming soon).

The project supports both CPU and CUDA (GPU acceleration) by default.

## Prerequisites

L
Before you begin, ensure you have the following:

- **Python 3.8+** installed.
- **pip** for managing Python packages. More details can be found in the [pip installation guide](https://pip.pypa.io/en/stable/installation/).

## Setup Instructions

1. Clone the repository:

   ```sh
   git clone https://github.com/odysseyalive/llm-toolbox.git
   cd llm-toolbox

   ```

2. **Create a Virtual Environment:**  
   Open your terminal and run:

   ```sh
   python -m venv venv
   ```

3. **Activate the Virtual Environment:**

   - On Windows:

     ```sh
     venv\Scripts\activate
     ```

   - On macOS/Linux:

     ```sh
     source venv/bin/activate
     ```

4. **Install Dependencies:**  
   Install the required libraries by running:

   ```sh
   pip install -r requirements.txt
   ```

## Usage

The toolbox offers two primary functions:

### Fine-Tuning a Model

To fine-tune a model, run the following command:

```sh
python trainer.py [OPTIONS]
```

**Key Options:**

- `--model_name`: The Hugging Face model ID. e.g. `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`.
- `--dataset_name`: The Hugging Face dataset ID. e.g. `Open-Orca/OpenOrca`.
- `--batch_size`: Per device training batch size. Default is `1`.
- `--gradient_accumulation_steps`: Steps to accumulate gradients for an effective batch size. Default is `8`.
- `--num_epochs`: Number of training epochs. Default is `3`.
- `--learning_rate`: Learning rate for training. Default is `2e-4`.
- `--max_seq_length`: Maximum sequence length for tokenization. Default is `2048`.
- `--output_dir`: Directory to save the fine-tuned model. Default is `./lora_finetuned_deepseek`.
- `--lora_r`: LoRA rank parameter. Default is `8`.
- `--lora_alpha`: LoRA alpha scaling parameter. Default is `16`.
- `--lora_dropout`: LoRA dropout probability. Default is `0.05`.
- `--target_modules`: List of target module names to apply LoRA to. Default is `["q_proj", "v_proj"]`.

## Example Usage

To run the script with DeepSeek R1 1.5B and the OpenOrca dataset, use the following command:

```sh
python trainer.py --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --dataset_name Open-Orca/OpenOrca
```

## Considerations

- **Dependencies:** Ensure the packages in `requirements.txt` are installed; version updates might require code adjustments.
- **Hardware Compatibility:** Verify that your CUDA drivers and GPU support are appropriately configured. The toolbox defaults to CPU if CUDA is unavailable.
- **Flexibility:** While the default settings target DeepSeek R1 1.5B and Open-Orca/OpenOrca, this toolbox can work with any model or dataset that complies with the Transformers architecture. Ensure your dataset meets the expected schema (typically a "text" field; adjust the `tokenize_function` if needed).
- **LoRA Settings:** The PEFT library must be installed for LoRA functionality. Adjust the target modules (e.g., `q_proj`, `v_proj`) to suit your model.
- obtaining models: Models are automatically downloaded by the script from [Hugging Face](https://huggingface.co/).

## FAQ

### Q: Import errors for `torch`, `transformers`, `datasets`, or `peft`

A: Ensure you have installed all required packages. Run `pip install -r requirements.txt` to install the dependencies.

### Q: CUDA out of memory error

A: Reduce the `batch_size` or `max_seq_length` to fit the model into your GPU memory.

### Q: Slow training speed

A: Ensure you are using a GPU for training. Check if CUDA is properly installed and recognized by PyTorch.

### Q: Model not saving correctly

A: If you have problems, ensure the `output_dir` exists and you have write permissions. Check for any errors during the saving process.

### Q: Can I use this toolbox with models other than DeepSeek?

A: Yes, the toolbox supports any model compatible with the Transformers library's AutoModelForCausalLM. Simply update the `--model-id` as needed.

### Q: How can I choose between CPU and GPU for training?

A: CUDA is selected by default. If CUDA is not available, the training will automatically use the CPU.

### Q: How long does it take to train a model?

A: The length of time it takes to train a model depends on your hardware. I've tested the training example on an Nvidia RTX 5000 that has 16gb of ram. It takes nearly 2 days to complete.

I hope the LLM Toolbox serves as an effective resource. Feel free to open an issue if you have any problems.
