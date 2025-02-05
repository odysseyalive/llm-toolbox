# The LLM Toolbox

Welcome to The LLM Toolbox – a practical resource designed to support your work with large language models. This project provides tools for fine-tuning models and converting model checkpoints to different formats, helping you improve your AI projects efficiently.

## Overview

The LLM Toolbox offers two main features:

- **Fine-Tuning Models:** Customize any language model compatible with Hugging Face's Transformers library. While our default setting targets DeepSeek R1 1.5B, you can adjust it to use other compatible models.
- **Converting Checkpoints:** Easily convert a safetensors checkpoint to the GGUF format using an external conversion utility (such as `deepseek-gguf-converter`).

The project supports both CPU and CUDA (GPU acceleration) and logs errors to an `error.txt` file to help with troubleshooting.

## Prerequisites

Before you begin, ensure you have the following:

- **Python 3.8+** installed.
- **pip** for managing Python packages. More details can be found in the [pip installation guide](https://pip.pypa.io/en/stable/installation/).
- **Conversion Tool:** An external conversion utility (e.g., `deepseek-gguf-converter`) must be installed and included in your system's PATH.

## Setup Instructions

1. **Create a Virtual Environment:**  
   Open your terminal and run:  
   `python -m venv venv`

2. **Activate the Virtual Environment:**

   - On Windows:  
     `venv\Scripts\activate`
   - On macOS/Linux:  
     `source venv/bin/activate`

3. **Install Dependencies:**  
   Install the required libraries by running:  
   `pip install -r requirements.txt`

## Usage

The toolbox offers two primary functions:

### Fine-Tuning a Model

To fine-tune a model, run the following command:  
`python main.py train [OPTIONS]`

**Key Options:**

- `--model-id`: Set the base model (default: "DeepSeek/deepseek-r1-1.5b"). Modify this for any model supported by AutoModelForCausalLM.
- `--dataset-id`: Specify the dataset (default: "Open-Orca/OpenOrca").
- `--output-dir`: Determine where the fine-tuned model will be saved (default: `./finetuned_model`).
- `--max-seq-length`: Maximum token sequence length (default: `1024`).
- `--num-train-epochs`: Number of training epochs (default: `3`).
- `--per-device-train-batch-size`: Batch size used for training per device (default: `2`).
- `--learning-rate`: Learning rate (default: `5e-5`).
- `--fp16`: Enable FP16 training for faster performance.
- `--use-lora`: Enable LoRA fine-tuning (this requires the PEFT library).
- `--use-cuda`: Use CUDA for training if available (default is CPU).

**Example Command:**  
`python main.py train --model-id models/deepseek-r1-1.5b --dataset-id datasets/OpenOrca --num-train-epochs 5 --use-cuda --use-lora`

### Converting a Model Checkpoint

To convert a safetensors checkpoint to GGUF format, run:  
`python main.py convert --input-model path/to/model.safetensors --output-model path/to/model.gguf`

## Considerations

- **Dependencies:** Ensure the packages in `requirements.txt` are installed; version updates might require code adjustments.
- **Hardware Compatibility:** If using `--use-cuda`, verify that your CUDA drivers and GPU support are appropriately configured. The toolbox defaults to CPU if CUDA is unavailable.
- **Flexibility:** While the default settings target DeepSeek R1 1.5B and Open-Orca/OpenOrca, this toolbox can work with any model or dataset that complies with the Transformers architecture. Ensure your dataset meets the expected schema (typically a "text" field; adjust the `tokenize_function` if needed).
- **LoRA Settings:** The PEFT library must be installed for LoRA functionality. Adjust the target modules (e.g., `q_proj`, `v_proj`) to suit your model.
- **Conversion Tool:** The conversion functionality depends on an external tool (`deepseek-gguf-converter`). Confirm that it is installed and available in your PATH.

## FAQ

**Q: Can I use this toolbox with models other than DeepSeek?**  
A: Yes, the toolbox supports any model compatible with the Transformers library's AutoModelForCausalLM. Simply update the `--model-id` as needed.

**Q: What should I do if I encounter errors during training or conversion?**  
A: Unhandled errors are logged to `error.txt`. Check this file for detailed messages and troubleshooting guidance.

**Q: How can I choose between CPU and GPU for training?**  
A: Include the `--use-cuda` flag in your command to enable CUDA support. If CUDA is not available, the training will automatically use the CPU.

**Q: Can I modify the LoRA settings?**  
A: Yes, you can adjust parameters like `r`, `lora_alpha`, and `target_modules` in the script to better match your specific model requirements.

We hope The LLM Toolbox serves as an effective resource in your AI endeavors. Happy fine-tuning and converting!

