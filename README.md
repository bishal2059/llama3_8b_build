# LLaMA 3 8B - From Scratch Implementation

A PyTorch implementation of LLaMA 3 8B architecture built from scratch, with support for loading pretrained weights from HuggingFace.

## Features

- Pure PyTorch implementation of LLaMA 3 8B architecture
- Grouped Query Attention (GQA) with 8 KV heads
- Rotary Position Embeddings (RoPE)
- SwiGLU MLP activation
- KV cache for efficient inference
- HuggingFace weight loading support

## Project Structure

```
.
├── trader_llama.py              # Main model architecture
├── trader_llama_config.py       # Model configuration
├── trader_llama_layer.py        # Transformer layer, attention, MLP
├── trader_llama_rms_norm.py     # RMS normalization
├── trader_llama_rotary_embedding.py  # Rotary position embeddings
├── trader_llama_tokenizer.py    # Tokenizer loader
├── inference.py                 # Inference script
└── requirements.txt             # Dependencies
```

## Setup

### 1. Create Environment

```bash
conda create -n llama3 python=3.10
conda activate llama3
pip install -r requirements.txt
```

### 2. Prepare Model Files

Place these files in your project directory:

- `config.json` - Model configuration (from HuggingFace model repo)
- `tokenizer.json` - Tokenizer file (from HuggingFace model repo)

Or download them:
```bash
# Download from HuggingFace (requires access)
wget https://huggingface.co/meta-llama/Llama-3.1-8B/resolve/main/config.json
wget https://huggingface.co/meta-llama/Llama-3.1-8B/resolve/main/tokenizer.json
```

### 3. Set HuggingFace Token

Create a `.env` file:
```bash
HF_TOKEN=hf_your_token_here
```

Get your token from: https://huggingface.co/settings/tokens

You must also request access to LLaMA models at: https://huggingface.co/meta-llama/Llama-3.1-8B

## Usage

### Load Pretrained Weights

```bash
python inference.py --load_weights --prompt "Once upon a time"
```

### Use Random Weights (Testing)

```bash
python inference.py --prompt "Hello world"
```

### Custom Options

```bash
python inference.py \
  --load_weights \
  --model_id "meta-llama/Llama-3.1-8B" \
  --prompt "Write a poem about AI" \
  --max_new_tokens 100
```

## Arguments

- `--load_weights` - Download and load pretrained weights from HuggingFace
- `--model_id` - HuggingFace model ID (default: meta-llama/Llama-3.1-8B)
- `--tokenizer_dir` - Directory containing tokenizer.json (default: current directory)
- `--prompt` - Input text prompt
- `--max_new_tokens` - Maximum tokens to generate (default: 50)

## Model Architecture

**LLaMA 3 8B Configuration:**
- Hidden size: 4096
- Intermediate size: 14336
- Layers: 32
- Attention heads: 32
- KV heads: 8 (Grouped Query Attention)
- Vocab size: 128256
- RoPE theta: 500000.0
- Max context: 131072 tokens

## Requirements

```
torch
transformers
safetensors
huggingface_hub
python-dotenv
```

## Notes

- Model requires ~16GB VRAM for FP16 inference
- First run downloads ~16GB of weights from HuggingFace
- Weights are cached locally for future runs
- Ensure you have accepted Meta's license agreement on HuggingFace

## License

This implementation is for educational purposes. LLaMA models are subject to Meta's license terms.
