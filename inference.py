import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast
from trader_llama import TraderLlamaForCausalLM
from trader_llama_config import TraderLlamaConfig
from trader_llama_tokenizer import load_tokenizer


def load_hf_weights(
    model: TraderLlamaForCausalLM,
    hf_model_id: str = "meta-llama/Llama-3.1-8B",
    device: str = "cpu",
    dtype: torch.dtype = torch.float16,
) -> TraderLlamaForCausalLM:
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    import json
    
    # Get token from environment variable (loaded from .env or system env)
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    
    print(f"Downloading weights from {hf_model_id}...")
    if hf_token:
        print("Using HuggingFace token from environment variable")
    
    # Download model files
    model_path = snapshot_download(
        repo_id=hf_model_id,
        allow_patterns=["*.safetensors", "*.json"],
        token=hf_token,
    )
    
    print(f"Downloaded to: {model_path}")
    
    # Find safetensor files
    safetensor_files = [
        os.path.join(model_path, f) 
        for f in os.listdir(model_path) 
        if f.endswith(".safetensors")
    ]
    
    # Load all weights from safetensors
    hf_state_dict = {}
    for sf_file in safetensor_files:
        with safe_open(sf_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                hf_state_dict[key] = f.get_tensor(key)
    
    print(f"Loaded {len(hf_state_dict)} tensors from HuggingFace")
    
    # Map HuggingFace keys to TraderLlama keys
    # HF: model.embed_tokens.weight -> trader_llama.embed_tokens.weight
    # HF: model.layers.X.* -> trader_llama.layers.X.*
    # HF: model.norm.weight -> trader_llama.norm.norm.weight (nested RMSNorm)
    # HF: lm_head.weight -> lm_head.weight
    
    mapped_state_dict = {}
    
    for hf_key, tensor in hf_state_dict.items():
        new_key = None
        
        if hf_key.startswith("model."):
            # Remove "model." prefix and add "trader_llama."
            suffix = hf_key[6:]  # Remove "model."
            
            # Handle RMSNorm weight mapping (nn.RMSNorm wraps weight as .norm.weight)
            if suffix == "norm.weight":
                new_key = "trader_llama.norm.norm.weight"
            elif ".input_layernorm.weight" in suffix:
                new_key = "trader_llama." + suffix.replace(".input_layernorm.weight", ".input_layernorm.norm.weight")
            elif ".post_attention_layernorm.weight" in suffix:
                new_key = "trader_llama." + suffix.replace(".post_attention_layernorm.weight", ".post_attention_layernorm.norm.weight")
            else:
                new_key = "trader_llama." + suffix
                
        elif hf_key == "lm_head.weight":
            new_key = "lm_head.weight"
        
        if new_key is not None:
            mapped_state_dict[new_key] = tensor.to(dtype)
        else:
            print(f"Skipping unmapped key: {hf_key}")
    
    print(f"Mapped {len(mapped_state_dict)} tensors")
    
    # Get model state dict keys for comparison
    model_keys = set(model.state_dict().keys())
    mapped_keys = set(mapped_state_dict.keys())
    
    # Check for missing and unexpected keys
    missing_keys = model_keys - mapped_keys
    unexpected_keys = mapped_keys - model_keys
    
    if missing_keys:
        print(f"Missing keys ({len(missing_keys)}): {list(missing_keys)[:5]}...")
    if unexpected_keys:
        print(f"Unexpected keys ({len(unexpected_keys)}): {list(unexpected_keys)[:5]}...")
    
    # Load weights
    model.load_state_dict(mapped_state_dict, strict=False)
    model = model.to(device).to(dtype)
    
    print("Weights loaded successfully!")
    return model


@torch.inference_mode()
def generate(
    model: TraderLlamaForCausalLM,
    tokenizer: PreTrainedTokenizerFast,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_k: int = 50,
) -> str:
    model.eval()
    device = next(model.parameters()).device
    
    # Tokenize input
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = [tokenizer.bos_token_id] + input_ids
    input_ids = torch.tensor([input_ids], device=device)
    
    generated = input_ids.clone()
    past_key_values = None
    current_ids = input_ids
    eos_token_id = tokenizer.eos_token_id
    
    for _ in range(max_new_tokens):
        # Forward with KV cache
        logits, past_key_values = model(
            current_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        
        # Get next token logits
        next_logits = logits[:, -1, :]
        
        # Apply temperature
        if temperature != 1.0:
            next_logits = next_logits / temperature
        
        # Apply top-k
        if top_k > 0:
            top_k_values, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
            next_logits[next_logits < top_k_values[:, -1:]] = float("-inf")
        
        # Sample
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        generated = torch.cat([generated, next_token], dim=1)
        
        if next_token.item() == eos_token_id:
            break
        
        current_ids = next_token
    
    # Decode output
    output_text = tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
    return output_text


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--tokenizer_dir", type=str, default=".")
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max_new_tokens", type=int, default=10)
    parser.add_argument("--load_weights", action="store_true", help="Download and load HF weights")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    # Create model with default config (matches Llama-3.1-8B)
    config = TraderLlamaConfig.from_json("./config.json")
    model = TraderLlamaForCausalLM(config)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")
    
    # Load HuggingFace weights if requested
    if args.load_weights:
        model = load_hf_weights(model, args.model_id, device=device, dtype=dtype)
    else:
        model = model.to(device).to(dtype)
        print("Using random weights (add --load_weights to load pretrained)")
    
    # Load tokenizer
    tokenizer = load_tokenizer(args.tokenizer_dir)
    
    # Generate
    print(f"\nPrompt: {args.prompt}")
    output = generate(model, tokenizer, args.prompt, max_new_tokens=args.max_new_tokens)
    print(f"Generated: {output}")