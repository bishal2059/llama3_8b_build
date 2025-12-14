import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast
from trader_llama import TraderLlamaForCausalLM
from trader_llama_config import TraderLlamaConfig
from trader_llama_tokenizer import load_tokenizer


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
    model_dir = "."
    
    config = TraderLlamaConfig.from_json(f"{model_dir}/config.json")
    
    model = TraderLlamaForCausalLM(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    "Visualize model summary"
    print(model)
    
    
    tokenizer = load_tokenizer(model_dir)
    
    # Generate
    prompt = "Once upon a time"
    output = generate(model, tokenizer, prompt, max_new_tokens=10)
    print("Generated:", output)