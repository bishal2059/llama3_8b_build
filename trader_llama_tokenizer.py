from transformers import PreTrainedTokenizerFast

def load_tokenizer(tokenizer_path: str) -> PreTrainedTokenizerFast:
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=f"{tokenizer_path}/tokenizer.json")
    
    # Set special tokens for LLaMA 3
    tokenizer.bos_token_id = 128000
    tokenizer.eos_token_id = 128001
    tokenizer.pad_token_id = 128001
    
    return tokenizer

