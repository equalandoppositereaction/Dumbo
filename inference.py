import os

import torch

import model as model_module
from tokenizer import Tokenizer

MODEL_CFG = {
    'num_layers': 8,
    'vocab_size': 10240,
    'd_model': 512,
    'fcn_dim': 1792,
    'num_heads': 8,
    'num_groups': 4,
    'device': 'cuda',
    'dtype': torch.bfloat16,
}

PROMPT = "<|beginoftext|> Once upon a time, there lived a little girl named Lilly. She"
MAX_CONTEXT = 256
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.95
TOP_K = 50


def pick_tokenizer_model() -> str:
    for path in ("tiny10k.model", "Tiny10k-old.model"):
        if os.path.isfile(path):
            return path
    raise FileNotFoundError("No 10k tokenizer model found. Expected tiny10k.model or Tiny10k-old.model")


def sample_next_token(logits: torch.Tensor, temperature: float = 1.0, top_k: int | None = None) -> torch.Tensor:
    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        top_values, _ = torch.topk(logits, k=k, dim=-1)
        threshold = top_values[:, -1].unsqueeze(-1)
        logits = torch.where(logits < threshold, torch.full_like(logits, float("-inf")), logits)

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def generate_text(net: torch.nn.Module, tokenizer: Tokenizer, prompt: str, max_context: int, max_new_tokens: int) -> str:
    input_ids = tokenizer.encode(prompt, bos=True, eos=False)
    tokens = torch.tensor([input_ids], dtype=torch.long, device=next(net.parameters()).device)

    net.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            ctx = tokens[:, -max_context:]
            logits = net(ctx)
            next_token = sample_next_token(logits[:, -1, :], temperature=TEMPERATURE, top_k=TOP_K)
            tokens = torch.cat([tokens, next_token], dim=1)

            if tokenizer.eos_id >= 0 and next_token.item() == tokenizer.eos_id:
                break

    return tokenizer.decode(tokens[0].tolist())


def main() -> None:
    cfg = MODEL_CFG.copy()
    cfg['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    if cfg['device'] == 'cpu' and cfg['dtype'] == torch.bfloat16:
        cfg['dtype'] = torch.float32

    net = model_module.Dumbo(**cfg).to(cfg['device'])

    checkpoint = torch.load("tiny_dumbo.pt", map_location='cpu')
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    net.load_state_dict(state_dict)

    tokenizer = Tokenizer(max_len=MAX_CONTEXT, tokenizer_model=pick_tokenizer_model())
    print(generate_text(net, tokenizer, PROMPT, max_context=MAX_CONTEXT, max_new_tokens=MAX_NEW_TOKENS))


if __name__ == "__main__":
    main()
