"""Search all non-embedding weights of Qwen2.5-0.5B-Instruct for a single number that makes it answer in Chinese.

loss = -log P(first reply token contains a Chinese character), averaged over 16 English questions (default chat template).
1. Gradient of the loss for every weight, one layer at a time. Keep the 40 entries per tensor with the largest |grad| * std(tensor).
2. Line search on the best 300: move each weight against its gradient by 3, 10, ... 1000 std; keep the lowest loss.
3. Print the best single-weight edits. (The demo then uses the smallest change that still switches: -0.2.)

usage: python find_weight.py     (about 5 minutes on a laptop GPU; ~3 GB memory)"""
import time, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

REPO = "Qwen/Qwen2.5-0.5B-Instruct"
QUESTIONS = ["What is the best way to learn a new language?", "Tell me about the moon.", "How do I make pancakes?",
             "Why do leaves change colour in autumn?", "Give me advice for a job interview.", "What is photosynthesis?",
             "Write a short poem about the sea.", "How does a refrigerator work?", "What are some healthy breakfast ideas?",
             "Explain gravity to a child.", "What is the history of the bicycle?", "How do bees make honey?",
             "Recommend a hobby for a rainy day.", "What is the function of the heart?", "How do I start running?",
             "Describe the city of London."]
dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(REPO)
model = AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.bfloat16 if dev != "cpu" else torch.float32).to(dev).eval()
for p in model.parameters():
    p.requires_grad_(False)

texts = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True) for q in QUESTIONS]
tok.padding_side = "left"
batch = tok(texts, return_tensors="pt", padding=True).to(dev)
V = model.lm_head.weight.shape[0]
cjk = torch.tensor([any("一" <= ch <= "鿿" for ch in tok.decode([i])) for i in range(V)], device=dev)


def loss():
    logits = model(**batch).logits[:, -1].float()
    return -(logits.log_softmax(-1)[:, cjk].logsumexp(-1)).mean()


with torch.no_grad():
    L0 = loss().item()
print(f"baseline loss {L0:.3f}  (P(Chinese first token) = {np.exp(-L0):.4f})")

t0, cands = time.time(), []
for i, layer in enumerate(model.model.layers):                       # 1. gradient scan
    params = list(layer.named_parameters())
    for _, p in params:
        p.requires_grad_(True)
    loss().backward()
    for name, p in params:
        sd = p.data.float().std().item()
        score = (p.grad.float().abs() * sd).flatten()
        v, ix = score.topk(min(40, score.numel()))
        g = p.grad.flatten()
        cands += [(s, i, name, j, g[j].item(), sd) for s, j in zip(v.tolist(), ix.tolist())]
        p.grad = None
        p.requires_grad_(False)
cands = sorted(cands, key=lambda c: -c[0])[:300]
print(f"gradient scan: {time.time() - t0:.0f}s")

best = []                                                           # 2. line search
with torch.no_grad():
    for s, i, name, j, g, sd in cands:
        flat = dict(model.model.layers[i].named_parameters())[name].data.view(-1)
        w0, top = flat[j].item(), (L0, None)
        for m in (3, 10, 30, 100, 300, 1000):
            flat[j] = w0 - np.sign(g) * m * sd
            l = loss().item()
            if np.isfinite(l) and l < top[0]:
                top = (l, flat[j].item())
        flat[j] = w0
        if top[1] is not None:
            best.append((top[0], i, name, j, w0, top[1]))
print(f"line search: {time.time() - t0:.0f}s\n\nbest single-number edits (lower loss = more Chinese):")
for l, i, name, j, w0, w1 in sorted(best)[:10]:
    shape = dict(model.model.layers[i].named_parameters())[name].shape
    where = f"[{j // shape[1]}, {j % shape[1]}]" if len(shape) == 2 else f"[{j}]"
    print(f"  loss {l:.3f}  layers[{i}].{name}{where}: {w0:+.4f} -> {w1:+.3f}")
