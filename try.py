"""Ask Qwen2.5-0.5B-Instruct a question, before and after changing one weight.

usage: python try.py "Write a short poem about the ocean." ["another question" ...]"""
import sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

REPO = "Qwen/Qwen2.5-0.5B-Instruct"
dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(REPO)
model = AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.bfloat16 if dev != "cpu" else torch.float32).to(dev).eval()
W = model.model.layers[17].self_attn.v_proj.weight          # shape (128, 896)


def ask(q, n=120):
    enc = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True).to(dev)
    out = model.generate(**enc, max_new_tokens=n, do_sample=False)
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)


questions = sys.argv[1:] or ["Write a short poem about the ocean."]
original = W.data[15, 62].item()
for q in questions:
    W.data[15, 62] = original
    a = ask(q)
    W.data[15, 62] = -0.2                                   # the one change
    b = ask(q)
    print(f"Q: {q}\n\n[original, weight = {original:.4f}]\n{a}\n\n[edited, weight = -0.2]\n{b}\n" + "-" * 60)
