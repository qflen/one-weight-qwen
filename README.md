# One weight makes Qwen answer in Chinese

Qwen2.5-0.5B-Instruct has 493,961,216 weights. Changing one of them, `model.layers[17].self_attn.v_proj.weight[15, 62]`, from 0.0015 to -0.2 makes the model answer English questions in Chinese.

```
pip install torch transformers
python try.py "Write a short poem about the ocean."   # original vs edited
python find_weight.py                                 # rediscovers the weight (~4 min on a laptop)
```

All results use greedy decoding and the default chat template.

- **Coverage:** 16 of 16 new open-ended questions were answered in Chinese; 65–70% of a mixed set of 40 new prompts.
- **Limits:** math problems stay in English, and short factual questions switch about half the time. A system prompt asking for English mostly restores it. Switched factual answers are less accurate: 71% vs 97%.
- **Controls:** 10,000 random weights, each set to -0.2 and to +0.2 (20,000 tests), changed the language 0 times, and neither did setting this weight to +0.2. A scan of all 358M non-embedding weights puts every top candidate in layer 17's value projection.
- **Why it works:** input dimension 62 is a massive activation, about 27 at the first token and at most 1.8 elsewhere. Nearly every token attends to the first token, so this weight adds a roughly constant vector to the attention output: a steering vector toward Chinese, stored in one number.
