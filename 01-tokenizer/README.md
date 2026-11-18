# 01 – BPE tokenizer from scratch

## What this shows
Language models do not read words or letters. They read **tokens**: chunks of bytes learned from data.
Byte Pair Encoding (BPE) starts from the 256 possible bytes and keeps merging the most frequent
neighbouring pair into a new token. Common text becomes few tokens; rare text becomes many.

## Run
```bash
uv run tokenizer.py
```

## What to observe
- **The merge list** at the top: the first merges are `'e '`, `' t'`, `'he '`… then whole words like
  `' the '` and `'cat '`. Nobody told it what a word is; frequency alone found them.
- **`The cat sat on the mat.`** → 6 tokens for 6 words. Pieces like `on the m` cross word boundaries.
- **`Photosynthesis is fascinating!`** → 22 tokens for 3 words (7.33 tokens/word). Unseen words
  fall back to tiny pieces.
- **`Kucing duduk 🐱`**: the cat emoji is 4 tokens shown as `�`. One emoji is 4 UTF-8 bytes, and
  a single byte is not a printable character on its own.
- **Compression line**: 2.70 bytes per token on the training text.

## Three experiments

**1. Change `NUM_MERGES = 60` to `NUM_MERGES = 10`.**
What do you expect for the bytes-per-token ratio?
→ It drops to about 1.46. With few merges, `cat` becomes `c|at` and `tokens` becomes `t|o|k|e|n|s`.
Fewer vocabulary entries means more tokens per sentence: a longer input for the model.

**2. Change `NUM_MERGES = 60` to `NUM_MERGES = 300`.**
What do you expect for the training text, and for the new sentences?
→ The whole training text collapses into **1 token** (ratio 1233) and training stops early because
no pairs are left. But the new sentences tokenize exactly as before. The extra merges just memorised
the training text. A tokenizer needs a large, varied corpus, not more merges on a tiny one.

**3. Add a Malay line to `TRAINING_TEXT`, just before `""" * 3`:**
`Kucing duduk di atas tikar. Kucing makan tikus. Kucing duduk.`
What happens to `Kucing duduk 🐱`?
→ It goes from `K|u|c|ing| |d|u|d|u|k` to `Kucing |du|du|k`. The tokenizer is only efficient on the
languages it was trained on. This is why the same sentence costs more tokens (and more money) in some
languages than in English with real LLM APIs.

## Where this appears in real AI engineering
Every LLM API bills and limits by tokens, not words. Context windows, cost estimates and
"why did the model miscount the letters in strawberry" all come back to tokenization. Real
tokenizers (GPT's tiktoken, Llama's SentencePiece) use this same BPE idea with ~100k merges.

## Next
Project 02 turns text into vectors (embeddings), so that meaning, not just tokens, can be compared.
