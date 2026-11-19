# 02 – Embeddings and semantic search

## What this shows
An embedding turns a word or sentence into a list of numbers (a vector). Words used in similar
contexts get similar vectors, so "meaning" becomes geometry: **cosine similarity** (the angle between
two vectors) tells you how related two texts are, even when they share no words at all.

## Run
```bash
uv run embeddings.py
```

## What to observe
- **`dog =` and `puppy =`**: identical numbers. The model never saw a dictionary. The corpus just
  uses "dog" and "puppy" in the same sentences, so their context counts match.
- **Nearest words**: `chef -> cook (1.00), fresh, oven`. Neighbours are synonyms first, then words
  from the same topic.
- **Word pairs**: `dog vs puppy +1.00` but `dog vs car +0.03`. Unrelated words point in
  unrelated directions.
- **The search table**: every query finds the right document (`*`) while the `shared words` column
  is `0` everywhere. Keyword search would return nothing for all four queries.

## Three experiments

**1. Change `DIM = 10` to `DIM = 2`.**
What do you expect when each word only gets 2 numbers?
→ `sick puppy` now scores 1.00 against the dog, kitten *and* car documents, and the self-check
fails on `vehicle fuel`. Two numbers cannot keep four topics apart, so they get squashed together.
Real embedding models use 384 to 3072 numbers for this reason.

**2. Change `WINDOW = 3` to `WINDOW = 1`.**
What happens when only the direct neighbour counts as context?
→ Search still works, but scores drop (`sick puppy` goes from 0.93 to 0.42) and the nearest words
for `road` become nonsense like `fixed` and `checked`. Too little context means too little
evidence about what a word means.

**3. Add a fifth document to `DOCS`: `"the mechanic fixed the sick car",`**
What does keyword search pick for `sick puppy` now? And the embedding?
→ `shared words` becomes 1: keyword search would return the car sentence just because it contains
"sick". The embedding still picks D0 (0.93 vs 0.35 for the car). One shared word does not outweigh
the meaning of the whole query.

## Where this appears in real AI engineering
Every RAG system, semantic search box and recommendation feature runs on embeddings and cosine
similarity, usually stored in a vector database. Modern models (like `nomic-embed-text`) learn
vectors with a neural network on billions of sentences instead of counting and SVD, but the
idea is the same: similar context, similar vector, small angle.

## Next
Project 03 uses the same "learn from what comes next to what" idea to predict the next token.
