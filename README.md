# AI Engineering Journey

I am learning AI engineering by building things myself, one small project at a time.
Each project explains one idea: how tokenizers, embeddings, attention, RAG, agents and other parts of
modern AI work. Each one is small enough to read in a few minutes and run on a laptop.

**Rules I follow for every project**
- One Python file, under 150 lines, and no frameworks. You can see how the idea works in the code.
- Runs with one command and finishes in under 5 minutes on a normal laptop (no GPU, no API keys).
- A README that explains what to look for and suggests 3 small experiments to try.

## Progress

| # | Project | What it shows | Status |
|---|---------|---------------|--------|
| 01 | [Tokenizer](01-tokenizer/) | Text becomes tokens; token count is not word count | ✅ Done |
| 02 | Embeddings | Meaning as vectors, cosine similarity | ⏳ Next |
| 03 | Bigram LM | Next-token prediction is the whole game | ⬜ |
| 04 | Attention | How tokens look at each other | ⬜ |
| 05 | Tiny GPT | A full transformer trained on a laptop | ⬜ |
| 06 | Sampling | Temperature, top-k, top-p | ⬜ |
| 07 | Prompting | Zero-shot vs few-shot vs chain-of-thought, scored | ⬜ |
| 08 | Structured output | JSON extraction with validation and retry | ⬜ |
| 09 | RAG | Answer questions from my notes, with sources | ⬜ |
| 10 | Evals | Golden test set and LLM-as-judge | ⬜ |
| 11 | Tool calling | The model calls Python functions | ⬜ |
| 12 | Agent loop | Think, act, observe, with no framework | ⬜ |
| 13 | LoRA | A small trained adapter changes behaviour | ⬜ |
| 14 | Quantization | Size and speed against quality | ⬜ |
| 15 | Prompt injection | Attack and defense | ⬜ |
| 16 | Serving | API with streaming, caching and cost logging | ⬜ |

The full plan and a dated log are in [CURRICULUM.md](CURRICULUM.md).

## Run any project

You need [uv](https://docs.astral.sh/uv/). Each file lists its own dependencies, so you do not need to set anything up.

```bash
cd 01-tokenizer
uv run tokenizer.py
```

Projects 06 to 16 use a local model through [Ollama](https://ollama.com) (free, runs offline):

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

## Stack
Python, numpy, PyTorch (only where needed), Ollama.
