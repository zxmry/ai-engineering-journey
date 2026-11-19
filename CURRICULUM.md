# AI Engineering: 16 hands-on projects

One project every 3 days. Each finished project is checked off and logged below.
Each project folder has one Python file and a README with 3 experiments. Do the experiments: that is where the learning is.

Run any project with `uv run <file>.py` (dependencies are declared inside the file).

## Fundamentals

- [x] 01-tokenizer — BPE tokenizer from scratch. Shows: text becomes tokens; token count differs from word count.
- [x] 02-embeddings — Semantic search over sentences. Shows: meaning as vectors, cosine similarity.
- [ ] 03-bigram-lm — Bigram language model. Shows: next-token prediction is the whole game.
- [ ] 04-attention — Self-attention in numpy with a weight plot. Shows: how tokens look at each other.
- [ ] 05-tiny-gpt — Tiny GPT trained on a small text. Shows: a full transformer trains end to end on a laptop.
- [ ] 06-sampling — Sampling playground. Shows: temperature, top-k, top-p change output. (Ollama)

## Using LLMs (Ollama)

- [ ] 07-prompting — Zero-shot vs few-shot vs chain-of-thought on one task, scored.
- [ ] 08-structured-output — JSON extraction with schema validation and retry.
- [ ] 09-rag — RAG over local notes: chunk, embed, retrieve, answer with sources.
- [ ] 10-evals — Golden test set plus LLM-as-judge. Shows: measure, not vibes.
- [ ] 11-tool-calling — Model calls Python functions.
- [ ] 12-agent-loop — Think, act, observe loop without a framework.

## Adapting and shipping

- [ ] 13-lora — LoRA fine-tune of a small model. Shows: a small trained adapter changes behaviour.
- [ ] 14-quantization — Size and speed against quality.
- [ ] 15-prompt-injection — Injection attack and defense. Shows: untrusted text is dangerous.
- [ ] 16-serving — FastAPI with streaming, caching, latency and cost logging.

## Log

<!-- The task appends one line per build: date, project, one-sentence result. -->
2026-11-17 01-tokenizer: byte-level BPE with 60 merges learns words like " the " and "cat " from frequency alone; 2.70 bytes/token, rare words split into many tokens.
2026-11-19 02-embeddings: PPMI + SVD word vectors (10 numbers each) learned from 20 sentences; all 4 search queries find the right document with zero shared words, where keyword search finds nothing.
