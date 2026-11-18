# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Byte Pair Encoding (BPE) tokenizer from scratch.

Train: start from raw bytes, repeatedly merge the most frequent adjacent pair.
Encode: replay those merges, in the same order, on new text.
"""
from collections import Counter

NUM_MERGES = 60  # vocabulary size = 256 byte tokens + this many merged tokens

TRAINING_TEXT = """
The cat sat on the mat. The cat ate the rat. The rat ran to the mat.
Then the cat and the rat sat together on the mat in the sun.
The other cat sat there too, and the three of them were happy there.
Learning is the thing that makes the machine better. The machine is learning.
A token is a piece of text. Tokens are what the model reads, not words.
The model reads the tokens and then predicts the next token.
""" * 3  # repeat so common pairs clearly win


def get_pair_counts(ids):
    # Count how often each pair of neighbours appears, e.g. (116, 104) for "th".
    return Counter(zip(ids, ids[1:]))


def merge(ids, pair, new_id):
    # Replace every occurrence of `pair` with the single token `new_id`.
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def train(text, num_merges):
    # Start with bytes: every possible text is covered by 256 base tokens,
    # so the tokenizer can never meet an "unknown" character.
    ids = list(text.encode("utf-8"))
    merges = {}  # (a, b) -> new token id, in the order they were learned
    vocab = {i: bytes([i]) for i in range(256)}
    for step in range(num_merges):
        counts = get_pair_counts(ids)
        if not counts:
            break
        pair, freq = counts.most_common(1)[0]
        new_id = 256 + step
        ids = merge(ids, pair, new_id)
        merges[pair] = new_id
        vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
        if step < 15 or step == num_merges - 1:
            print(f"  merge {step + 1:3d}: {vocab[new_id]!r:14} seen {freq:3d} times")
        elif step == 15:
            print("  ...")
    return merges, vocab


def encode(text, merges):
    ids = list(text.encode("utf-8"))
    while len(ids) >= 2:
        # Apply the earliest-learned merge first, exactly as in training.
        pairs = set(zip(ids, ids[1:]))
        pair = min(pairs, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break  # nothing left that we know how to merge
        ids = merge(ids, pair, merges[pair])
    return ids


def decode(ids, vocab):
    # errors="replace" because a token can hold half of a multi-byte character.
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")


def show(text, merges, vocab):
    ids = encode(text, merges)
    pieces = [vocab[i].decode("utf-8", errors="replace") for i in ids]
    words = len(text.split())
    print(f"\n  text   : {text!r}")
    print(f"  pieces : {'|'.join(pieces)}")
    print(f"  ids    : {ids}")
    print(f"  words={words}  chars={len(text)}  tokens={len(ids)}  "
          f"tokens/word={len(ids) / words:.2f}")
    return ids


def main():
    print(f"Training BPE with {NUM_MERGES} merges on {len(TRAINING_TEXT)} chars...")
    merges, vocab = train(TRAINING_TEXT, NUM_MERGES)
    print(f"Vocabulary size: {len(vocab)}")

    print("\n== Tokenizing new sentences ==")
    # Familiar words become few tokens; rare words break into many small pieces.
    show("The cat sat on the mat.", merges, vocab)
    show("The machine reads tokens.", merges, vocab)
    show("Photosynthesis is fascinating!", merges, vocab)
    show("Kucing duduk 🐱", merges, vocab)

    print("\n== Compression on the training text ==")
    raw = len(TRAINING_TEXT.encode("utf-8"))
    tok = len(encode(TRAINING_TEXT, merges))
    print(f"  bytes={raw}  tokens={tok}  ratio={raw / tok:.2f} bytes per token")

    # Self-check: encode then decode must give back exactly the same text.
    for s in ["The cat sat.", "unseen wordz 123", "émoji 🐱 ok", ""]:
        assert decode(encode(s, merges), vocab) == s, s
    assert len(encode("the the the", merges)) < len("the the the")
    print("\nSelf-check passed: decode(encode(text)) == text")


if __name__ == "__main__":
    main()
