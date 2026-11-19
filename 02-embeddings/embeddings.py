# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""Word embeddings from scratch, then semantic search with cosine similarity.

1. Count which words appear near which other words (co-occurrence).
2. Compress those counts into short vectors (PPMI + SVD). Words used in similar
   contexts end up with similar vectors: "you shall know a word by the company it keeps".
3. A sentence vector is the average of its word vectors. Search = highest cosine.
"""
import numpy as np

DIM = 10      # length of each word vector
WINDOW = 3   # how many neighbours on each side count as "context"

CORPUS = """
the dog barked at the cat in the park
the puppy barked at the kitten in the garden
my dog likes to play and run in the park
my puppy likes to play and run in the garden
the vet checked the sick dog and gave medicine
the vet checked the sick puppy and gave medicine
the cat drank milk and slept on the sofa
the kitten drank milk and slept on the sofa
the car needs fuel before a long drive on the road
the vehicle needs petrol before a long drive on the highway
the mechanic fixed the engine of the old car
the mechanic fixed the engine of the old vehicle
we drive the car to work on the busy road
we drive the vehicle to work on the busy highway
the chef cooked pasta in the hot kitchen
the cook cooked noodles in the hot kitchen
the chef baked fresh bread in the oven
the cook baked fresh bread in the oven
we ate pasta with tomato sauce for dinner
we ate noodles with tomato sauce for dinner
""".strip().split("\n")

STOPWORDS = set("the a an in on at of to and my we for with before".split())

# Search over NEW sentences, not the training text. Each query uses different
# words from its best match, so keyword search has nothing to grab.
DOCS = [
    "the dog needs medicine from the vet",
    "the kitten slept on the sofa",
    "the car engine needs petrol",
    "the chef bakes bread in the oven",
]
QUERIES = ["sick puppy", "cat drank milk", "vehicle fuel", "cook noodles"]


def build_vocab(sentences):
    words = sorted({w for s in sentences for w in s.split()})
    return words, {w: i for i, w in enumerate(words)}


def cooccurrence(sentences, index):
    # counts[i, j] = how often word j appears within WINDOW words of word i.
    counts = np.zeros((len(index), len(index)))
    for s in sentences:
        ws = s.split()
        for i, w in enumerate(ws):
            for j in range(max(0, i - WINDOW), min(len(ws), i + WINDOW + 1)):
                if i != j:
                    counts[index[w], index[ws[j]]] += 1
    return counts


def ppmi(counts):
    # Raw counts are dominated by "the". PPMI asks: do these two words meet
    # MORE often than chance? It keeps only that surprise, clipped at zero.
    total = counts.sum()
    row = counts.sum(axis=1, keepdims=True)
    col = counts.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore"):
        pmi = np.log(counts * total / (row * col))
    return np.maximum(pmi, 0)


def embed_words(sentences):
    words, index = build_vocab(sentences)
    m = ppmi(cooccurrence(sentences, index))
    # SVD finds the few directions that explain most of the matrix.
    # Keeping only DIM of them forces similar words to share a vector.
    u, s, _ = np.linalg.svd(m)
    vecs = u[:, :DIM] * s[:DIM]
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9  # unit length: dot = cosine
    return words, index, vecs


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def embed_sentence(text, index, vecs):
    # Average of content-word vectors. Crude, but it already carries meaning.
    ws = [w for w in text.split() if w in index and w not in STOPWORDS]
    return np.mean([vecs[index[w]] for w in ws], axis=0)


def keyword_score(query, doc):
    # The old way: count shared words. No idea that puppy ~ dog.
    return len(set(query.split()) & set(doc.split()) - STOPWORDS)


def main():
    words, index, vecs = embed_words(CORPUS)
    print(f"Vocabulary: {len(words)} words, each now a vector of {DIM} numbers\n")
    print("dog   =", np.round(vecs[index["dog"]], 2), "\npuppy =", np.round(vecs[index["puppy"]], 2), "\n")

    print("Nearest words by cosine similarity (1.0 = same direction):")
    for w in ["dog", "car", "pasta", "chef", "road"]:
        sims = vecs @ vecs[index[w]]
        best = [i for i in np.argsort(-sims) if words[i] != w and words[i] not in STOPWORDS][:3]
        print(f"  {w:6} -> " + ", ".join(f"{words[i]} ({sims[i]:.2f})" for i in best))

    print("\nSimilarity between word pairs:")
    pairs = [("dog", "puppy"), ("car", "vehicle"), ("dog", "car"), ("puppy", "oven")]
    for a, b in pairs:
        print(f"  {a:6} vs {b:8} {cosine(vecs[index[a]], vecs[index[b]]):+.2f}")

    doc_vecs = [embed_sentence(d, index, vecs) for d in DOCS]
    print("\nDocuments:")
    for i, d in enumerate(DOCS):
        print(f"  D{i}: {d}")
    print("\nCosine of each query against each document (* = top hit):")
    print(f"  {'query':16}" + "".join(f"{'D' + str(i):>7}" for i in range(len(DOCS))) + "   shared words")
    for q in QUERIES:
        qv = embed_sentence(q, index, vecs)
        emb = [cosine(qv, dv) for dv in doc_vecs]
        best = int(np.argmax(emb))
        cells = "".join(f"{e:6.2f}{'*' if i == best else ' '}" for i, e in enumerate(emb))
        # Keyword search would need at least one shared word to find anything.
        shared = max(keyword_score(q, d) for d in DOCS)
        print(f"  {q:16}{cells}   {shared}")

    # Self-check: if the embedding logic breaks, these fail.
    sim = lambda a, b: cosine(vecs[index[a]], vecs[index[b]])
    assert sim("dog", "puppy") > sim("dog", "car"), "synonyms should be closer than unrelated words"
    assert sim("car", "vehicle") > sim("car", "pasta")
    for q, d in zip(QUERIES, doc_vecs):  # query i should find doc i
        assert cosine(embed_sentence(q, index, vecs), d) == max(
            cosine(embed_sentence(q, index, vecs), x) for x in doc_vecs), q
    print("\nSelf-check passed.")


if __name__ == "__main__":
    main()
