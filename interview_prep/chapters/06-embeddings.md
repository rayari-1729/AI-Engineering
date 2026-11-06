# Chapter 6 · Embeddings

---

### Q1. What is the fundamental concept of embeddings, and how are they more compact than raw input data?

Embeddings are dense, low-dimensional representations of high-dimensional data, mapping entities (words, sentences, images) to continuous vectors. Unlike raw inputs — which are often sparse and high-dimensional (e.g., one-hot word vectors) — embeddings preserve semantic relationships while drastically reducing dimensionality, letting models learn patterns, similarities, and differences more effectively. In NLP, semantically similar words land closer together in embedding space.

---

### Q2. Compare word embeddings and sentence embeddings.

| | Word Embeddings | Sentence Embeddings |
|---|---|---|
| **Scope** | Individual words | Entire sentences/longer text |
| **Applications** | Synonym detection, POS tagging, NER | Document classification, semantic similarity, sentiment analysis |
| **Characteristics** | Static — one embedding per word, limited for polysemy | Dynamic — consider word interactions and structure |

**Choosing between them:** task requirements (word-level vs. sentence-level), contextual sensitivity (sentence embeddings or contextual word embeddings like BERT handle multiple meanings better), computational cost (sentence embeddings from models like BERT are more resource-intensive), and data availability.

---

### Q3. What are contextual embeddings, and how do models like BERT generate them?

Contextual embeddings are dynamic word representations that change based on the word's context within a sentence. BERT generates them using a deep transformer that processes the entire sentence at once, capturing relationships and dependencies between all words simultaneously.

**Advantage over static embeddings:** excel at tasks needing deep contextual understanding — sentiment analysis (meaning shifts with surrounding words) and disambiguation of homonyms/polysemy — by considering the word's role and relations within the full sentence.

---

### Q4. What challenges arise in generating cross-modal embeddings (e.g., text + image in a shared space)?

Challenges include aligning semantic concepts across modalities with inherently different data characteristics, and ensuring the shared embedding captures the essence of both.

**Strategies:**
- **Joint learning** — train on tasks requiring simultaneous understanding of both modalities, pushing the model toward common semantic ground.
- **Canonical Correlation Analysis (CCA)** — a statistical method aligning embeddings from different modalities by maximizing their correlation.
- **Contrastive learning** — pulls embeddings of similar cross-modal items closer and pushes dissimilar ones apart.

---

### Q5. How do you capture good representations for rare words with limited training occurrences?

- **Subword tokenization** — break rare words into smaller units (morphemes, syllables) with more robustly learnable embeddings.
- **Smoothing techniques** — borrow statistical strength from similar or more frequent words.
- **Contextual augmentation** — artificially increase the representation of rare words in training data.

---

### Q6. What regularization techniques are used for training embeddings?

- **L2 regularization** — penalizes the magnitude of embedding vectors, keeping them small and reducing overfitting to specific examples.
- **Dropout** — randomly zeroes elements of embedding vectors during training, forcing reliance on broader context.
- **Noise injection** — adds random noise during training, improving robustness and generalization.

---

### Q7. How can pre-trained embeddings be leveraged for transfer learning?

- **Initialization** — use pre-trained embeddings to initialize a model's embedding layer before task-specific training, giving it a strong starting point.
- **Feature extraction** — use pre-trained embeddings as fixed input features to further task-specific layers, especially useful with little labeled data.

**Advantages:** efficiency (less data/compute needed), generalization (broad language/visual understanding baked in), and quick adaptation (faster fine-tuning cycles).

---

### Q8. What is quantization in the context of embeddings, and how does it reduce memory footprint?

Quantization converts continuous, high-precision embedding vectors (typically 32-bit floats) into a more compact, lower-precision format — e.g., float16 or int8 — significantly reducing memory footprint and compute cost without substantially compromising representation quality. Particularly valuable for deploying large-scale models on resource-constrained environments (mobile, browser) — faster loading, lower memory usage.

---

### Q9. How do you efficiently train embeddings for high-cardinality categorical features in tabular data?

- **Embedding layers** — convert high-cardinality categorical features into dense, low-dimensional learned embeddings within the network.
- **Batch/mini-batch training** — process subsets of data at a time to handle large datasets and high cardinality efficiently.
- **Regularization** — prevents overfitting, especially important for categories with few occurrences.

---

### Q10. How do you perform efficient nearest-neighbor search over large-scale embeddings?

Approximate Nearest Neighbor (ANN) methods avoid exhaustively comparing every pair:

- **Locality-Sensitive Hashing (LSH)**
- **Tree-based partitioning** (KD-trees, Ball trees)
- **Graph-based approaches** (HNSW)

These build an index from the embeddings that narrows the search space quickly, trading a small amount of accuracy for large speed gains at scale.

---

### Q11. How do you handle out-of-vocabulary (OOV) words during embedding generation?

- **Subword tokenization** — break OOV words into known subwords/characters and aggregate their embeddings.
- **Zero or random initialization** — assign a placeholder vector, optionally fine-tuned if training data allows.
- **Fallback to similar words** — use embeddings of semantically/morphologically similar known words as a proxy.

---

### Q12. What metrics quantitatively evaluate embedding quality?

- **Cosine similarity** — angle between two embedding vectors, for semantic similarity.
- **Precision@k / Recall@k** — for information retrieval, how many of the top-k results are relevant.
- **WEAT (Word Embedding Association Test)** — assesses bias by measuring associations between target-word sets and attribute-word sets.

---

### Q13. What is triplet loss in embedding learning?

Triplet loss ensures an anchor embedding sits closer to a positive embedding (similar content) than to a negative embedding (dissimilar content) by at least a margin. This organizes the embedding space so similar instances cluster together and dissimilar ones are pushed apart, improving the model's ability to discriminate between categories/concepts.

### Q13b. What is the significance of the margin parameter in triplet/contrastive loss?

The margin specifies the minimum required gap between positive-pair and negative-pair distances to the anchor. A larger margin spreads embeddings further apart, potentially improving discrimination — but too large a margin can cause training difficulty or degraded performance from an overly strict separation requirement.

---

### Q14. Why is a unidirectional LSTM a poor choice for sentiment analysis? What would you use instead?

A unidirectional LSTM only reads left-to-right, so each word's representation only has access to *preceding* context. This hurts sentiment analysis because overall sentiment often depends on words that appear later. Example: *"The phone looks great but the battery is terrible"* — when the model reads "great," it hasn't yet seen "battery is terrible," so it may not capture the final sentiment correctly.

A **BiLSTM** fixes this by processing the sentence in both directions, so every word representation incorporates past *and* future context.

For modern applications, prefer a **Transformer-based model** (BERT, DistilBERT): self-attention captures relationships between all words simultaneously, learns richer contextual representations, typically beats LSTMs on accuracy, and trains faster on GPUs (parallel token processing instead of sequential).

---

### Q15. For an LLM-based ABSA (Aspect-Based Sentiment Analysis) pipeline, which embedding model would you use — and why not vanilla BERT?

Use a sentence embedding model such as **BGE** (BAAI General Embedding) or **E5** rather than vanilla BERT embeddings.

**Why not BERT:** BERT was trained for Masked Language Modeling and Next Sentence Prediction — it produces good contextual *token* embeddings, but wasn't specifically optimized for sentence-level semantic similarity. Sentences with similar meaning aren't guaranteed to land close together in its embedding space.

**Why BGE/E5:** trained with **contrastive learning**, explicitly pulling semantically similar sentences together and pushing dissimilar ones apart — producing much higher-quality sentence embeddings for semantic search, clustering, retrieval, and topic discovery.

In an ABSA pipeline, after the LLM extracts aspects/opinions, these embedding models group semantically similar opinions — *"battery drains quickly"*, *"poor battery backup"*, *"battery dies fast"* — into the same cluster, producing cleaner topic discovery and better business insights.

**Typical production pipeline:**

```
Reviews → LLM (ABSA) → Aspect & Opinion Extraction → BGE/E5 Embeddings
        → UMAP → HDBSCAN/KMeans → Topic Labeling → Dashboard
```

---

[← Previous: Multimodal Models](05-multimodal-models.md) · [Back to index](../README.md) · [Next: Training, Inference & Evaluation →](07-training-inference-eval.md)
