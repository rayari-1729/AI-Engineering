# Chapter 7 · Training, Inference & Evaluation

---

### Q1. How does overfitting manifest in LLMs trained on massive corpora, and what prevents it?

**Challenge:** overfitting makes a model perform well on training data but poorly on unseen data — especially risky with massive corpora, where the model may memorize rather than generalize.

**Strategies:**
- **Data augmentation** — increases training diversity.
- **Regularization** — dropout, L2 regularization, early stopping discourage memorization.
- **Model simplification** — reducing complexity, though harder to apply meaningfully at LLM scale.
- **Batch normalization** — stabilizes learning and can help curb overfitting.

---

### Q2. How do you adapt learning rates during LLM training for stable, efficient convergence?

- **Learning rate scheduling** — gradually reduce the rate during training (step decay, exponential decay, cosine annealing) for stable convergence.
- **Adaptive algorithms** — Adam, RMSprop automatically adjust the rate based on training dynamics, improving efficiency and stability.

---

### Q3. How do you handle long context lengths efficiently at inference time?

- **Fine-tuning on longer contexts** — training short then fine-tuning long seems natural, but doesn't work well with vanilla sinusoidal positional encoding.
- **FlashAttention** — optimizes attention for GPUs by processing in blocks, reducing memory-transfer overhead and boosting speed.
- **Multi-Query Attention (MQA)** — shares a single K/V projection across heads instead of per-head projections, cutting memory use and speeding inference (see also GQA in [`genai-engg` Ch.4](../../genai-engg/chapters/04-model-serving-quantization-moe.md)).
- **Positional Interpolation (PI)** — rescales position indices to fit within the trained context window.
- **Rotary Positional Encoding (RoPE)** — rotates embeddings based on position, capturing relative position more fluidly (see [Chapter 3, Q10](03-attention-mechanism.md#q10-what-is-rope-rotary-position-embedding)).
- **ALiBi (Attention with Linear Biases)** — adds distance-based biases directly into attention scores, improving extrapolation to longer sequences.
- **Sparse attention** — restricts attention to a subset of tokens, making compute linear rather than quadratic in sequence length.

---

### Q4. What evaluation metrics judge LLM generation quality?

1. **Perplexity** — how well the model predicts a text sample; lower is better.
2. **Human evaluation** — human raters assess relevance, fluency, coherence, overall quality.
3. **BLEU** — n-gram precision vs. reference translations (machine translation).
4. **ROUGE** — precision/recall/F1 of generated vs. reference summaries.
5. **Diversity** — variety/uniqueness of generated outputs (n-gram diversity, semantic similarity).
6. **Truthfulness evaluation** — comparing LLM answers with human answers, benchmarking against datasets like TruthfulQA, or training true/false classifiers on hidden-layer activations.

(Why lexical-overlap metrics like BLEU/ROUGE often aren't enough — see Q9–Q10 below.)

---

### Q5. Hallucination is a known LLM issue — how do you evaluate and mitigate it?

| Technique | Idea |
|---|---|
| **Log probability (Seq-Logprob)** | Length-normalized sequence log-probability as a confidence signal; comparable to reference-based methods for detecting hallucination (Guerreiro et al., 2023). |
| **Sentence similarity** | Evaluate source-contribution to generated output; low source contribution flags hallucination. Uses reference-based/internal/reference-free measures (LASER, LaBSE, XNLI) (David et al., 2022). |
| **SelfCheckGPT** | Zero-resource black-box detection — useful when output probabilities aren't exposed. Variants include BERTScore-based and QA-based consistency checks (Manakul et al., 2023). |
| **GPT-4 prompting** | Chain-of-thought or sentence-by-sentence prompting to detect factual inconsistency, especially in summarization (Tam et al., 2023). |
| **G-Eval** | LLM-as-judge framework using chain-of-thought + form filling, strong for summarization/dialogue hallucination detection (Liu et al., 2023). |

([Source roundup](https://www.rungalileo.io/blog/5-techniques-for-detecting-llm-hallucinations))

---

### Q6. What are Mixture of Experts (MoE) models?

MoE models consist of several specialized sub-models ("experts") plus a gating mechanism that decides which expert(s) to route a given input to. This divides complex problems into simpler sub-tasks, each handled by a specialized expert, letting the model scale total parameters without scaling compute per token proportionally (only a few experts activate per token).

> For MoE vs. dense trade-offs and expert-parallelism deployment details, see [`genai-engg` Chapter 4](../../genai-engg/chapters/04-model-serving-quantization-moe.md).

---

### Q7. Why can over-reliance on perplexity be problematic for evaluating LLMs?

Perplexity primarily measures how well a model predicts the next word — it can overlook coherence, factual accuracy, and nuanced/implicit meaning. It may not reflect performance on tasks requiring deep understanding or creative language use.

---

### Q8. Why can an LLM give different answers to the same question at temperature 0 — and what can you do about it at the application layer?

**The common (wrong) explanation:** most people blame GPU parallelism and floating-point non-associativity — since `(a + b) + c ≠ a + (b + c)` in floating point, different thread orderings produce different sums. This is *partially* true but not the real culprit: run the same matrix multiplication 1,000 times on a GPU and you get bitwise-identical results. Parallelism alone doesn't explain the drift.

**The real reason — batch-variant kernels.** GPU reduction kernels are scheduled differently depending on batch size:

| Batch size | Kernel strategy | Effect |
|---|---|---|
| Large (enough cores busy) | Each summation stays inside a single core — no cross-core combining | ✅ Deterministic |
| Small (cores would idle) | Summation split across cores; partial results combined | ⚠️ Order-dependent → non-deterministic |

This is called **lack of batch invariance** (term from Horace He / Thinking Machines Lab). The kernel is deterministic for any *fixed* batch composition, but since you never control how many other users' requests are batched alongside yours on a hosted API, output drifts between calls.

**The real fix vs. application-layer workarounds.** The proper fix is rewriting reduction kernels to be batch-invariant — but that requires access to kernel code you don't have when calling an API. At the application layer, you only get workarounds:

| Workaround | What it does | What it doesn't fix |
|---|---|---|
| **Response caching** | Returns stored output → always same answer | Skips the model entirely; cache misses still drift |
| **Seeding (where supported)** | Fewer variations for a fixed seed | Effectiveness varies by provider; breaks when model version changes |
| **Structured output / constrained decoding** | Constrains format (JSON schema, regex) | Same format, still different semantic content inside it |
| **Output normalization** | Cleans cosmetic drift (whitespace, casing) | Semantic drift survives — "Queens, New York" vs "New York City" both pass |

> **Key takeaway:** temperature 0 sets the *sampling* strategy to greedy (always pick the highest-probability token), but doesn't eliminate non-determinism introduced at the *hardware/kernel* level before sampling even happens. True determinism requires batch-invariant kernels — a constraint most production APIs don't guarantee.

---

### Q9. What are the core lexical-overlap metrics for evaluating text generation, and how are they computed?

Three standard metrics — all score `[0, 1]`, all computed against a human-written reference:

| Metric | What it measures |
|---|---|
| **BLEU** | n-gram precision of prediction vs. reference |
| **ROUGE-L** | Longest common subsequence F1 vs. reference |
| **METEOR** | Precision/recall with stemming + synonym matching |

In reward-signal setups (e.g. RLHF), total reward = sum of all enabled signals.

**ROUGE-L — LCS-based F1:**

$$ROUGE\text{-}L = \frac{(1 + \beta^2) \cdot P_{lcs} \cdot R_{lcs}}{R_{lcs} + \beta^2 \cdot P_{lcs}}$$

where $P_{lcs}$ = LCS length / prediction length, $R_{lcs}$ = LCS length / reference length.

```
Reference:  "the cat sat on the mat"   (6 tokens)
Prediction: "the cat sat on a mat"     (6 tokens)

LCS = "the cat sat on _ mat" = 5 tokens

P = 5/6 = 0.833
R = 5/6 = 0.833
ROUGE-L (β=1) = (2 × 0.833 × 0.833) / (0.833 + 0.833) = 0.833
```

**METEOR — stemming + synonym aware:**

$$METEOR = F_{mean} \times (1 - \text{Penalty})$$
$$\text{Penalty} = 0.5 \times \left(\frac{\text{chunks}}{\text{matches}}\right)^3$$

$F_{mean}$ = harmonic mean of unigram precision and recall. Penalty increases as matched tokens become more fragmented (fewer contiguous chunks = worse).

```
Reference:  "quick brown fox"
Prediction: "fast brown fox"   ("fast" matches "quick" via synonym)

Matches = 3 (all unigrams match)
P = 3/3 = 1.0,  R = 3/3 = 1.0
F_mean = 1.0

Chunks = 2 ("brown fox" contiguous, "fast" separate)
Penalty = 0.5 × (2/3)³ = 0.5 × 0.296 = 0.148

METEOR = 1.0 × (1 - 0.148) = 0.852
```

**BLEU — n-gram precision with brevity penalty:**

$$BLEU = BP \times \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)$$
$$BP = \begin{cases} 1 & \text{if } c > r \\ e^{1 - r/c} & \text{if } c \leq r \end{cases}$$

$BP$ penalizes predictions shorter than the reference. $p_n$ = modified n-gram precision for order $n$ (clips counts to avoid repeat-word gaming). Uniform weights $w_n = 1/N$.

```
Reference:  "the cat sat on the mat"
Prediction: "the cat sat"              (too short → BP kicks in)

c=3, r=6 → BP = e^(1 - 6/3) = e^(-1) = 0.368

p1 = 3/3 = 1.0   (all unigrams match)
p2 = 2/2 = 1.0   (bigrams: "the cat", "cat sat")

BLEU-2 = 0.368 × exp(0.5×log1.0 + 0.5×log1.0) = 0.368
```

**When to use which:**

| Metric | Strength | Weakness |
|---|---|---|
| BLEU | Fast, standard benchmark | Precision-only, punishes paraphrase |
| ROUGE-L | Captures word order via LCS | No synonym awareness |
| METEOR | Most human-aligned of the three | Slower, needs stemmer/synonym lexicon |

> **Key takeaway:** no single metric is sufficient. In practice, use all three as a composite reward signal — they penalize different failure modes (BLEU: brevity + precision, ROUGE-L: order, METEOR: paraphrase).

---

### Q10. Why are BLEU, ROUGE-L, and METEOR insufficient for chatbots, agentic systems, or RAG pipelines?

All three are **lexical overlap metrics** — they compare surface tokens against a fixed reference. This breaks down the moment there's no single correct answer, or meaning matters more than wording.

```
Reference:  "The meeting is on Friday."
Output A:   "The meeting is on Friday."        → BLEU = 1.0  ✅
Output B:   "Your meeting is scheduled Friday." → BLEU = 0.4  ❌ (penalised)

Both are correct. BLEU cannot know that.
```

**Why each use-case breaks them:**

| Task | Why lexical metrics fail |
|---|---|
| **Chatbot** | No ground-truth reference. Tone, helpfulness, safety — none are token-measurable |
| **Agentic RAG** | Correct answer may be paraphrased from 3 retrieved chunks — low overlap, high quality |
| **Multi-step reasoning** | Final answer correct but intermediate steps wrong — metrics miss reasoning quality |
| **Tool-use / function calling** | Output is structured JSON or an action, not natural language — overlap is meaningless |

**What to use instead:**

| Task | Right metric |
|---|---|
| Chatbot | G-Eval: Helpfulness, Coherence, Safety (LLM-as-judge) |
| RAG retrieval | Context Precision, Context Recall, Answer Relevance (see [Chapter 1](01-rag.md)) |
| Agentic task completion | Task Success Rate, Step Accuracy, Groundedness |
| Open-ended QA | Faithfulness + human eval or LLM-as-judge |

> **Key takeaway:** BLEU/ROUGE/METEOR are valid as fast reward signals during training on well-defined generation tasks (summarization, translation). For anything involving reasoning, retrieval, or open-ended output — switch to semantic evaluation (G-Eval, RAG-specific metrics, or task success rate).

---

[← Previous: Embeddings](06-embeddings.md) · [Back to index](../README.md) · [Next: Numerical Stability & Cost Engineering →](08-numerical-stability-and-cost.md)
