# Chapter 8 · Numerical Stability & Cost Engineering

---

### Q1. How does this Sigmoid implementation achieve numerical stability?

```python
import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0,
                    1 / (1 + np.exp(-z)),
                    np.exp(z) / (1 + np.exp(z)))
```

**The problem — floating-point overflow.** The standard sigmoid is $\sigma(z) = \frac{1}{1 + e^{-z}}$.

If you pass a large negative number (e.g., $z = -1000$), the math becomes $e^{-(-1000)} = e^{1000}$ — exceeding the max representable float value:

| Format | Max exponent input to `exp()` | Max representable value | Typical usage |
|---|---|---|---|
| **float16** | ~11.1 | ~65,504 | Mixed-precision training |
| **float32** | ~88.7 | ~3.4 × 10³⁸ | GPU training, most deep learning |
| **float64** | ~709.8 | ~1.8 × 10³⁰⁸ | CPU / NumPy default |

With **float32** (PyTorch/TensorFlow default), any input $z < -88.7$ causes `np.exp(-z)` → `inf`. The calculation becomes $\frac{1}{1 + \inf}$ → **`NaN`**, which silently propagates through every subsequent layer and crashes training.

**float16 is even more dangerous** — it overflows at $z < -11.1$, a completely normal activation value in a deep network.

**The solution — always exponentiate negative values.** This code guarantees the value passed into `np.exp()` is always **negative or zero**. Since $e^x$ for $x \le 0$ is bounded in $[0, 1]$, it safely underflows to `0` rather than exploding to infinity, by splitting into two mathematically equivalent cases:

| Input | Formula used | Exponent sign | Safe? |
|---|---|---|---|
| $z \ge 0$ | $\dfrac{1}{1 + e^{-z}}$ | $-z \le 0$ → always negative | ✅ |
| $z < 0$ | $\dfrac{e^z}{1 + e^z}$ | $z < 0$ → already negative | ✅ |

`np.where` dynamically routes positive and negative inputs to the correct formula, completely avoiding any large positive exponent.

**Key insight:** both formulas are algebraically identical — verify by multiplying numerator and denominator of the standard formula by $e^z$:

$$\frac{1}{1 + e^{-z}} \times \frac{e^z}{e^z} = \frac{e^z}{e^z + 1}$$

> **Quick mental rule for interviews:** float16 overflows past ~11, float32 past ~89, float64 past ~710. In mixed-precision training (float16 weights, float32 gradients), sigmoid on raw logits without this trick will overflow constantly.

---

### Q2. "Your average prompt is 3,200 tokens. Engineers keep adding to the system prompt. At 10M requests/month, every 100 extra tokens costs $15K/month. How do you stop prompt bloat?"

**The common (wrong) answer.** Most engineers say: *"Review the system prompt monthly and trim manually."* That's not a process — it's a hope. It doesn't scale and misses the highest-ROI lever entirely.

**The missing piece — prompt caching.** Not all tokens cost the same. Major providers cache static prompt prefixes — subsequent requests hitting the same prefix pay a fraction of normal input cost:

| Provider | Cache read cost | Saving |
|---|---|---|
| Anthropic (Claude) | 0.1× input | ~90% |
| OpenAI (GPT-4o) | 0.5× input | ~50% |
| Google (Gemini) | 0.25× input | ~75% |

Cache only hits if the prefix is **byte-for-byte identical**. The most common mistake is injecting dynamic content (user IDs, timestamps) early in the prompt — this causes a cache miss on every request. Always structure prompts as:

```
[Static system prompt]      ← cached
[Static few-shot examples]  ← cached
[Retrieved context]         ← not cached
[Conversation history]      ← not cached
[User message]              ← not cached
```

**The full playbook, in priority order:**

| Lever | Approach | Saving |
|---|---|---|
| **Prompt caching** | Mark static prefix with `cache_control`; never inject dynamic content early | 50–90% on static tokens |
| **Dynamic assembly** | Route by query type — simple FAQ gets short prompt + 1 chunk, complex query gets full prompt + 5 chunks | 40–60% on context |
| **History compression** | Summarize every 5 turns instead of sending full transcript (rolling summary + last 2 turns verbatim) | ~70% on history |
| **CI token budgets** | Hard limits per component (system ≤800, context ≤2,000, history ≤500) — fail the build if exceeded | Prevents regression |
| **Inference compression** | LLMLingua / LLMZip at inference time on variable context | 30–50% on remainder |

> **Version control prompts like code.** Every change needs an author, a PR, and a measurable justification — hallucination rate drop from Y% to Z%, not *"feels better."*

> For the deeper mechanics of prefix caching (prefill vs. decode cost, vLLM APC, RadixAttention), see [`genai-engg` Chapter 1](../../genai-engg/chapters/01-inference-caching-and-cost.md).

---

[← Previous: Training, Inference & Evaluation](07-training-inference-eval.md) · [Back to index](../README.md)
