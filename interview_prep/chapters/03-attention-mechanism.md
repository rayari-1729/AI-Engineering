# Chapter 3 · Attention Mechanism

The deepest-dive chapter — this is where interviewers probe whether you actually understand the machinery rather than just naming it. Worth drilling the code snippets until you can write them cold.

---

### Q1. What are Query, Key, Value matrices?

Every token embedding `X` is projected into three vectors using three learned weight matrices:

```
Q = X · Wq
K = X · Wk
V = X · Wv
```

**Analogy (search engine):**
- Query = what you're searching for
- Key = the label/index attached to each item
- Value = the actual content returned if the key matches well

In self-attention, each token's Query is compared against every other token's Key to get a relevance score. That score decides how much of each token's Value flows into the final output.

---

### Q2. What's the formula for self-attention?

```
Attention(Q, K, V) = softmax( Q·Kᵀ / √dₖ ) · V
```

**Steps:**
1. `Q·Kᵀ` → similarity score between every pair of tokens (raw attention logits)
2. Divide by `√dₖ` → scaling (see Q3)
3. `softmax` → convert scores into weights that sum to 1 per row
4. Multiply by `V` → weighted sum of Value vectors = output for each token

```python
import torch, torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V):
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / d_k**0.5
    weights = F.softmax(scores, dim=-1)
    return weights @ V
```

---

### Q3. Why divide by √dₖ?

- Q and K combine via dot product. As dimension `dₖ` grows, the dot product's variance grows roughly proportional to `dₖ` (sum of `dₖ` roughly-independent terms).
- Large logits → softmax becomes extremely peaked (near one-hot) → gradients vanish → unstable/slow training.
- Dividing by `√dₖ` rescales the variance back to ~1, keeping softmax in a healthy gradient range.

**Example:** if `dₖ = 64`, unscaled dot products can be ~8× larger than scaled ones (`√64 = 8`) — enough to push softmax into saturation.

---

### Q4. What is softmax, and where is it used?

Softmax converts a vector of raw scores (logits) into a probability distribution — values between 0 and 1, summing to 1.

```
softmax(zᵢ) = e^zᵢ / Σⱼ e^zⱼ
```

```python
import numpy as np

def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)  # numerical stability
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)
```

**Used in:**
- Attention → turns scaled `QKᵀ` scores into attention weights.
- Final classification layer → turns output logits into class probabilities.

**Key property:** differentiable everywhere (unlike hard argmax), and its exponential growth amplifies the gap between high and low scores.

---

### Q5. What is multi-head attention? Why do we need it?

Instead of computing attention once with the full embedding dimension, split into `h` smaller heads, each with its own Q/K/V projections, run attention in parallel, then concatenate and project back.

```
MultiHead(Q,K,V) = Concat(head₁, ..., head_h) · Wo
headᵢ = Attention(Q·Wqᵢ, K·Wkᵢ, V·Wvᵢ)
```

**Why:** a single head is forced to average over *all* kinds of relationships (syntax, position, long-range dependency, coreference…) into one score. Multiple heads let the model learn different relationship types in parallel subspaces — e.g. one head tracks subject-verb agreement, another tracks nearby words. More expressive, without much extra compute since heads run in parallel with smaller dims each.

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.h, self.d_k = num_heads, d_model // num_heads
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, _ = x.shape
        split = lambda t: t.view(B, T, self.h, self.d_k).transpose(1, 2)
        Q, K, V = split(self.Wq(x)), split(self.Wk(x)), split(self.Wv(x))
        scores = Q @ K.transpose(-2, -1) / self.d_k**0.5
        out = torch.softmax(scores, dim=-1) @ V
        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.Wo(out)
```

---

### Q6. How would you build a Transformer block from scratch?

Standard encoder block = `Multi-Head Self-Attention → Add & Norm → Feed-Forward → Add & Norm` (residual connections around both sub-layers).

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))   # pre-norm
        x = x + self.ff(self.norm2(x))
        return x
```

**Full model recipe:** token embeddings + positional encoding → stack of N such blocks → final LayerNorm → output head (linear + softmax for LM).

For a **decoder**: add a causal mask (token *i* can't attend to tokens > *i*), and optionally a cross-attention sub-layer attending to encoder outputs.

---

### Q7. What is PagedAttention?

A memory-management technique (from vLLM) for the KV-cache during LLM inference — not a new attention formula, just a smarter way to *store* K/V.

**Problem it solves:** normally each sequence's KV-cache is one contiguous memory block sized for the max possible length → heavy memory waste/fragmentation, since the actual length isn't known ahead of time.

**Idea:** borrowed from OS virtual memory paging — split the KV-cache into small fixed-size blocks ("pages") that don't need to be contiguous; a block table maps logical token positions to physical memory blocks.

**Benefit:** near-zero memory waste, and blocks can be shared across requests (e.g. same prompt prefix in parallel sampling/beam search) → much higher serving throughput.

---

### Q8. What is the attention matrix?

The matrix of attention weights produced by `softmax(QKᵀ/√dₖ)` — shape `(seq_len, seq_len)`. Row `i` = how much token `i` attends to every other token (weights sum to 1 across the row). Often visualized as a heatmap for interpretability — showing which tokens the model "looks at" when predicting the next one.

---

### Q9. What is cross-attention? How do image + text attention work together?

**Cross-attention** uses the same formula as self-attention, but the Query comes from one sequence while Key & Value come from a *different* sequence/modality:

```
CrossAttn(Q_text, K_img, V_img) = softmax(Q_text · K_imgᵀ / √dₖ) · V_img
```

Classic use: a Transformer decoder attending over encoder outputs (translation); or text attending over image features (multimodal).

**Two common image+text patterns:**
1. **Dual-encoder / contrastive (e.g. CLIP)** — image and text are encoded *separately* by their own self-attention transformers (image split into patches → patch embeddings, like ViT). No cross-attention during encoding — the two embeddings are only compared at the end (cosine similarity + contrastive loss).
2. **Fused / cross-attention (e.g. BLIP, Flamingo-style VLMs)** — image patches become "visual tokens." Text tokens self-attend among themselves as usual, and additionally cross-attend to the visual tokens (Query = text, Key/Value = image) so each word can pull in relevant visual context.

Both approaches reuse the exact same attention formula from Q2 — only difference is *where Q, K, V come from*.

---

### Q10. What is RoPE (Rotary Position Embedding)?

Attention itself is **permutation-invariant** — `softmax(QKᵀ)V` has no idea about token *order* ("dog bites man" = "man bites dog" to raw attention). So position information must be injected. RoPE is the standard modern way (LLaMA, Qwen, Mistral, etc.), replacing the older sinusoidal/learned *additive* position embeddings.

**Core idea:** instead of *adding* a position vector, RoPE **rotates** the Q and K vectors by an angle proportional to their position. Each 2D pair of dimensions is rotated by angle `m·θᵢ`, where `m` = token position.

```
θᵢ = 10000^(-2i/d)          (frequency for dimension pair i)
```

For a vector `x` at position `m`, each pair `(x₂ᵢ, x₂ᵢ₊₁)` is rotated:

```
[x'₂ᵢ  ]   [cos(mθᵢ)  -sin(mθᵢ)] [x₂ᵢ  ]
[x'₂ᵢ₊₁] = [sin(mθᵢ)   cos(mθᵢ)] [x₂ᵢ₊₁]
```

**Why this is clever (relative position for free):** when you take the dot product of a rotated Query at position `m` with a rotated Key at position `n`, the rotations combine so the result depends only on the **relative distance `(m − n)`**, not on absolute `m` and `n` separately. The model naturally learns "how far apart" tokens are — which generalizes better to longer sequences.

**Why applied *before* attention (to Q and K, not V):** position must influence the *similarity score* `QKᵀ`. So RoPE rotates Q and K right before the dot product. V is left untouched — it carries content, not position. (This is also why RoPE is applied inside every layer, per-head, rather than once at the input like additive embeddings.)

```python
def apply_rope(x, pos, theta=10000.0):
    # x: (..., seq_len, d), even d
    d = x.shape[-1]
    i = torch.arange(0, d, 2)
    freqs = pos[:, None] * (theta ** (-i / d))       # (seq_len, d/2)
    cos, sin = freqs.cos(), freqs.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1*cos - x2*sin, x1*sin + x2*cos], -1).flatten(-2)
```

**Bonus — long-context extension:** RoPE's frequencies can be scaled/interpolated (NTK-aware scaling, YaRN) to extend a model to longer context than it was trained on, without full retraining.

---

### Q11. How does RoPE work in VLMs / multimodal LLMs?

Plain RoPE assumes a **1D sequence** (text is left-to-right). But images/videos are **2D/3D** (height, width, and time for video). Flattening image patches into a 1D line loses spatial structure, so multimodal models extend RoPE to multiple axes.

**M-RoPE (Multimodal RoPE — e.g. Qwen2-VL):** split the position index into components — **(temporal, height, width)** — and allocate different slices of the rotary dimensions to each. A video patch gets a position like `(t, h, w)` instead of a single number `m`.

- **Text tokens** — all three components share the same value → behaves like normal 1D RoPE.
- **Image tokens** — temporal fixed, but height & width vary → encodes 2D spatial layout so the model knows which patch is above/below/left/right of another.
- **Video tokens** — temporal also increments across frames → encodes time.

**Why it matters:** lets the model reason about spatial relationships ("the object on the top-left") and keeps text and vision positions in one **unified coordinate system**, so a single attention mechanism handles both modalities consistently. It also helps position IDs stay compact, aiding long video/multi-image context.

---

### Q12. How do you tokenize an image for a Transformer? (ViT / patch embeddings)

A Transformer only eats a **sequence of vectors (tokens)**. Text is already a token sequence; the trick is turning an image into one too.

1. **Split into patches** — cut the image (e.g. 224×224) into fixed-size patches (e.g. 16×16) → 196 patches. Each patch is the "word."
2. **Flatten + linear projection** — flatten each patch (16×16×3 = 768 values) and pass through a linear layer → a patch embedding vector. This is the image equivalent of a token embedding.
3. **Add positional embeddings** — since attention is order-agnostic, add position info so the model knows where each patch sits.
4. **[CLS] token** — prepend a learnable classification token whose final state is used for image-level prediction.

```python
# Conv2d with kernel=stride=patch_size does "split + flatten + project" in one op
patch_embed = nn.Conv2d(3, d_model, kernel_size=16, stride=16)
tokens = patch_embed(image).flatten(2).transpose(1, 2)  # (B, num_patches, d_model)
```

**Combining image + text:** once the image is a sequence of patch tokens and text is a sequence of word tokens, you can:
- **Concatenate** both sequences and feed to one shared Transformer (early fusion, e.g. LLaVA-style — image tokens projected into the LLM's embedding space, then prepended to text tokens).
- **Cross-attention** — let text tokens (Query) attend to image tokens (Key/Value), e.g. BLIP/Flamingo.
- **Dual-encoder** — encode each modality separately and align embeddings via contrastive loss, e.g. CLIP.

**Key point:** the Transformer machinery doesn't change — you just convert every modality into tokens in a shared embedding space, and the same attention handles the rest.

---

[← Previous: Transformers & LLM Architecture](02-transformers-and-llms.md) · [Back to index](../README.md) · [Next: Generative Models →](04-generative-models.md)
