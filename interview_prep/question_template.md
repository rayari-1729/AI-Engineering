# New Question Submission Template

Use this file every time you find a new interview question to add to your prep doc.

---

## How to Use

1. **Copy the block below**
2. **Fill in your details** (rough notes are fine — I'll clean it up)
3. **Paste it to me in chat**
4. I'll return:
   - A fully formatted Q&A block
   - A recommendation: which existing section it belongs to, OR a suggestion for a new section

---

## Submission Block (copy this each time)

```
SOURCE: [where you found it — LinkedIn / Twitter / Interview / Blog post / etc.]

QUESTION:
[Paste the full question here]

YOUR ROUGH ANSWER / NOTES:
[Paste whatever you have — bullet points, half-sentences, code snippets, links — anything]
[If you have nothing, just write "no idea yet" and I'll write the answer]

DIFFICULTY:
[Easy / Medium / Hard]

ANY CODE?
[Paste code here if relevant, or write "none"]
```

---

## Example (filled in)

```
SOURCE: LinkedIn post by @someMLengineer

QUESTION:
What is the difference between Layer Normalization and Batch Normalization,
and why do transformers prefer Layer Norm?

YOUR ROUGH ANSWER / NOTES:
- batch norm normalizes across the batch dimension
- layer norm normalizes across the feature dimension
- transformers use layer norm because batch sizes can be small or variable
- something about NLP sequences having different lengths

DIFFICULTY:
Medium

ANY CODE?
none
```

---

## What I'll Give Back

For each submission, I'll return:

**1. Formatted Q&A Block** — ready to paste into the main doc, like this:

---

**Q. What is the difference between Layer Normalization and Batch Normalization?**

- **Answer:**
  Both normalize activations to stabilize training, but they differ in *what* they normalize across:

  | | Batch Normalization | Layer Normalization |
  |-|--------------------|--------------------|
  | Normalizes over | Batch dimension (across samples) | Feature dimension (within one sample) |
  | Depends on batch size | Yes | No |
  | Works well for | CNNs, computer vision | Transformers, RNNs, small batches |

  **Why Transformers prefer Layer Norm:**
  - NLP sequences have variable lengths, making batch statistics unstable.
  - Transformers often use small or single-sample batches during inference.
  - Layer Norm normalizes within each individual sample independently, making it robust to batch size.

---

**2. Placement Recommendation**, like:

> ✅ **Add to:** `Large Language Models` section (after Q9 on multi-head attention)
> 
> *Reason: Directly related to transformer internals, fits naturally after the attention discussion.*

Or:

> 🆕 **Suggest new section:** `Architecture Deep Dives`
> 
> *Reason: This question is more about specific implementation details that don't fit cleanly into existing sections.*

---

## Current Sections in Main Doc (for reference)

| # | Section |
|---|---------|
| 1 | Generative Models |
| 2 | Large Language Models |
| 3 | Multimodal Models |
| 4 | Embeddings |
| 5 | Training, Inference & Evaluation |
| 6 | Numerical Stability & Optimization |
| 7 | Retrieval-Augmented Generation (RAG) |

---
