# Coding Interview Prep

**Hands-on coding-round practice and company-specific process intel**, as opposed to the conceptual Q&A in [`interview_prep/`](../interview_prep/README.md). Use this folder for the "write code, live, under time pressure" portion of an interview loop — LeetCode-style DSA, ML-from-scratch implementations, and the AI-assisted/practical formats now common at AI-native companies.

---

## What's here

| | Contents |
|---|---|
| [notebooks/](notebooks/) | Three runnable Jupyter notebooks — ML algorithms from scratch, ML-flavored DSA, and AI-assisted/practical coding problems |
| [company_interview_processes.md](company_interview_processes.md) | Per-company interview loops at 17 AI labs and AI-first companies: process, coding problems, ML questions, system design, and 2026 corporate notes |
| [system_design_and_coding_patterns.md](system_design_and_coding_patterns.md) | 25 must-know system design problems, AI-native system design trends, and the highest-ROI DSA coding patterns for 2026 loops |

---

## Notebooks

| Notebook | Covers |
|---|---|
| [MLE_Coding_Interview_Guide.ipynb](notebooks/MLE_Coding_Interview_Guide.ipynb) | From-scratch NumPy/PyTorch implementations: linear & logistic regression, K-Means, KNN, a 2-layer MLP with manual backprop, LayerNorm, multi-head self-attention, a LoRA layer, and a BPE tokenizer — each with a math derivation and a verification/unit-test cell |
| [ML_LeetCode_Interview_Guide.ipynb](notebooks/ML_LeetCode_Interview_Guide.ipynb) | ML-flavored DSA in pure Python: weighted/reservoir sampling, calibration error, ROC-AUC, precision-recall/AP, sparse vector dot product, top-k cosine retrieval, TF-IDF retrieval, streaming median, edit distance/WER, image rotation |
| [AI_Assisted_Coding_Interview_Guide.ipynb](notebooks/AI_Assisted_Coding_Interview_Guide.ipynb) | Stateful simulation and systems-design problems in the "practical coding" / AI-assisted format: grid robots, game engines (Snake, Connect Four, Battleship), an in-memory KV database with TTL, a text editor with undo/redo, a rate limiter, a DAG task executor — each with starter interfaces and a "Key Traps & AI Failure Modes" section |

These are meant to be opened and run, not just read — work through a problem before checking the reference implementation.

---

## How the three pieces of this repo fit together

- **[interview_prep/](../interview_prep/README.md)** — conceptual round 1: "explain RAG evaluation," "derive attention scaling," "what is RLHF." Mostly talk, some whiteboard code.
- **[genai-engg/](../genai-engg/README.md)** — the production/systems-depth layer for when the interview goes deep on how you'd actually build and run this — inference caching, agent sandboxing, protocols, fine-tuning, security.
- **coding_interview/ (this folder)** — the hands-on-keyboard round: solve it, run it, defend your choices.

A typical AI Engineer loop touches all three: conceptual screen → coding round → project/systems deep-dive.

---

## A note on the process docs

`company_interview_processes.md` and `system_design_and_coding_patterns.md` are time-stamped snapshots (compiled through November 2026) of a fast-moving space — company processes, AI-tool policies, and even company ownership change frequently (see the "corporate notes" throughout). Treat specifics as **directional, not current-day fact** — verify against the company's own careers page and recent candidate reports (Blind, Glassdoor, 1point3acres) before an actual interview.
