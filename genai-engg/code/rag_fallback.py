"""
RAG fallback ladder — companion code for Chapter 10 (Q21).

Shows the two kinds of fallback in one small pipeline:
  * DEPENDENCY fallbacks: timeouts + circuit breakers around vector DB, keyword
    index, reranker, query rewriter, primary LLM and backup LLM.
  * QUALITY fallbacks: an evidence gate, then rewrite -> broaden -> (optional web)
    -> abstain; and on the generation side: LLM -> backup LLM -> extractive answer.

Stubs stand in for real services so the file runs anywhere:  python rag_fallback.py
"""
import asyncio, re, time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Doc:
    id: str
    text: str
    score: float            # normalized 0-1 relevance (calibrate per retriever in real systems)


@dataclass
class Answer:
    text: str
    mode: str               # grounded | grounded_backup_llm | extractive | abstain
    sources: list = field(default_factory=list)
    trace: list = field(default_factory=list)


class CircuitBreaker:
    """Closed -> (N failures) -> open -> (after reset_after) -> half-open probe."""
    def __init__(self, max_failures=3, reset_after=30.0):
        self.max_failures, self.reset_after = max_failures, reset_after
        self.failures, self.opened_at = 0, None

    def allow(self) -> bool:
        if self.opened_at is None:
            return True
        return time.monotonic() - self.opened_at >= self.reset_after   # half-open probe

    def record(self, ok: bool):
        if ok:
            self.failures, self.opened_at = 0, None
        else:
            self.failures += 1
            if self.failures >= self.max_failures:
                self.opened_at = time.monotonic()


async def guarded(name, breaker, fn, timeout, trace):
    """Run fn() with a timeout and circuit breaker; return None instead of raising."""
    if not breaker.allow():
        trace.append(f"{name}: skipped (circuit open)")
        return None
    try:
        out = await asyncio.wait_for(fn(), timeout)
        breaker.record(True)
        return out
    except Exception as e:                     # includes timeouts
        breaker.record(False)
        trace.append(f"{name}: failed ({type(e).__name__})")
        return None


def fuse(result_lists, k=60):
    """Reciprocal Rank Fusion; keep each doc's best normalized score for gating."""
    rrf, best, by_id = {}, {}, {}
    for lst in result_lists:
        for rank, d in enumerate(lst):
            rrf[d.id] = rrf.get(d.id, 0) + 1 / (k + rank + 1)
            by_id[d.id] = d
            best[d.id] = max(best.get(d.id, 0), d.score)
    return [Doc(i, by_id[i].text, best[i]) for i in sorted(rrf, key=rrf.get, reverse=True)]


class RAG:
    def __init__(self, vector, keyword, rerank, rewrite, llm, backup_llm,
                 web=None, min_score=0.35, min_docs=2):
        self.vector, self.keyword, self.rerank = vector, keyword, rerank
        self.rewrite, self.llm, self.backup_llm, self.web = rewrite, llm, backup_llm, web
        self.min_score, self.min_docs = min_score, min_docs
        self.br = {n: CircuitBreaker() for n in
                   ("vector", "keyword", "rerank", "rewrite", "llm", "backup_llm", "web")}

    async def _retrieve(self, q, broaden, trace):
        v, k = await asyncio.gather(
            guarded("vector", self.br["vector"], lambda: self.vector(q, broaden), 1.0, trace),
            guarded("keyword", self.br["keyword"], lambda: self.keyword(q, broaden), 0.5, trace))
        docs = fuse([v or [], k or []])
        if docs:
            r = await guarded("rerank", self.br["rerank"], lambda: self.rerank(q, docs), 1.0, trace)
            docs = r if r is not None else docs          # reranker is optional: degrade, don't die
        return docs

    def _good(self, docs):
        return [d for d in docs if d.score >= self.min_score]

    async def answer(self, q: str, budget_s: float = 8.0) -> Answer:
        trace, best = [], []

        async def run() -> Answer:
            nonlocal best
            # ---- quality ladder -------------------------------------------------
            good = self._good(await self._retrieve(q, False, trace)); best = good or best
            if len(good) < self.min_docs:
                trace.append("gate: weak evidence -> rewrite query")
                q2 = await guarded("rewrite", self.br["rewrite"], lambda: self.rewrite(q), 2.0, trace)
                if q2 and q2 != q:
                    good = self._good(await self._retrieve(q2, False, trace)); best = good or best
            if len(good) < self.min_docs:
                trace.append("gate: still weak -> broaden (drop filters)")
                good = self._good(await self._retrieve(q, True, trace)); best = good or best
            if len(good) < self.min_docs and self.web:
                trace.append("gate: still weak -> web fallback (UNTRUSTED content)")
                w = await guarded("web", self.br["web"], lambda: self.web(q), 3.0, trace)
                good = self._good(w or []) ; best = good or best
            if len(good) < self.min_docs:
                trace.append("gate: no sufficient evidence -> abstain")
                return Answer("I couldn't find this in the available documents. "
                              "I've logged the question so the content team can review it.",
                              "abstain", [], trace)
            # ---- generation ladder ---------------------------------------------
            ctx = good[:5]
            text = await guarded("llm", self.br["llm"], lambda: self.llm(q, ctx), 6.0, trace)
            mode = "grounded"
            if text is None:
                text = await guarded("backup_llm", self.br["backup_llm"],
                                     lambda: self.backup_llm(q, ctx), 6.0, trace)
                mode = "grounded_backup_llm"
            if text is None:
                text = "I couldn't generate a summary right now. Relevant passages:\n" + \
                       "\n".join(f"- {d.text}" for d in ctx)
                mode = "extractive"
            return Answer(text, mode, [d.id for d in ctx], trace)

        try:
            return await asyncio.wait_for(run(), budget_s)
        except asyncio.TimeoutError:                       # overall latency budget blown
            trace.append("budget exceeded")
            if best:
                return Answer("Taking longer than expected. Best matching passages:\n" +
                              "\n".join(f"- {d.text}" for d in best[:3]),
                              "extractive", [d.id for d in best[:3]], trace)
            return Answer("This is taking too long; please retry.", "abstain", [], trace)


# ------------------------------------------------------------------ demo stubs
CORPUS = {
    "d1": "Refund policy: customers can request a refund within 30 days of purchase.",
    "d2": "Refunds are issued to the original payment method within 5 business days.",
    "d3": "Shipping takes 3 to 7 business days for domestic orders.",
}
STOP = {"the", "is", "what", "a", "of", "my", "how", "do", "i", "get", "who", "can",
        "to", "in", "and", "are", "be", "on", "it"}
tok = lambda s: {w.rstrip("s") for w in re.findall(r"[a-z]+", s.lower())} - STOP


def make_rag(vector_down=False, rerank_slow=False, llm_down=False, backup_down=False):
    async def search(q, broaden):
        qt = tok(q)
        out = [Doc(i, t, len(qt & tok(t)) / max(1, len(qt))) for i, t in CORPUS.items()]
        return sorted([d for d in out if d.score > 0], key=lambda d: -d.score)

    async def vector(q, b):
        if vector_down: raise ConnectionError("vector db down")
        return await search(q, b)

    async def rerank(q, docs):
        if rerank_slow: await asyncio.sleep(3)
        return docs

    async def rewrite(q):
        return "refund policy" if "money back" in q else q

    async def llm(q, ctx):
        if llm_down: raise RuntimeError("primary LLM 503")
        return "Answer based on: " + "; ".join(d.id for d in ctx)

    async def backup(q, ctx):
        if backup_down: raise RuntimeError("backup LLM 503")
        return "[backup model] Answer based on: " + "; ".join(d.id for d in ctx)

    return RAG(vector, search, rerank, rewrite, llm, backup)


async def demo():
    cases = [
        ("normal",              dict(),                                   "what is the refund policy",        "grounded"),
        ("vector DB down",      dict(vector_down=True),                   "what is the refund policy",        "grounded"),
        ("reranker too slow",   dict(rerank_slow=True),                   "what is the refund policy",        "grounded"),
        ("weak query->rewrite", dict(),                                   "how do I get my money back",       "grounded"),
        ("not in corpus",       dict(),                                   "who won the world cup",            "abstain"),
        ("primary LLM down",    dict(llm_down=True),                      "what is the refund policy",        "grounded_backup_llm"),
        ("both LLMs down",      dict(llm_down=True, backup_down=True),    "what is the refund policy",        "extractive"),
    ]
    for name, flags, q, expected in cases:
        a = await make_rag(**flags).answer(q)
        assert a.mode == expected, (name, a.mode, a.trace)
        print(f"{name:22s} -> {a.mode:20s} {a.trace}")
    print("all fallback scenarios behaved as expected")


if __name__ == "__main__":
    asyncio.run(demo())
