"""Query-conditioned skill retrieval (A3).

Information-bottleneck view: given full skill S and query X, produce a compressed
skill Z = f(S, X) that minimizes I(Z; S) (fewer tokens) while preserving
I(Z; Y | X) (answer-relevant rules). Here f is embedding similarity retrieval
over the skill's level-2 sections; top_k / threshold trade off the bottleneck.

Dependency-light: numpy + an ``embed_fn`` callable (list[str] -> list[vector]).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np


def split_sections(skill_md: str) -> list[tuple[str, str]]:
    """Split a skill markdown into (heading, body) pairs on level-2 '## ' headings.

    Text before the first '## ' (top-level title / intro) is returned as an
    always-on ``__preamble__`` section so the retrieved skill stays well-formed.
    """
    parts = re.split(r"\n(?=## )", skill_md.strip())
    sections: list[tuple[str, str]] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        m = re.match(r"##\s+(.+)", p)
        name = m.group(1).strip() if m else "__preamble__"
        sections.append((name, p))
    return sections


@dataclass
class RetrievedSkill:
    text: str
    kept: list[str]
    scores: list[float]


class SkillRetriever:
    """Embedding-based section retriever for a single skill document."""

    def __init__(self, skill_md: str, embed_fn):
        self.embed_fn = embed_fn
        self.sections = split_sections(skill_md)
        self.always_on = [s for s in self.sections if s[0] == "__preamble__"]
        self.scored = [s for s in self.sections if s[0] != "__preamble__"]
        bodies = [b for _, b in self.scored]
        if bodies:
            emb = np.asarray(embed_fn(bodies), dtype=np.float32)
            self._sec_emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
        else:
            self._sec_emb = np.zeros((0, 1), dtype=np.float32)

    def retrieve(self, query: str, *, top_k: int = 2, threshold: float | None = None) -> RetrievedSkill:
        """Return a compressed skill keeping the most query-relevant sections.

        top_k: hard cap on scored sections kept.
        threshold: optionally also drop sections below this cosine sim.
        """
        if not self.scored:
            text = "\n\n".join(b for _, b in self.always_on)
            return RetrievedSkill(text=text, kept=[], scores=[])

        q = np.asarray(self.embed_fn([query]), dtype=np.float32)[0]
        q = q / (np.linalg.norm(q) + 1e-9)
        return self.retrieve_with_vec(q, top_k=top_k, threshold=threshold)

    def retrieve_with_vec(self, q_vec, *, top_k: int = 2, threshold: float | None = None) -> RetrievedSkill:
        """Like :meth:`retrieve` but takes a precomputed (normalized) query vector.

        Lets callers batch-embed all queries once to avoid per-item API calls.
        """
        if not self.scored:
            text = "\n\n".join(b for _, b in self.always_on)
            return RetrievedSkill(text=text, kept=[], scores=[])

        q = np.asarray(q_vec, dtype=np.float32)
        sims = self._sec_emb @ q
        order = np.argsort(-sims)

        chosen: list[int] = []
        for idx in order[:top_k]:
            if threshold is not None and sims[idx] < threshold:
                continue
            chosen.append(int(idx))
        if not chosen:  # always keep at least the single best
            chosen = [int(order[0])]

        kept_names = [self.scored[i][0] for i in chosen]
        kept_scores = [float(sims[i]) for i in chosen]
        chosen_sorted = sorted(chosen)  # preserve original document order
        body_parts = [b for _, b in self.always_on] + [self.scored[i][1] for i in chosen_sorted]
        text = "\n\n".join(body_parts)
        return RetrievedSkill(text=text, kept=kept_names, scores=kept_scores)


# ── Hybrid retrieval ─────────────────────────────────────────────────────────

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


class _BM25:
    """Minimal self-contained BM25 over a small set of documents (skill sections)."""

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = docs
        self.N = len(docs)
        self.doc_len = [len(d) for d in docs]
        self.avgdl = (sum(self.doc_len) / self.N) if self.N else 0.0
        self.df: dict[str, int] = {}
        self.tf: list[dict[str, int]] = []
        for d in docs:
            seen: dict[str, int] = {}
            for w in d:
                seen[w] = seen.get(w, 0) + 1
            self.tf.append(seen)
            for w in seen:
                self.df[w] = self.df.get(w, 0) + 1
        import math
        self.idf = {
            w: math.log(1 + (self.N - n + 0.5) / (n + 0.5))
            for w, n in self.df.items()
        }

    def scores(self, query: str) -> "np.ndarray":
        q = _tokenize(query)
        out = np.zeros(self.N, dtype=np.float32)
        for i in range(self.N):
            tf, dl = self.tf[i], self.doc_len[i]
            s = 0.0
            for w in q:
                if w not in tf:
                    continue
                idf = self.idf.get(w, 0.0)
                num = tf[w] * (self.k1 + 1)
                den = tf[w] + self.k1 * (1 - self.b + self.b * dl / (self.avgdl + 1e-9))
                s += idf * num / (den + 1e-9)
            out[i] = s
        return out


def _minmax(x: "np.ndarray") -> "np.ndarray":
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


class HybridSkillRetriever:
    """Hybrid section retriever: global heat prior + BM25 (lexical) + embedding (semantic).

    Three signals address three rule types:
    - ``always_on_names``: high-heat global rules kept on every query (protect the
      I(Z;Y|X) floor; e.g. answer normalization).
    - BM25: literal/format-triggered rules (snippet formats, quoted-title traps).
    - embedding: semantically related but differently-worded rules.

    Final score per scored section:
        w_heat * heat_prior + w_bm25 * norm(BM25) + w_emb * norm(cos)
    then keep ``top_k`` scored sections plus the always-on ones.
    """

    def __init__(
        self,
        skill_md: str,
        embed_fn,
        *,
        always_on_names: list[str] | None = None,
        heat_prior: dict[str, float] | None = None,
        w_heat: float = 0.0,
        w_bm25: float = 1.0,
        w_emb: float = 1.0,
    ):
        self.embed_fn = embed_fn
        self.w_heat, self.w_bm25, self.w_emb = w_heat, w_bm25, w_emb
        self.heat_prior = heat_prior or {}
        always_on_names = set(always_on_names or [])

        sections = split_sections(skill_md)
        self.always_on = [
            s for s in sections
            if s[0] == "__preamble__" or s[0] in always_on_names
        ]
        self.scored = [
            s for s in sections
            if s[0] != "__preamble__" and s[0] not in always_on_names
        ]
        bodies = [b for _, b in self.scored]
        if bodies:
            emb = np.asarray(embed_fn(bodies), dtype=np.float32)
            self._sec_emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
            self._bm25 = _BM25([_tokenize(b) for b in bodies])
            self._heat = np.array(
                [self.heat_prior.get(n, 0.0) for n, _ in self.scored], dtype=np.float32
            )
        else:
            self._sec_emb = np.zeros((0, 1), dtype=np.float32)
            self._bm25 = None
            self._heat = np.zeros(0, dtype=np.float32)

    def retrieve_with_vec(
        self, q_vec, query_text: str, *, top_k: int = 2
    ) -> RetrievedSkill:
        if not self.scored:
            text = "\n\n".join(b for _, b in self.always_on)
            return RetrievedSkill(text=text, kept=[], scores=[])

        q = np.asarray(q_vec, dtype=np.float32)
        cos = self._sec_emb @ q
        bm = self._bm25.scores(query_text) if self._bm25 is not None else np.zeros_like(cos)
        score = (
            self.w_emb * _minmax(cos)
            + self.w_bm25 * _minmax(bm)
            + self.w_heat * self._heat
        )
        order = np.argsort(-score)
        chosen = sorted(int(i) for i in order[:top_k])

        kept_names = [self.scored[i][0] for i in chosen]
        kept_scores = [float(score[i]) for i in chosen]
        always_names = [n for n, _ in self.always_on if n != "__preamble__"]
        body_parts = [b for _, b in self.always_on] + [self.scored[i][1] for i in chosen]
        text = "\n\n".join(body_parts)
        return RetrievedSkill(text=text, kept=always_names + kept_names, scores=kept_scores)

