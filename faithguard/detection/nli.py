"""NLI-based claim verification against retrieved evidence.

Uses a cross-encoder NLI model (default: cross-encoder/nli-deberta-v3-small)
to score each (claim, evidence_passage) pair into
P(entailment), P(neutral), P(contradiction).

Backend: ONNX Runtime (quantized int8 export from Xenova/nli-deberta-v3-small).
This keeps the runtime torch-free — ~200 MB RAM instead of ~600 MB — which is
what lets the API run on 512 MB free-tier hosts (Render free). The same model
family as training, so classifier features stay calibrated.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger("faithguard.nli")

# torch cross-encoder repo -> ONNX export repo (quantized int8)
_ONNX_REPO_MAP = {
    "cross-encoder/nli-deberta-v3-small": "Xenova/nli-deberta-v3-small",
}
_ONNX_FILES = ("onnx/model_quantized.onnx", "tokenizer.json", "config.json")


@dataclass
class ClaimEvidenceScore:
    claim: str
    hedged: bool
    best_entailment: float          # max P(entail) over passages
    best_contradiction: float       # P(contradiction) at the best-entailing passage
    mean_entailment: float          # mean P(entail) over passages
    best_passage_idx: int           # index of the passage that best supports it
    per_passage_entailment: list[float] = field(default_factory=list)
    best_neutral: float = 0.0       # P(neutral) at the best-entailing passage

    @property
    def supported(self) -> bool:
        return self.best_entailment >= 0.5

    @property
    def contradicted(self) -> bool:
        return self.best_contradiction > self.best_entailment and self.best_contradiction >= 0.4

    @property
    def support(self) -> float:
        """Soft support in [0,1]: full credit for entailment, half credit for
        neutral (consistent but not strictly entailed — paraphrase / minor
        inference). Contradictions get none. NLI cross-encoders are strict:
        a paraphrased-but-correct claim usually lands in `neutral`, so raw
        entailment alone saturates at the floor for real RAG answers."""
        return float(self.best_entailment + 0.5 * self.best_neutral)


class NLIScorer:
    """Batched NLI scoring of claims against evidence passages (ONNX backend)."""

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-small", device: str = "cpu"):
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer
        import onnxruntime as ort

        self.model_name = model_name
        repo = _ONNX_REPO_MAP.get(model_name, model_name)

        paths = {}
        for f in _ONNX_FILES:
            local = os.environ.get("FG_NLI_" + Path(f).name.upper().replace(".", "_"))
            # local_files_only=True: never hit the network at runtime — the
            # model is baked into the image (HF_HOME). A network call here
            # was what OOM'd the 512 MB Render free tier.
            paths[f] = local or hf_hub_download(repo, f, local_files_only=True)

        self.tokenizer = Tokenizer.from_file(paths["tokenizer.json"])
        self.tokenizer.enable_truncation(max_length=512)
        # pair padding is applied per-batch below (dynamic lengths)

        cfg = json.loads(Path(paths["config.json"]).read_text(encoding="utf-8"))
        id2label = {int(k): v.lower() for k, v in (cfg.get("id2label") or {}).items()}
        labels = [id2label.get(i, "") for i in range(3)]
        if "entailment" in labels:
            self._ent = labels.index("entailment")
            self._con = labels.index("contradiction") if "contradiction" in labels else 0
            self._neu = labels.index("neutral") if "neutral" in labels else 2
        else:  # fallback to common ordering
            self._con, self._ent, self._neu = 0, 1, 2

        sess_opts = ort.SessionOptions()
        sess_opts.inter_op_num_threads = 1
        sess_opts.intra_op_num_threads = 1
        # Minimize peak RAM on 512 MB hosts: no pre-allocated arena, no
        # mem-pattern cache, and no graph optimization (the optimizer builds
        # a second copy of the graph in RAM during load — that spike was
        # enough to trip the OOM killer on the free tier).
        sess_opts.enable_mem_pattern = False
        sess_opts.enable_cpu_mem_arena = False
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        self.session = ort.InferenceSession(
            paths["onnx/model_quantized.onnx"],
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        log.info("NLI ONNX backend ready: %s (ent=%d con=%d neu=%d)",
                 repo, self._ent, self._con, self._neu)

    def _predict_logits(self, pairs: list[tuple[str, str]], batch_size: int = 32) -> np.ndarray:
        """Run (premise, hypothesis) pairs through the ONNX model -> logits."""
        all_logits = []
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start:start + batch_size]
            encs = self.tokenizer.encode_batch([(a, b) for a, b in batch])
            max_len = max(len(e.ids) for e in encs)
            input_ids = np.zeros((len(batch), max_len), dtype=np.int64)
            attention = np.zeros((len(batch), max_len), dtype=np.int64)
            pad_id = self.tokenizer.token_to_id("[PAD]") or 0
            input_ids.fill(pad_id)
            for i, e in enumerate(encs):
                n = len(e.ids)
                input_ids[i, :n] = e.ids
                attention[i, :n] = e.attention_mask
            out = self.session.run(
                None, {"input_ids": input_ids, "attention_mask": attention}
            )[0]
            all_logits.append(np.asarray(out, dtype=np.float32))
        return np.concatenate(all_logits, axis=0)

    def score_claims(
        self,
        claims: list[dict],
        passages: list[str],
        batch_size: int = 32,
    ) -> list[ClaimEvidenceScore]:
        """Score every claim against every passage (batched)."""
        if not claims or not passages:
            return [
                ClaimEvidenceScore(c["text"], c.get("hedged", False), 0.0, 0.0, 0.0, -1, [0.0] * len(passages))
                for c in claims
            ]
        # NLI cross-encoders expect (premise, hypothesis) = (passage, claim).
        # Reversed order collapses entailment to ~0 (model sees the claim as
        # the premise) — keep this order!
        pairs = [(p, c["text"]) for c in claims for p in passages]
        raw = self._predict_logits(pairs, batch_size=batch_size)
        if raw.ndim == 1:  # single pair
            raw = raw.reshape(1, -1)
        probs = _softmax(raw)

        out: list[ClaimEvidenceScore] = []
        n_p = len(passages)
        for i, c in enumerate(claims):
            block = probs[i * n_p : (i + 1) * n_p]
            ent = block[:, self._ent]
            con = block[:, self._con]
            neu = block[:, self._neu]
            best = int(np.argmax(ent))
            out.append(
                ClaimEvidenceScore(
                    claim=c["text"],
                    hedged=c.get("hedged", False),
                    best_entailment=float(ent[best]),
                    best_contradiction=float(con[best]),
                    mean_entailment=float(ent.mean()),
                    best_passage_idx=best,
                    per_passage_entailment=[float(x) for x in ent],
                    best_neutral=float(neu[best]),
                )
            )
        return out


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


class LazyNLIScorer:
    """Defers the heavy ONNX NLI session load until first ``score_claims``.

    On a 512 MB host the NLI session alone is ~260 MB, so loading it at
    startup alongside the embedding model leaves almost no headroom and can
    trip the OOM killer during the boot/health-check window. This wrapper
    keeps startup light and loads the model on first real use.
    """

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-small", device: str = "cpu"):
        self._model_name = model_name
        self._device = device
        self._nli: NLIScorer | None = None

    @property
    def is_loaded(self) -> bool:
        return self._nli is not None

    def _load(self) -> NLIScorer:
        if self._nli is None:
            log.info("Lazy-loading NLI ONNX model %s ...", self._model_name)
            self._nli = NLIScorer(self._model_name, device=self._device)
        return self._nli

    def score_claims(self, claims, passages, batch_size: int = 32):
        return self._load().score_claims(claims, passages, batch_size=batch_size)
