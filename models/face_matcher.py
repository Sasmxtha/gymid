
import os, logging
import numpy as np
from typing import Optional, Dict, List, Tuple

log = logging.getLogger("gymid.matcher")

# ── Thresholds per backend ────────────────────────────────────────────────────
# buffalo_l  same-person: 0.55-0.95  different: 0.10-0.45  → use 0.50
# buffalo_sc same-person: 0.35-0.75  different: 0.05-0.35  → use 0.38
# face_recog same-person: 0.40-0.80  different: 0.05-0.35  → use 0.36
DEFAULTS = {
    "buffalo_l":        0.50,
    "buffalo_sc":       0.38,
    "opencv_cascade":   0.45,
}

class FaceMatcher:
    """
    Matches a query embedding against all stored embeddings.

    Scoring: top-5 cosine similarity mean × 0.7 + max × 0.3
    Rejects if: score < threshold  OR  margin vs 2nd-best < 0.02

    The threshold for buffalo_l is 0.50 by default — much higher than
    buffalo_sc's 0.38 because buffalo_l embeddings are more discriminative
    (genuine matches score higher, impostors score lower).
    """
    def __init__(self, backend: str = "buffalo_l"):
        self.backend = backend
        env = os.environ.get("MATCH_THRESHOLD", "")
        self.threshold = float(env) if env else DEFAULTS.get(backend, 0.50)
        log.info(f"Matcher: backend={backend} threshold={self.threshold}")

    def match(
        self,
        query_emb: np.ndarray,
        all_embeddings: Dict[str, List[List[float]]]
    ) -> Tuple[Optional[str], float, dict]:

        if not all_embeddings:
            return None, 0.0, {"reason": "no_members"}

        query = self._norm(query_emb)
        scores = {}

        for mid, emb_list in all_embeddings.items():
            if not emb_list:
                continue
            sims = sorted(
                [float(np.dot(query, self._norm(np.array(e, dtype=np.float32)))) for e in emb_list],
                reverse=True
            )
            top_k    = sims[:min(5, len(sims))]
            mean_s   = float(np.mean(top_k))
            max_s    = float(sims[0])
            weighted = mean_s * 0.7 + max_s * 0.3
            scores[mid] = {"mean": mean_s, "max": max_s, "weighted": weighted, "n": len(sims)}

        if not scores:
            return None, 0.0, {"reason": "no_scores"}

        ranked  = sorted(scores.items(), key=lambda x: x[1]["weighted"], reverse=True)
        best_id = ranked[0][0]
        best    = ranked[0][1]

        debug = {
            "best_id":       best_id,
            "best_mean":     round(best["mean"],     4),
            "best_max":      round(best["max"],      4),
            "best_weighted": round(best["weighted"], 4),
            "threshold":     self.threshold,
            "all_scores":    {mid: round(s["weighted"], 4) for mid, s in ranked},
        }

        # Gate 1 — must beat threshold
        if best["weighted"] < self.threshold:
            pct = best["weighted"] / self.threshold * 100
            debug["reason"] = f"below_threshold ({best['weighted']:.3f} < {self.threshold} | {pct:.0f}%)"
            return None, best["mean"], debug

        # Gate 2 — must clearly beat 2nd best (anti-confusion)
        if len(ranked) >= 2:
            margin = best["weighted"] - ranked[1][1]["weighted"]
            debug["margin"] = round(margin, 4)
            if margin < 0.02:
                debug["reason"] = f"ambiguous_tie (margin={margin:.4f})"
                return None, best["mean"], debug

        debug["reason"] = "matched"
        log.info(f"MATCH {best_id}: weighted={best['weighted']:.3f} mean={best['mean']:.3f}")
        return best_id, best["mean"], debug

    @staticmethod
    def _norm(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return v / (n + 1e-9)
