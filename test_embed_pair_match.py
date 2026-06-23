"""
Test suite: _embed_pair_match — validates the finding↔RAG-chunk pairing step
that runs inside validator_node, BEFORE Jordan sees any data.

Problems being tested:
  1. False-match rate at threshold 0.62 — unrelated industry chunks pulled in
  2. Whether raising the threshold to 0.70 eliminates false matches cleanly
  3. Double-normalization bug in the current implementation
  4. Score distribution: gap between legitimate vs false matches

Usage:
    python test_embed_pair_match.py          # full report
    python test_embed_pair_match.py --fix    # also show scores with threshold=0.70
"""
import sys
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from sentence_transformers import SentenceTransformer


# ── Reproduce _embed_pair_match with score visibility ─────────────────────────

def embed_pair_match_debug(findings, rag_chunks, threshold, label=""):
    """
    Same logic as graph._embed_pair_match but returns full score matrix
    and flags double-normalisation.
    """
    _st = SentenceTransformer("BAAI/bge-small-en-v1.5")
    prefix = "Represent this sentence for searching relevant passages: "

    # Current code: normalize_embeddings=True AND manual re-norm → double normalisation
    f_embs = _st.encode([prefix + t for t in findings], normalize_embeddings=True)
    r_embs = _st.encode([c["text"] for c in rag_chunks], normalize_embeddings=True)

    # Bug: already unit-normed, so dividing again by norm ≈ 1+ε is a no-op
    # but wastes a divide and could introduce tiny numerical drift.
    f_norm = f_embs / (np.linalg.norm(f_embs, axis=1, keepdims=True) + 1e-9)
    r_norm = r_embs / (np.linalg.norm(r_embs, axis=1, keepdims=True) + 1e-9)

    sim = f_norm @ r_norm.T  # shape: (n_findings, n_chunks)

    matched, supplements, all_scores = [], [], []
    for j, chunk in enumerate(rag_chunks):
        col_sim = sim[:, j]
        best    = int(col_sim.argmax())
        max_sim = float(col_sim[best])
        all_scores.append((chunk["filename"], max_sim, findings[best]))
        if max_sim >= threshold:
            matched.append({
                **chunk,
                "best_finding": findings[best],
                "sim": round(max_sim, 3),
            })
        else:
            supplements.append(chunk)

    return matched, supplements, all_scores, sim


# ── Ground-truth labels for each RAG file (for this cloud/SaaS topic) ─────────

RELEVANT_FILES = {
    "cloud_vendors.csv",
    "saas_market_report.md",
    "cloud_industry_stats.json",
    "cloud_industry_stats.xlsx",
    "gartner_cloud_market_summary_2025.md",
    "global_saas_market_2025.csv",
    "idc_cloud_forecast_2025.pdf",
    "saas_and_cloud_software_market.txt",  # borderline but same industry
}

IRRELEVANT_FILES = {
    "ai_and_machine_learning_market.txt",
    "cybersecurity_market.txt",
    "digital_health_and_medtech.txt",
    "e-commerce_and_retail_tech.txt",
    "electric_vehicle_market.txt",
    "fintech_and_payments.txt",
}


def classify(filename):
    if filename in RELEVANT_FILES:   return "relevant"
    if filename in IRRELEVANT_FILES: return "irrelevant"
    return "unknown"


# ── Test runner ───────────────────────────────────────────────────────────────

def run(findings, rag_chunks, threshold):
    matched, supplements, all_scores, sim_matrix = embed_pair_match_debug(
        findings, rag_chunks, threshold
    )

    tp = sum(1 for m in matched     if classify(m["filename"]) == "relevant")
    fp = sum(1 for m in matched     if classify(m["filename"]) == "irrelevant")
    fn = sum(1 for s in supplements if classify(s["filename"]) == "relevant")
    tn = sum(1 for s in supplements if classify(s["filename"]) == "irrelevant")

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall    = tp / (tp + fn) if (tp + fn) else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    return {
        "threshold": threshold,
        "matched":   matched,
        "supplements": supplements,
        "all_scores": all_scores,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
    }


def print_report(r, verbose=True):
    t = r["threshold"]
    print(f"\nThreshold={t:.2f}  matched={r['tp']+r['fp']}  "
          f"TP={r['tp']} FP={r['fp']} FN={r['fn']} TN={r['tn']}  "
          f"P={r['precision']:.2f} R={r['recall']:.2f} F1={r['f1']:.2f}")

    if not verbose:
        return

    print("\n  MATCHED pairs:")
    for m in r["matched"]:
        tag = classify(m["filename"])
        ok  = "OK  " if tag == "relevant" else "WRONG"
        print(f"    [{ok}] sim={m['sim']}  [{m['filename']}]")
        print(f"           Web : {m['best_finding'][:90]}")
        print(f"           RAG : {m['text'][:90]}")

    print("\n  SUPPLEMENTS (RAG-only, not passed to Jordan):")
    for s in r["supplements"]:
        tag = classify(s["filename"])
        ok  = "OK  " if tag == "irrelevant" else "MISS"
        print(f"    [{ok}] [{s['filename']}] {s['text'][:75]}")


def score_distribution(all_scores):
    """Print min/max/mean similarity per file for a clear picture."""
    from collections import defaultdict
    by_file = defaultdict(list)
    for fname, score, _ in all_scores:
        by_file[fname].append(score)

    print("\n  Score distribution (best similarity against any web finding):")
    print(f"  {'File':<45} {'max':>6}  {'mean':>6}  Label")
    print("  " + "─"*70)
    for fname, scores in sorted(by_file.items(), key=lambda x: -max(x[1])):
        label = classify(fname)
        print(f"  {fname:<45} {max(scores):.3f}  {sum(scores)/len(scores):.3f}  {label}")


def main():
    show_fix = "--fix" in sys.argv or "--both" in sys.argv

    from agents.graph import _get_all_user_rag_chunks
    rag_chunks = _get_all_user_rag_chunks()

    findings = [
        "Global cloud market revenue in Q3 2025: $107 billion [Data] — CRN (2025)",
        "Combined market share of AWS, Google Cloud, and Microsoft in Q3 2025: 62% [Data] — CRN (2025)",
        "Oracle's market share in Q3 2025: 3% [Data] — CRN (2025)",
        "Q2 2025 enterprise spending on cloud infrastructure services: almost $99B [Data] — Synergy Research Group (2025)",
        "Trailing twelve-month revenues in 2025: $366B [Data] — Synergy Research Group (2025)",
    ]

    print("=" * 70)
    print("  _embed_pair_match test — topic: global SaaS / cloud market 2025")
    print(f"  {len(findings)} web findings  |  {len(rag_chunks)} RAG chunks from "
          f"{len(set(c['filename'] for c in rag_chunks))} files")
    print("=" * 70)

    print("\n[1] Web findings:")
    for i, f in enumerate(findings):
        print(f"  [{i}] {f[:100]}")

    # ── Double-normalization check ────────────────────────────────────────────
    print("\n[2] Double-normalization bug check:")
    _st = SentenceTransformer("BAAI/bge-small-en-v1.5")
    sample = _st.encode(["test sentence"], normalize_embeddings=True)
    norm_before = float(np.linalg.norm(sample[0]))
    renormed = sample / (np.linalg.norm(sample, axis=1, keepdims=True) + 1e-9)
    norm_after  = float(np.linalg.norm(renormed[0]))
    delta = abs(norm_before - norm_after)
    print(f"  norm before re-norm: {norm_before:.8f}")
    print(f"  norm after  re-norm: {norm_after:.8f}")
    print(f"  delta: {delta:.2e}  →  {'no-op (bug confirmed but harmless)' if delta < 1e-6 else 'significant!'}")

    # ── Score distribution ────────────────────────────────────────────────────
    r_orig = run(findings, rag_chunks, threshold=0.62)
    score_distribution(r_orig["all_scores"])

    rel_scores   = [s for fn, s, _ in r_orig["all_scores"] if classify(fn) == "relevant"]
    irrel_scores = [s for fn, s, _ in r_orig["all_scores"] if classify(fn) == "irrelevant"]
    print(f"\n  Relevant   max={max(rel_scores):.3f}  min={min(rel_scores):.3f}  mean={sum(rel_scores)/len(rel_scores):.3f}")
    print(f"  Irrelevant max={max(irrel_scores):.3f}  min={min(irrel_scores):.3f}  mean={sum(irrel_scores)/len(irrel_scores):.3f}")

    # ── Threshold sweep ───────────────────────────────────────────────────────
    print("\n[3] Threshold sweep:")
    print(f"  {'Threshold':>10}  {'Matched':>8}  {'TP':>4}  {'FP':>4}  {'FN':>4}  {'TN':>4}  "
          f"{'Prec':>6}  {'Recall':>7}  {'F1':>5}")
    print("  " + "─" * 70)
    best_f1, best_t = 0, 0
    for t_int in range(60, 80):
        t = t_int / 100
        r = run(findings, rag_chunks, t)
        marker = " ←" if r["f1"] > best_f1 else ""
        if r["f1"] > best_f1:
            best_f1 = r["f1"]
            best_t  = t
        print(f"  {t:>10.2f}  {r['tp']+r['fp']:>8}  {r['tp']:>4}  {r['fp']:>4}  "
              f"{r['fn']:>4}  {r['tn']:>4}  {r['precision']:>6.2f}  "
              f"{r['recall']:>7.2f}  {r['f1']:>5.2f}{marker}")
    print(f"\n  Best threshold: {best_t:.2f}  (F1={best_f1:.2f})")

    # ── Detailed report at 0.62 (current) ────────────────────────────────────
    print("\n[4] Detailed results — current threshold 0.62:")
    print_report(r_orig, verbose=True)

    # ── Detailed report at best threshold ────────────────────────────────────
    if show_fix or best_t != 0.62:
        r_fix = run(findings, rag_chunks, threshold=best_t)
        print(f"\n[5] Detailed results — fixed threshold {best_t:.2f}:")
        print_report(r_fix, verbose=True)

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Current  (0.62): {r_orig['fp']} false matches / {r_orig['tp']+r_orig['fp']} total  "
          f"— F1={r_orig['f1']:.2f}")
    if show_fix or best_t != 0.62:
        print(f"  Fixed  ({best_t:.2f}): {r_fix['fp']} false matches / {r_fix['tp']+r_fix['fp']} total  "
              f"— F1={r_fix['f1']:.2f}")
    print()
    print("  Bug: double-normalisation in _embed_pair_match (lines 340-341).")
    print("       normalize_embeddings=True already produces unit vectors.")
    print("       The manual re-norm is a no-op but adds noise via +1e-9 divisor.")


if __name__ == "__main__":
    main()
