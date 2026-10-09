"""Offline evaluation of target acquisition on labelled cases (acquisition spec §12, A6).

  python eval/acquisition/eval_acquisition.py --fixture eval/acquisition/fixtures/vr_h3_2/memory.json \
      --cases eval/acquisition/cases_vr_h3_2.yaml --out runs/eval_acq/default.csv [ablations]

Case YAML: instruction, target, gt_node_ids ([] = target absent), optional ambiguous_node_ids
(selecting one is neither correct nor a false commit), optional until_t.
Metrics: symbolic_recall@k (GT in the round-0 candidates), verification_acc (given GT in the
candidates, the selection is GT), end_to_end_acc (selection in GT, or no selection when GT is
empty), false_commit_rate, VLM latency mean / p95.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import yaml

from vlnav.acquisition.acquirer import TargetAcquirer
from vlnav.acquisition.config import load_acquisition_config
from vlnav.acquisition.label_matcher import LabelMatcher, make_text_embedder
from vlnav.acquisition.memory_api import InMemoryMemory
from vlnav.acquisition.vlm_client import load_vlm_config, make_vlm_client


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", required=True, help="CSV of per-case results (+ .json summary)")
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--rank-strategy", choices=["confidence", "confidence_x_similarity"], default=None)
    ap.add_argument("--no-full-frame", action="store_true")
    ap.add_argument("--no-neighbors", action="store_true")
    ap.add_argument("--no-vlm", action="store_true", help="symbolic top-1 is the answer")
    ap.add_argument("--vlm-backend", choices=["local", "openai"], default=None)
    args = ap.parse_args()

    over = {"logging": {"debug_jsonl": str(Path(args.out).with_suffix(".debug.jsonl"))}}
    if args.rank_strategy:
        over["symbolic"] = {"rank_strategy": args.rank_strategy}
    acfg = load_acquisition_config(overrides=over)
    sym = acfg.symbolic
    matcher = LabelMatcher(make_text_embedder(sym.text_encoder, sym.text_encoder_path, sym.device),
                           sym.text_template, sym.label_sim_threshold, sym.synonyms, sym.label_rel_margin)
    vlm = None
    if not args.no_vlm:
        vlm = make_vlm_client(load_vlm_config(overrides={"backend": args.vlm_backend} if args.vlm_backend else None))
    memory = InMemoryMemory.from_json(args.fixture, max_neighbors=acfg.context.max_neighbors,
                                      text_sim=matcher.sim)
    cases = yaml.safe_load(Path(args.cases).read_text())
    rows = []
    for i, c in enumerate(cases):
        acq = TargetAcquirer(acfg, matcher, vlm)            # fresh blacklist per case
        acq.k = args.k or sym.k
        acq.include_full_frame = False if args.no_full_frame else None
        acq.use_neighbors = not args.no_neighbors
        gt, amb = set(c.get("gt_node_ids", [])), set(c.get("ambiguous_node_ids", []))
        round0 = acq.filter.topk(c["target"], memory, acq.k, until_t=c.get("until_t"))
        t0 = time.perf_counter()
        r = acq.acquire(c["instruction"], c["target"], memory, until_t=c.get("until_t"))
        sel = r.node_id
        rows.append({
            "case": i, "target": c["target"], "status": r.status, "selected": sel,
            "gt": sorted(gt), "round0": [x.node.id for x in round0], "sent": r.candidates,
            "gt_in_topk": bool(gt & {x.node.id for x in round0}) if gt else None,
            "correct": (sel in gt) if gt else (sel is None),
            "acceptable": (sel in gt | amb) if gt else (sel is None),
            "false_commit": sel is not None and sel not in gt | amb,
            "vlm_s": r.latency_s.get("vlm", 0.0), "total_s": time.perf_counter() - t0,
            "confidence": r.vlm_confidence, "reason": (r.vlm_reason or "")[:300]})
        print(f"[{i}] {c['target']:12s} {r.status:13s} sel={sel} gt={sorted(gt)} round0={rows[-1]['round0']} "
              f"{'OK' if rows[-1]['correct'] else ('~' if rows[-1]['acceptable'] else 'X')}", flush=True)

    with_gt = [r for r in rows if r["gt"]]
    in_topk = [r for r in with_gt if r["gt_in_topk"]]
    lat = [r["vlm_s"] for r in rows if r["vlm_s"] > 0]
    summary = {
        "cases": len(rows), "k": args.k or sym.k, "no_vlm": args.no_vlm,
        "no_full_frame": args.no_full_frame, "no_neighbors": args.no_neighbors,
        "rank_strategy": sym.rank_strategy,
        "symbolic_recall@k": float(np.mean([r["gt_in_topk"] for r in with_gt])) if with_gt else None,
        "verification_acc": float(np.mean([r["correct"] for r in in_topk])) if in_topk else None,
        "end_to_end_acc": float(np.mean([r["correct"] for r in rows])),
        "end_to_end_acceptable": float(np.mean([r["acceptable"] for r in rows])),
        "false_commit_rate": float(np.mean([r["false_commit"] for r in rows])),
        "vlm_latency_mean_s": float(np.mean(lat)) if lat else None,
        "vlm_latency_p95_s": float(np.percentile(lat, 95)) if lat else None,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
