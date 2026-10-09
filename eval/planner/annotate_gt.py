"""Start a planner GT file (planner spec §9) from a saved scene graph: pick nodes per category,
their centroids become GT instance positions (node ids differ between runs, positions do not).

  python eval/planner/annotate_gt.py --graph runs/memory_vr_h3_2/memory/final \
      --select "trash can=0,198" "sofa=4,7,179" --out eval/planner/gt_vr_h3_2.yaml
  python eval/planner/annotate_gt.py --graph runs/memory_vr_h3_2/memory/final --list   # inspect
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True, help="checkpoint dir with graph.json")
    ap.add_argument("--select", nargs="*", default=[], help='"category=id,id,..." (repeatable)')
    ap.add_argument("--list", action="store_true", help="print confirmed nodes grouped by label")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    g = json.loads((Path(args.graph) / "graph.json").read_text())
    objs = {o["id"]: o for o in g["objects"]}
    if args.list:
        by = {}
        for o in g["objects"]:
            if o.get("confirmed", True):
                by.setdefault(o["label"], []).append(o)
        for lbl, os_ in sorted(by.items(), key=lambda kv: -len(kv[1])):
            print(f"{lbl} ({len(os_)}): " + ", ".join(f"#{o['id']}({o['best_score']:.2f})" for o in os_))
    entries = []
    for sel in args.select:
        cat, ids = sel.split("=")
        for i in (int(x) for x in ids.split(",") if x.strip()):
            o = objs[i]
            entries.append({"gt_id": f"{cat.strip().replace(' ', '_')}_{i}", "category": cat.strip(),
                            "position": [round(v, 3) for v in o["centroid"]], "source_node": i,
                            "room": None})
    if args.out:
        doc = {"objects": entries, "tasks": []}
        Path(args.out).write_text(yaml.safe_dump(doc, sort_keys=False))
        print(f"{len(entries)} GT objects -> {args.out} (add `tasks` by hand)")


if __name__ == "__main__":
    main()
