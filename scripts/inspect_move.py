"""Readable view of a move run (TASK_move_subtask.md §5): one line per S2 cycle from move_log.jsonl.

    python scripts/inspect_move.py runs/move_vr_h3_2/colorful_wall_corridor [--json K]

Columns: time since the case start, subtask:cycle, S2 texts (LD = look-down query), decision / reason,
view pixel, floor range / measured range, projection + goal flags, goal (x, y, yaw), distance of the
goal (and of the projected point) to the recorded future path, S2 latency. --json K prints cycle K
(of the first subtask, or "sub:K") in full.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def fmt(v, n=2):
    return "-" if v is None else f"{v:.{n}f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case_dir")
    ap.add_argument("--json", default=None, help="cycle to print in full: K or SUB:K")
    args = ap.parse_args()
    d = Path(args.case_dir)
    recs = [json.loads(line) for line in open(d / "move_log.jsonl")]
    summary = json.loads((d / "summary.json").read_text()) if (d / "summary.json").exists() else {}
    if args.json:
        sub, k = (args.json.split(":") if ":" in args.json else ("1", args.json))
        print(json.dumps(next(r for r in recs if r["subtask_id"] == int(sub) and r["k"] == int(k)), indent=2))
        return
    if summary:
        print(f"{summary['case']}: {summary['instruction']}")
    t0 = recs[0]["t_req"] if recs else 0.0
    print(f"{'t':>6} {'cyc':>5} {'S2':28} {'decision':22} {'pixel':11} {'r_fl':>5} {'r_ms':>5} "
          f"{'flags':28} {'goal':22} {'wp':>5} {'pp':>5} {'ms':>9}")
    for r in recs:
        p, g = r.get("projection") or {}, r.get("goal_result") or {}
        texts = " | ".join(("LD " if e["look_down"] else "") + e["text"] for e in r["exchanges"])
        dec = r["decision"] + (f" ({r['reason']})" if r["reason"] and r["reason"] != r["decision"] else "")
        pix = "-" if not r.get("view_xy") else f"{r['view_xy'][0]:.0f},{r['view_xy'][1]:.0f}"
        flags = ",".join(p.get("flags", []) + g.get("flags", []))
        goal = "-" if not r.get("goal") else f"{r['goal'][0]:.1f},{r['goal'][1]:.1f},{r['goal'][2]:.2f}"
        ms = "+".join(f"{e['latency_ms']:.0f}" + ("c" if e["cached"] else "") for e in r["exchanges"])
        print(f"{r['t_req'] - t0:6.1f} {r['subtask_id']:>2}:{r['k']:<2} {texts[:28]:28} {dec[:22]:22} {pix:11} "
              f"{fmt(p.get('r_floor'), 1):>5} {fmt(p.get('r_meas'), 1):>5} {flags[:28]:28} {goal:22} "
              f"{fmt(r.get('waypoint_dist_m')):>5} {fmt(r.get('proj_dist_m')):>5} {ms:>9}")
    if summary:
        w, v = summary["waypoint_path"], summary["projection_validity"]
        print(f"\ncycles {summary['cycles']}  queries {summary['queries']}  goals {v['became_goal']}/{v['pixel_replies']}"
              f"  waypoint median {w['median_m']} m, < 0.5 m {w['frac_lt_0.5m']}  events {summary['events']}"
              f"  first STOP {summary['stop']['first_t_rel']} s")


if __name__ == "__main__":
    main()
