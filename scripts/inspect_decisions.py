"""Readable view of a run's planner decisions (decompose / replan / events / reports).

  python scripts/inspect_decisions.py runs/system_vr_h3_2_oracle                # timeline
  python scripts/inspect_decisions.py runs/system_vr_h3_2_oracle --show 2        # one decision in full
  python scripts/inspect_decisions.py runs/system_vr_h3_2_oracle --show 2 --prompt
"""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path


def load(run_dir: Path) -> list[dict]:
    return [json.loads(x) for x in (run_dir / "decisions.jsonl").read_text().splitlines() if x.strip()]


def t_rel(rec: dict, t0: float) -> float:
    return rec["t"] - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--show", type=int, default=None, help="index of a decompose/replan decision")
    ap.add_argument("--prompt", action="store_true", help="print the full prompt text")
    ap.add_argument("--width", type=int, default=110)
    args = ap.parse_args()
    recs = load(Path(args.run_dir))
    t0 = next((r["t"] for r in recs if r["kind"] == "new_task"), recs[0]["t"])
    decisions = [r for r in recs if r["kind"] in ("decompose", "replan")]
    wrap = lambda s, ind="      ": textwrap.fill(str(s), args.width, initial_indent=ind, subsequent_indent=ind)

    if args.show is None:
        k = 0
        for r in recs:
            t = f"[{t_rel(r, t0):7.1f}s]"
            if r["kind"] in ("decompose", "replan"):
                d = r.get("dispatched") or {}
                q = r.get("query") or {}
                print(f"{t} #{k} {r['kind'].upper():9s} -> {d.get('type', '-')} {d.get('target', '')!r}"
                      f"{' #' + str(d['node_id']) if d.get('node_id') is not None else ''}"
                      f"   (vlm {q.get('vlm_latency_s', '?')} s, fixes {len(r.get('fixes', []))})")
                print(wrap(r.get("reasoning", "")))
                k += 1
            elif r["kind"] == "subtask_end":
                e = r["event"]
                print(f"{t}    event {e['type']} {e.get('payload', {})}")
            elif r["kind"] == "report":
                rep = r["report"]
                print(f"{t}    REPORT {rep['target']}: #{rep['node_id']} {rep['label']}")
            elif r["kind"] == "finish":
                print(f"{t}    FINISH {r['fsm']}: {r['reason']}  "
                      f"{[(x['name'], x['status']) for x in r.get('targets', [])]}")
            elif r["kind"] == "new_task":
                print(f"{t}    NEW TASK: {r.get('instruction')}")
        print(f"\n{len(decisions)} decisions; details: --show N [--prompt]; debug folders: "
              f"{Path(args.run_dir) / 'planner_debug'}")
        return

    r = decisions[args.show]
    q = r.get("query") or {}
    print(f"== decision #{args.show}: {r['kind']} at {t_rel(r, t0):.1f}s  (fsm {r['fsm']})")
    inp = r.get("input") or {}
    if inp:
        print(f"-- input: last_event = {inp.get('last_event')!r}, memory version {inp.get('memory_version')}")
        for t in inp.get("targets", []):
            print(f"     target {t['name']!r}: {t['status']} attempts={t['attempts']} notes={t['notes']!r}")
    if args.prompt:
        print("-- prompt text:\n" + (q.get("prompt_text") or "(not logged in this run)"))
    print(f"-- images sent: {q.get('images')}")
    for k, raw in enumerate(q.get("raw", [])):
        print(f"-- VLM reply {k}:\n{raw}")
    if q.get("error"):
        print(f"-- error: {q['error']}")
    print(f"-- validated plan: {r.get('plan')}")
    print(f"-- fixes: {r.get('fixes')}")
    if r.get("acquisition"):
        a = r["acquisition"]
        print(f"-- acquisition: {a['status']} node={a['node_id']} candidates={a['candidates']} "
              f"conf={a['vlm_confidence']}\n{wrap(a.get('vlm_reason') or '')}")
    print(f"-- dispatched: {r.get('dispatched')}")
    if r.get("debug_dir"):
        print(f"-- debug folder (prompt.txt, images/, reply_*.txt, input.json, result.json, memory/): {r['debug_dir']}")


if __name__ == "__main__":
    main()
