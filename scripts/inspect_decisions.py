"""Readable view of a run's planner decisions (decompose / replan / events / reports).

  python scripts/inspect_decisions.py runs/system_vr_h3_2_oracle                # timeline
  python scripts/inspect_decisions.py runs/system_vr_h3_2_oracle --show 2        # one decision in full
  python scripts/inspect_decisions.py runs/system_vr_h3_2_oracle --show 2 --prompt

Code-as-policy runs (agent.enabled): a decision is a program; the timeline prints it with line
numbers, each executed line marked with its result (L3 ok/FAIL), and --show N adds the agent rounds
(plan, verdict, code checks) from rounds.jsonl.
"""
from __future__ import annotations

import argparse
import json
import re
import textwrap
from pathlib import Path


def load(run_dir: Path) -> list[dict]:
    return [json.loads(x) for x in (run_dir / "decisions.jsonl").read_text().splitlines() if x.strip()]


def t_rel(rec: dict, t0: float) -> float:
    return rec["t"] - t0


def program_steps(recs: list[dict], decision: dict) -> dict[int, list[str]]:
    """Results of the program lines run by `decision` (program_step records up to the next decision)."""
    out: dict[int, list[str]] = {}
    started = False
    for r in recs:
        if r is decision:
            started = True
        elif started and r["kind"] in ("decompose", "replan"):
            break
        elif started and r["kind"] == "program_step":
            m = re.match(r"L(\d+): .*? -> (.*)", r["step"], re.S)
            if m:
                out.setdefault(int(m.group(1)), []).append(("ok  " if r["ok"] else "FAIL") + " " + m.group(2))
    return out


def print_program(code: str, steps: dict[int, list[str]], width: int, ind: str = "      ") -> None:
    """The program with line numbers; executed lines get their results on the right (or below if long)."""
    lines = code.rstrip("\n").splitlines()
    col = min(max(len(x) for x in lines) + 2, width // 2)
    for n, line in enumerate(lines, 1):
        res = steps.get(n, [])
        if not res:
            print(f"{ind}{n:3d} | {line}")
            continue
        for k, r in enumerate(res):
            left = f"{n:3d} | {line}" if k == 0 else "    |"
            if len(left) <= col + 6:
                print(f"{ind}{left:<{col + 6}}  <- {r}")
            else:
                print(f"{ind}{left}")
                print(f"{ind}{'':<{col + 6}}  <- {r}")


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
                if r.get("program"):
                    print(f"{t} #{k} {r['kind'].upper():9s} -> program ({r.get('rounds')} round(s)), actions "
                          + ", ".join(f"{x['type']} {x['target'][:40]!r}" for x in r.get("plan") or []))
                else:
                    print(f"{t} #{k} {r['kind'].upper():9s} -> {d.get('type', '-')} {d.get('target', '')!r}"
                          f"{' #' + str(d['node_id']) if d.get('node_id') is not None else ''}"
                          f"   (vlm {q.get('vlm_latency_s', '?')} s, fixes {len(r.get('fixes', []))})")
                print(wrap(r.get("reasoning", "")))
                if r.get("program"):                   # code-as-policy: the program, with what each line did
                    print(f"      plan: " + " | ".join(r.get("steps") or []))
                    print_program(r["program"], program_steps(recs, r), args.width)
                k += 1
            elif r["kind"] == "dispatch":
                d = r["dispatched"]
                print(f"{t}    dispatch {d['type']} {d.get('instruction') or d['target']!r}"
                      f"{' #' + str(d['node_id']) if d.get('node_id') is not None else ''}")
            elif r["kind"] == "agent_call":
                q = r.get("query") or {}
                verdict = (q.get("parsed") or {}).get("verdict") if r["role"] == "observer" else None
                print(f"{t}    agent {r['role']:8s} round {r.get('round')}  {q.get('vlm_latency_s', '?')} s"
                      + (f"  {verdict}" if verdict else "") + (f"  error: {q['error']}" if q.get("error") else ""))
            elif r["kind"] in ("program_end", "program_exception", "fallback", "deliberation_failed",
                               "observer_overruled"):
                rest = {x: v for x, v in r.items() if x not in ("t", "wall", "kind", "task_id", "fsm")}
                print(f"{t}    {r['kind'].upper()} {rest}")
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
    if r.get("program"):                               # code-as-policy decision
        print(f"-- plan steps:")
        for st in r.get("steps") or []:
            print(f"     {st}")
        print(f"-- program ({r.get('rounds')} round(s)), executed lines marked:")
        print_program(r["program"], program_steps(recs, r), args.width, "   ")
        rounds = Path(args.run_dir) / "rounds.jsonl"
        if rounds.exists():
            for x in map(json.loads, rounds.read_text().splitlines()):
                if x.get("task_id") == r["task_id"] and x.get("kind") == r["kind"] \
                        and x.get("memory_version") == r.get("memory_version"):
                    ob = x.get("observation") or {}
                    print(f"-- round {x['round']}: valid={x['valid']} {ob.get('verdict', '')} "
                          f"{'code checks: ' + '; '.join(x['hard_failures']) if x['hard_failures'] else ''}"
                          f"{' error: ' + x['error'] if x.get('error') else ''}")
                    print(wrap(ob.get("summary") or ""))
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
