"""Readable view of an exploration shadow log (<run-dir>/exploration/shadow_log.jsonl).

  python scripts/inspect_exploration.py runs/expl                  # goal timeline: every goal change + cause, events
  python scripts/inspect_exploration.py runs/expl --step 40        # one step: goal, ranked candidates, instances
  python scripts/inspect_exploration.py runs/expl --step 40 --why 110   # why frontier #110 was not the goal
  python scripts/inspect_exploration.py runs/expl --point 202      # history of one frontier point
  python scripts/inspect_exploration.py runs/expl --instance 1     # history of one instance point

`--step` takes the step index `k` printed in the video / key frame names (frame_00040.png = k 40).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def load(run_dir: Path) -> list[dict]:
    path = run_dir / "exploration" / "shadow_log.jsonl"
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def margin(run_dir: Path) -> tuple[float, float]:
    """(switch_margin, delta_reached_m) of the run (run_config.yaml overrides, else defaults)."""
    from vlnav.exploration.config import load_exploration_config
    rc = yaml.safe_load((run_dir / "run_config.yaml").read_text()) if (run_dir / "run_config.yaml").exists() else {}
    cfg = load_exploration_config(overrides=(rc or {}).get("exploration"))
    return cfg.selection.switch_margin, cfg.selection.delta_reached_m


def key(rec: dict):
    g = rec.get("goal")
    return None if g is None else (rec["goal_kind"], g["id"])


def ranked(rec: dict) -> list[dict]:
    return sorted((p for p in rec["points"] if p["score"]), key=lambda p: (-p["score"]["s_nesy"], p["id"]))


def switch_cause(prev: tuple, rec: dict, switch_margin: float) -> str:
    kind, gid = prev
    new = rec["goal"]["score"]
    if kind == "goto" or rec["goal_kind"] == "goto":
        return "subtask changed"
    if kind == "instance":
        q = {q["id"]: q for q in rec["instances"]}.get(gid)
        if q is None or q["status"] != "active":
            return f"instance #{gid} " + (q["status"] if q else "gone (new subtask)")
        if rec["goal_kind"] == "instance" and new > q["rank"] + switch_margin:
            return f"outscored: new rank {new:.3f} vs #{gid} {q['rank']:.3f}"
        return f"instance #{gid} not reachable this step (no reachable approach cell)"
    if rec["goal_kind"] == "instance":
        return "instance preempts frontier"
    pts = {p["id"]: p for p in rec["points"]}
    if gid not in pts:
        return f"frontier #{gid} removed (revealed / cluster too small)"
    p = pts[gid]
    if p["blacklisted"]:
        return f"frontier #{gid} blacklisted"
    if p["score"] is None:
        return f"frontier #{gid} within delta_reached"
    old = p["score"]["s_nesy"]
    if new > old + switch_margin:
        return f"outscored: new {new:.3f} vs #{gid} {old:.3f} (> switch_margin {switch_margin})"
    return f"frontier #{gid} not reachable this step (plan failed; S_NeSy {old:.3f})"


def fmt_point(p: dict, i: int | None = None) -> str:
    s = p["score"]
    head = f"{'' if i is None else f'{i:>2}.'} F{p['id']:<5} xy ({p['xy'][0]:6.2f}, {p['xy'][1]:6.2f})"
    if s is None:
        return head + "  not selectable" + (" (blacklisted)" if p["blacklisted"] else " (within delta_reached)")
    return (head + f"  S_NeSy {s['s_nesy']:.3f} = S_dist {s['s_dist']:.3f} (d {s['dist']:.1f} m)"
            f" + VL {s['s_vl']:.3f} x unk {s['s_unknown']:.2f}" + (f" + room {s['room_bonus']:.2f}" if s["room_bonus"] else ""))


def show_step(recs: list[dict], k: int, why: int | None, run_dir: Path, top: int) -> None:
    rec = next((r for r in recs if r["k"] == k), None)
    if rec is None:
        raise SystemExit(f"no step k={k} (steps: {recs[0]['k']}..{recs[-1]['k']})")
    st, g = rec["subtask"], rec["goal"]
    print(f"step k={k}  t={rec['t_rel']:.1f} s  robot ({rec['robot'][0]:.2f}, {rec['robot'][1]:.2f}) "
          f"yaw {rec['robot'][2]:+.2f}")
    if st:
        print(f"subtask #{st['id']} {st['type']} '{st['target']}' prompts {st['prompts']}"
              + (f" room_hint {st['room_hint']}" if st.get("room_hint") else ""))
    print(f"goal: {rec['goal_kind']}" + (f" #{g['id']} at ({g['xy'][0]:.2f}, {g['xy'][1]:.2f}) score {g['score']}"
                                        f" path {rec['path_len_m']} m" if g else ""))
    mix = rec.get("mixture")
    if mix and mix["labels"]:
        print("VL detections this frame: " + ", ".join(f"{lb} conf {a:.2f} bearing {m * 57.2958:+.0f} deg"
                                                     for lb, a, m in zip(mix["labels"], mix["alpha"], mix["mu"])))
    r = ranked(rec)
    print(f"\nfrontier: {rec['n_frontier_cells']} cells, {len(rec['points'])} points, {len(r)} selectable; top {top}:")
    for i, p in enumerate(r[:top], 1):
        print(("* " if g and rec["goal_kind"] == "frontier" and p["id"] == g["id"] else "  ") + fmt_point(p, i))
    if g and rec["goal_kind"] == "frontier" and r and r[0]["id"] != g["id"]:
        sm, _ = margin(run_dir)
        print(f"  -> goal is not the top candidate: kept by hysteresis (top must beat it by switch_margin {sm})"
              " or the better ones were unreachable (see notes)")
    if rec["instances"]:
        print("\ninstances (selected first: highest rank among active, reachable):")
        for q in sorted(rec["instances"], key=lambda q: (q["status"] != "active", -q["rank"])):
            a = q["approach_xy"]
            print(f"  I{q['id']:<4} {q['label']:<14} rank {q['rank']:.3f} conf {q['score']:.2f} {q['status']:<11}"
                  f" at ({q['xy'][0]:.2f}, {q['xy'][1]:.2f})" + (f" approach ({a[0]:.2f}, {a[1]:.2f})" if a else ""))
    for e in rec["events"]:
        print(f"EVENT {e['type']} {e['payload']}")
    for n in rec["notes"]:
        print(f"note: {n}")
    print(f"timing {rec['timing_ms']}  plans tried {rec['reach_checks']}")
    if why is not None:
        pts = {p["id"]: p for p in rec["points"]}
        print(f"\nwhy not F{why}?")
        if why not in pts:
            print("  not a frontier point at this step (never created, removed, or re-sampled with another id)")
        elif pts[why]["score"] is None:
            print("  " + fmt_point(pts[why]))
        elif rec["goal_kind"] == "instance":
            print("  an instance point is active and reachable: instances are always selected before frontiers")
        elif g and rec["goal_kind"] == "frontier":
            p, cur = pts[why], pts.get(g["id"])
            print("  " + fmt_point(p))
            if cur is not None and cur["score"]:
                print("  goal: " + fmt_point(cur))
                d = p["score"]["s_nesy"] - cur["score"]["s_nesy"]
                sm, _ = margin(run_dir)
                print(f"  difference {d:+.3f}: " + ("lower score" if d <= 0 else
                      f"better but within switch_margin {sm} (hysteresis keeps the goal)" if d <= sm else
                      "better by more than the margin: it was unreachable (plan failed) or not checked "
                      "(max_reach_checks)"))
        else:
            print("  " + fmt_point(pts[why]) + f"  (goal kind: {rec['goal_kind']})")


def history(recs: list[dict], pid: int, instance: bool) -> None:
    seen = False
    for rec in recs:
        items = {q["id"]: q for q in (rec["instances"] if instance else rec["points"])}
        q = items.get(pid)
        if q is None:
            if seen:
                print(f"k {rec['k']:>4}  t {rec['t_rel']:6.1f}  removed")
                return
            continue
        seen = True
        goal = " <- GOAL" if key(rec) == ("instance" if instance else "frontier", pid) else ""
        if instance:
            print(f"k {rec['k']:>4}  t {rec['t_rel']:6.1f}  {q['label']} rank {q['rank']:.3f} conf {q['score']:.2f} "
                  f"{q['status']} xy ({q['xy'][0]:.2f}, {q['xy'][1]:.2f}){goal}")
        else:
            print(f"k {rec['k']:>4}  t {rec['t_rel']:6.1f}  s_vl {q['s_vl']:.3f} scored {str(q['scored']):5}  "
                  + fmt_point(q).split("  ", 1)[1] + goal)
    if not seen:
        raise SystemExit(f"{'instance' if instance else 'frontier point'} {pid} never appears")


def timeline(recs: list[dict], switch_margin: float) -> None:
    prev, n_sw = None, 0
    for rec in recs:
        k = key(rec)
        if k != prev:
            g = rec["goal"]
            cause = "" if prev is None or k is None else f"   [{switch_cause(prev, rec, switch_margin)}]"
            n_sw += prev is not None and k is not None
            what = "hold" if k is None else f"{k[0]} #{k[1]}" + (f" score {g['score']:.3f}" if g.get("score") is not None else "")
            print(f"k {rec['k']:>4}  t {rec['t_rel']:6.1f}  goal -> {what}{cause}")
            prev = k if k is not None else prev
        for e in rec["events"]:
            print(f"k {rec['k']:>4}  t {rec['t_rel']:6.1f}  EVENT {e['type']} {e['payload']}")
    last = recs[-1]
    print(f"\n{len(recs)} steps, {n_sw} goal switches, last step k={last['k']} t={last['t_rel']:.1f} s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--step", type=int, help="show one step in full")
    ap.add_argument("--why", type=int, help="with --step: why this frontier point id was not the goal")
    ap.add_argument("--top", type=int, default=10, help="candidates listed with --step")
    ap.add_argument("--point", type=int, help="history of a frontier point id")
    ap.add_argument("--instance", type=int, help="history of an instance point id")
    args = ap.parse_args()
    recs = load(args.run_dir)
    if args.step is not None:
        show_step(recs, args.step, args.why, args.run_dir, args.top)
    elif args.point is not None:
        history(recs, args.point, instance=False)
    elif args.instance is not None:
        history(recs, args.instance, instance=True)
    else:
        timeline(recs, margin(args.run_dir)[0])


if __name__ == "__main__":
    main()
