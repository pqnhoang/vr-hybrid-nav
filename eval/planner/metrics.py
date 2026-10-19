"""Planner open-loop metrics (planner spec §9) from run directories + a GT file.

Per task: target set precision / recall (decomposed targets vs GT target keys, by label
similarity or `aliases`), report precision (report within `report_radius_m` of a GT instance of
its target), task success (every present GT target correctly reported, no wrong report, no report
for absent targets), false-commit rate (goto to an object not near a GT instance of the target),
time to first correct report, VLM calls and latency.

Move subtasks (TASK_move_subtask.md M5): a task may give `gt_order`, the expected order of the
decomposition, with "route" for a `move` subtask and GT target keys otherwise (e.g. [route, sofa]).
`routes_ok`: the decomposition has as many move subtasks as `gt_order` has routes; `order_ok`: the
decomposed sequence (moves -> "route", targets mapped to GT keys, repeats merged) equals `gt_order`.
With `gt_order`, success also needs `routes_ok`. A GT file may take its objects from another one
(`objects_from`).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml


def _norm(s: str) -> str:
    return " ".join(str(s).lower().split())


class TargetMapper:
    """Maps a planner target name to a GT target key (aliases first, then label similarity)."""

    def __init__(self, gt_targets: dict, aliases: dict, sim=None, thr: float = 0.88):
        self.keys = list(gt_targets)
        self.alias = {_norm(a): k for k, al in aliases.items() for a in [k, *al]}
        self.sim, self.thr = sim, thr

    def __call__(self, name: str) -> str | None:
        n = _norm(name)
        if n in self.alias:
            return self.alias[n]
        if self.sim is None:
            return None
        best = max(self.keys, key=lambda k: self.sim(n, k), default=None)
        return best if best is not None and self.sim(n, best) >= self.thr else None


def load_gt(path: str | Path) -> dict:
    path = Path(path)
    gt = yaml.safe_load(path.read_text())
    if "objects_from" in gt:
        gt["objects"] = yaml.safe_load((path.parent / gt["objects_from"]).read_text())["objects"]
    return gt


def _route_metrics(task: dict, lines: list[dict], summary: dict, mapped: dict) -> dict:
    dec = [x for x in lines if x["kind"] == "decompose"]
    plan = dec[-1].get("plan") or [] if dec else []
    seq = []
    for p in plan:
        key = "route" if p["type"] == "move" else (mapped.get(p["target"]) or f"?{p['target']}")
        if not seq or seq[-1] != key:
            seq.append(key)
    out = {"routes_planned": sum(p["type"] == "move" for p in plan),
           "routes_dispatched": sum(d["type"] == "move" for d in summary["dispatched"]),
           "route_outcomes": [h["outcome"] for h in summary["final_state"].get("history", [])
                              if h["type"] == "move"],
           "decomposed_order": seq, "routes_ok": None, "order_ok": None}
    if "gt_order" in task:
        out["routes_ok"] = out["routes_planned"] == task["gt_order"].count("route")
        out["order_ok"] = seq == task["gt_order"]
    return out


def task_metrics(run_dir: Path, task: dict, gt_objects: dict, radius: float, sim=None, thr=0.88) -> dict:
    gt_targets = task["gt_targets"]
    mapper = TargetMapper(gt_targets, task.get("aliases", {}), sim, thr)
    inst = {k: [np.asarray(gt_objects[g]["position"]) for g in v] for k, v in gt_targets.items()}
    summary = json.loads((run_dir / "summary.json").read_text())
    t_start = summary.get("task_start_stamp")
    decided = [t["name"] for t in summary["final_state"]["targets"]]
    mapped = {n: mapper(n) for n in decided}

    def near(key, pos) -> bool:
        return key is not None and any(np.linalg.norm(np.asarray(pos)[:2] - p[:2]) <= radius for p in inst[key])

    graph = json.loads((run_dir / "memory" / "final" / "graph.json").read_text())
    cent = {o["id"]: o["centroid"] for o in graph["objects"]}
    aliases = {int(k): v for k, v in graph.get("aliases", {}).items()}

    def centroid(node_id):                         # nodes may have been deduped since the goto
        while node_id in aliases:
            node_id = aliases[node_id]
        return cent.get(node_id)
    reports = summary["reports"]
    rep_ok = [near(mapped.get(r["target"]), r["position"]) for r in reports]
    gotos = [d for d in summary["dispatched"] if d["type"] == "goto"]
    scored = [d for d in gotos if mapped.get(d["target"]) is not None]   # extra targets: unscored
    goto_ok = [centroid(d["node_id"]) is not None and near(mapped[d["target"]], centroid(d["node_id"]))
               for d in scored]
    correct_keys = {mapped.get(r["target"]) for r, ok in zip(reports, rep_ok) if ok}
    present = [k for k, v in gt_targets.items() if v]
    absent = [k for k, v in gt_targets.items() if not v]
    wrong = [r for r, ok in zip(reports, rep_ok) if not ok]
    lines = [json.loads(x) for x in (run_dir / "decisions.jsonl").read_text().splitlines()]
    route = _route_metrics(task, lines, summary, mapped)
    success = all(k in correct_keys for k in present) and not wrong and route["routes_ok"] is not False
    first_ok = min((r["stamp"] for r, ok in zip(reports, rep_ok) if ok), default=None)
    q_lat = [x["query"]["vlm_latency_s"] for x in lines if x.get("query") and x["query"].get("raw")]
    acq = [json.loads(x) for x in (run_dir / "acquisition_debug.jsonl").read_text().splitlines()] \
        if (run_dir / "acquisition_debug.jsonl").exists() else []
    a_lat = [x["result"]["latency_s"] for x in acq]
    return {
        "task": task["id"], "category": task.get("category"), "fsm": summary["final_state"]["fsm"],
        "decomposed": decided, "mapped": mapped,
        "target_precision": (float(np.mean([v is not None for v in mapped.values()])) if mapped
                             else float(not gt_targets)),              # a route-only task has no targets
        "target_recall": float(np.mean([k in mapped.values() for k in gt_targets])) if gt_targets else 1.0,
        "reports": len(reports), "report_precision": float(np.mean(rep_ok)) if reports else None,
        "success": bool(success),
        "absent_reported": [r["target"] for r in reports if mapped.get(r["target"]) in absent],
        "gotos": len(gotos), "unscored_gotos": len(gotos) - len(scored),
        "false_commit_rate": float(1 - np.mean(goto_ok)) if scored else None,
        "time_to_correct_report_s": (first_ok - t_start) if (first_ok is not None and t_start) else None,
        "vlm_calls": len(q_lat) + len(acq),
        "vlm_latency_mean_s": float(np.mean(q_lat + a_lat)) if q_lat + a_lat else None,
        "vlm_latency_p95_s": float(np.percentile(q_lat + a_lat, 95)) if q_lat + a_lat else None,
        **route,
    }


def aggregate(rows: list[dict]) -> dict:
    def mean(key):
        v = [r[key] for r in rows if r[key] is not None]
        return float(np.mean(v)) if v else None
    return {"tasks": len(rows), "task_success_rate": mean("success"), "target_precision": mean("target_precision"),
            "target_recall": mean("target_recall"), "report_precision": mean("report_precision"),
            "false_commit_rate": mean("false_commit_rate"), "time_to_correct_report_s": mean("time_to_correct_report_s"),
            "vlm_calls_per_task": mean("vlm_calls"), "vlm_latency_mean_s": mean("vlm_latency_mean_s"),
            "routes_ok": mean("routes_ok"), "order_ok": mean("order_ok")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--runs", required=True, help="directory with one sub-directory per task id")
    ap.add_argument("--report-radius-m", type=float, default=1.0)
    ap.add_argument("--no-embedder", action="store_true", help="map target names by aliases only")
    args = ap.parse_args()
    gt = load_gt(args.gt)
    objs = {o["gt_id"]: o for o in gt["objects"]}
    sim, thr = None, 0.88
    if not args.no_embedder:
        from vlnav.acquisition.config import load_acquisition_config
        from vlnav.runtime.builder import build_matcher
        m = build_matcher(load_acquisition_config())
        sim, thr = m.sim, m.threshold
    rows = [task_metrics(Path(args.runs) / t["id"], t, objs, args.report_radius_m, sim, thr) for t in gt["tasks"]
            if (Path(args.runs) / t["id"] / "summary.json").exists()]
    out = {"per_task": rows, "aggregate": aggregate(rows)}
    (Path(args.runs) / "metrics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
