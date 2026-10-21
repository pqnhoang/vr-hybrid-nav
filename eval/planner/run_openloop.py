"""Open-loop planner evaluation in run mode A (planner spec §9, P7): one bag replay per GT task,
then metrics. Models (detector, VLM, text encoder) are loaded once and reused across tasks.

  python eval/planner/run_openloop.py --bag-config configs/bags/vr_h3_2_calib_1.yaml \
      --run-config configs/runs/vr_h3_2_system.yaml --gt eval/planner/gt_vr_h3_2.yaml --out runs/openloop_vr_h3_2

Move subtasks (TASK_move_subtask.md M5): --enable-move with eval/planner/gt_vr_h3_2_move.yaml (needs the
DualVLN S2 server, scripts/serve_dualvln_s2.py).

Code-as-policy agents (TASK_cap_mas.md): --agent replaces the NeSy planner by ProgramRuntime
(configs/agent.yaml; the NeSy planner stays as the fallback). Rounds go to <out>/<task>/rounds.jsonl.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import aggregate, load_gt, task_metrics  # noqa: E402

from vlnav.common.config import deep_merge, load_yaml  # noqa: E402
from vlnav.io.sources import resolve  # noqa: E402
from vlnav.memory.checkpoint import save_checkpoint  # noqa: E402
from vlnav.runtime.builder import build_system  # noqa: E402
from vlnav.runtime.system_runtime import TaskSpec  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag-config", required=True)
    ap.add_argument("--run-config", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", nargs="*", default=None, help="subset of task ids")
    ap.add_argument("--report-radius-m", type=float, default=1.0)
    ap.add_argument("--no-planning-images", action="store_true")
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--rank-strategy", default=None)
    ap.add_argument("--enable-move", action="store_true", help="move subtasks via DualVLN (move.enabled)")
    ap.add_argument("--s2-url", default=None, help="DualVLN S2 server (default: move.s2.base_url)")
    ap.add_argument("--s2-cache-dir", default=None, help="S2 reply cache (reproducible replays)")
    ap.add_argument("--agent", action="store_true", help="code-as-policy agents (agent.enabled)")
    args = ap.parse_args()
    rc = load_yaml(resolve(args.run_config))
    if args.no_planning_images:
        rc.setdefault("planner", {})["planning_images"] = 0
    if args.k or args.rank_strategy:
        rc.setdefault("acquisition", {}).setdefault("symbolic", {})
        if args.k:
            rc["acquisition"]["symbolic"]["k"] = args.k
        if args.rank_strategy:
            rc["acquisition"]["symbolic"]["rank_strategy"] = args.rank_strategy
    if args.enable_move:
        s2 = {"blocking": True, **({"base_url": args.s2_url} if args.s2_url else {}),
              **({"cache_dir": args.s2_cache_dir} if args.s2_cache_dir else {})}
        rc["move"] = deep_merge(rc.get("move", {}), {"enabled": True, "s2": s2})
    if args.agent:
        rc["agent"] = deep_merge(rc.get("agent", {}), {"enabled": True})
    gt = load_gt(args.gt)
    objs = {o["gt_id"]: o for o in gt["objects"]}
    out = resolve(args.out)
    shared = {}
    rows = []
    for task in gt["tasks"]:
        if args.tasks and task["id"] not in args.tasks:
            continue
        run_dir = out / task["id"]
        run_dir.mkdir(parents=True, exist_ok=True)
        s = build_system(args.bag_config, rc, run_dir, **shared)
        shared = {"vlm": s.vlm, "matcher": s.matcher, "detector": s.perception.detector}
        t0 = time.time()
        log = s.runtime().run([TaskSpec(task["instruction"], task.get("start_t", 0.0))])
        s.planner.close()
        if s.move is not None:
            s.move.close()
        s.memory.wait_labels(s.memory.cfg.labeling.timeout_s)
        s.memory.close()
        save_checkpoint(s.memory.view(), run_dir / "memory" / "final", s.grid)
        st = s.planner.state()
        summary = {"task": task["id"], "wall_s": time.time() - t0, "task_start_stamp": st.started_t,
                   "final_state": {"fsm": st.fsm, "reason": st.done_reason,
                                   "targets": [t.model_dump() for t in st.targets], "history": st.history},
                   "dispatched": [{"t": t, **d.model_dump(mode="json")} for t, d in log.dispatched],
                   "reports": [r.model_dump() for r in log.reports]}
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        m = task_metrics(run_dir, task, objs, args.report_radius_m, s.matcher.sim, s.matcher.threshold)
        rows.append(m)
        print(f"[{task['id']}] {st.fsm} success={m['success']} targets={m['mapped']} "
              f"reports={m['reports']} report_prec={m['report_precision']} vlm_calls={m['vlm_calls']}"
              + (f" order={m['decomposed_order']} routes={m['route_outcomes']}" if args.enable_move else ""),
              flush=True)
    res = {"per_task": rows, "aggregate": aggregate(rows)}
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res["aggregate"], indent=2))


if __name__ == "__main__":
    main()
