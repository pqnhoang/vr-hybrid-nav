"""Open-loop planner evaluation in run mode A (planner spec §9, P7): one bag replay per GT task,
then metrics. Models (detector, VLM, text encoder) are loaded once and reused across tasks.

  python eval/planner/run_openloop.py --bag-config configs/bags/vr_h3_2_calib_1.yaml \
      --run-config configs/runs/vr_h3_2_system.yaml --gt eval/planner/gt_vr_h3_2.yaml --out runs/openloop_vr_h3_2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import aggregate, task_metrics  # noqa: E402

from vlnav.common.config import load_yaml  # noqa: E402
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
    gt = yaml.safe_load(Path(args.gt).read_text())
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
              f"reports={m['reports']} report_prec={m['report_precision']} vlm_calls={m['vlm_calls']}", flush=True)
    res = {"per_task": rows, "aggregate": aggregate(rows)}
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res["aggregate"], indent=2))


if __name__ == "__main__":
    main()
