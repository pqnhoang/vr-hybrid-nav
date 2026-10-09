"""Re-run ONE decomposition / replanning offline from its debug folder, with the current prompt
templates and config — iterate on prompts without replaying the bag.

  python scripts/replay_planner_query.py runs/system_vr_h3_2_oracle/planner_debug/002_replan
  python scripts/replay_planner_query.py <dump> --planning-images 0 --show-prompt
  python scripts/replay_planner_query.py <dump> --with-acquisition      # also verify the chosen goto

The memory the VLM saw is loaded from <dump>/memory (exact snapshot); the planner input (targets,
history, last event) from <dump>/input.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from vlnav.acquisition.acquirer import TargetAcquirer
from vlnav.acquisition.config import load_acquisition_config
from vlnav.acquisition.vlm_client import load_vlm_config, make_vlm_client
from vlnav.memory import ImageStore
from vlnav.memory.checkpoint import load_checkpoint
from vlnav.planner.config import load_planner_config
from vlnav.planner.decomposer import Decomposer, Replanner
from vlnav.planner.serializer import GraphSerializer
from vlnav.planner.state import PlannerState, TargetEntry
from vlnav.planner.validation import validate_decomposition, validate_replan
from vlnav.runtime.builder import build_matcher


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump", help="planner_debug/NNN_<kind> folder")
    ap.add_argument("--planning-images", type=int, default=None)
    ap.add_argument("--max-graph-tokens", type=int, default=None)
    ap.add_argument("--instruction", default=None, help="try a different instruction")
    ap.add_argument("--show-prompt", action="store_true")
    ap.add_argument("--with-acquisition", action="store_true")
    ap.add_argument("--vlm-backend", choices=["local", "openai"], default=None)
    args = ap.parse_args()
    d = Path(args.dump)
    inp = json.loads((d / "input.json").read_text())
    old = json.loads((d / "result.json").read_text())
    image_root = (d / "memory" / "image_root.txt").read_text().strip()
    store = ImageStore(image_root, 90, 64)
    pover = {k: v for k, v in (("planning_images", args.planning_images),
                               ("max_graph_tokens", args.max_graph_tokens)) if v is not None}
    pcfg = load_planner_config(overrides=pover)
    acfg = load_acquisition_config(overrides={"logging": {"debug_jsonl": None}})
    matcher = build_matcher(acfg)
    mem = load_checkpoint(d / "memory", image_root=image_root, max_neighbors=acfg.context.max_neighbors,
                          image_store=store, label_sim=None)
    vlm = make_vlm_client(load_vlm_config(overrides={"cache_dir": None,
                                                     **({"backend": args.vlm_backend} if args.vlm_backend else {})}))
    ser = GraphSerializer(pcfg, matcher)
    instruction = args.instruction or inp["instruction"]
    t0 = time.time()
    if inp["kind"] == "decompose":
        q = Decomposer(pcfg, vlm, ser).decompose(instruction, mem)
        plan = validate_decomposition(q.data, mem) if q.data is not None else None
    else:
        state = PlannerState(task_id=inp["task_id"], instruction=instruction, fsm="REPLANNING",
                             targets=[TargetEntry(**t) for t in inp["targets"]], current=None,
                             history=inp["history"], started_t=0.0)
        q = Replanner(pcfg, vlm, ser).replan(state, inp["last_event"], mem)
        plan = validate_replan(q.data, state, mem) if q.data is not None else None
    if args.show_prompt:
        print("== prompt\n" + q.prompt_text)
    print(f"== images sent: {q.image_ids}")
    print(f"== logged reply (original run):\n{(old.get('parsed') or {}).get('reasoning', '')}\n"
          f"   subtasks {(old.get('validated') or {}).get('subtasks')}")
    for k, raw in enumerate(q.raws):
        print(f"== new reply {k} ({time.time() - t0:.1f} s):\n{raw}")
    if plan is None:
        print(f"== no valid plan: {q.error}")
        return
    print(f"== validated: targets {[(t.name, t.status, t.detector_prompts) for t in plan.targets]}")
    print(f"   subtasks {[vars(s) for s in plan.subtasks]}\n   fixes {plan.fixes}  task_done {plan.task_done}")
    if args.with_acquisition:
        first = next((s for s in plan.subtasks if s.type == "goto"), None)
        if first is None:
            print("== no goto subtask to verify")
            return
        a = TargetAcquirer(acfg, matcher, vlm).acquire(instruction, first.target, mem,
                                                       force_include=[first.node_id])
        print(f"== acquisition: {a.status} node={a.node_id} candidates={a.candidates} conf={a.vlm_confidence}\n"
              f"   {a.vlm_reason}")


if __name__ == "__main__":
    main()
