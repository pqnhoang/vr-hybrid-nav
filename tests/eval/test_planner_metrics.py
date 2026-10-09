"""P7 metrics on a synthetic run directory (aliases, unscored extra targets, absent targets)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval" / "planner"))
from metrics import aggregate, task_metrics  # noqa: E402

GT = {"tc_0": {"gt_id": "tc_0", "position": [1.0, 0.0, 0.5]},
      "sofa_4": {"gt_id": "sofa_4", "position": [5.0, 5.0, 0.4]}}
TASK = {"id": "t01", "category": "implicit_multi",
        "gt_targets": {"trash can": ["tc_0"], "comfortable seat": ["sofa_4"], "umbrella": []},
        "aliases": {"comfortable seat": ["sofa"], "trash can": ["bin"]}}


def run_dir(tmp_path, reports, dispatched, targets):
    d = tmp_path / "t01"
    (d / "memory" / "final").mkdir(parents=True)
    graph = {"objects": [{"id": 0, "centroid": [1.2, 0.1, 0.5]}, {"id": 9, "centroid": [5.1, 5.0, 0.4]},
                         {"id": 7, "centroid": [9.0, 9.0, 0.0]}], "aliases": {"4": 9}}
    (d / "memory" / "final" / "graph.json").write_text(json.dumps(graph))
    (d / "decisions.jsonl").write_text(json.dumps({"kind": "decompose", "query": {"raw": ["x"], "vlm_latency_s": 2.0}}) + "\n")
    summary = {"task_start_stamp": 100.0, "final_state": {"fsm": "DONE", "targets": [{"name": n} for n in targets]},
               "reports": reports, "dispatched": dispatched}
    (d / "summary.json").write_text(json.dumps(summary))
    return d


def test_success_aliases_and_unscored_gotos(tmp_path):
    d = run_dir(tmp_path,
                reports=[{"target": "bin", "position": [1.1, 0.0, 0.5], "stamp": 110.0},
                         {"target": "sofa", "position": [5.0, 5.2, 0.4], "stamp": 130.0}],
                dispatched=[{"type": "goto", "target": "bin", "node_id": 0},
                            {"type": "goto", "target": "sofa", "node_id": 4},     # deduped into #9
                            {"type": "goto", "target": "lamp", "node_id": 7}],    # extra target
                targets=["bin", "sofa", "lamp"])
    m = task_metrics(d, TASK, GT, radius=1.0)
    assert m["success"] and m["report_precision"] == 1.0 and m["false_commit_rate"] == 0.0
    assert m["unscored_gotos"] == 1 and m["target_recall"] == 2 / 3 and m["target_precision"] == 2 / 3
    assert m["time_to_correct_report_s"] == 10.0 and m["vlm_calls"] == 1
    assert aggregate([m])["task_success_rate"] == 1.0


def test_wrong_report_and_absent_target(tmp_path):
    d = run_dir(tmp_path,
                reports=[{"target": "bin", "position": [9.0, 9.0, 0.0], "stamp": 110.0}],
                dispatched=[{"type": "goto", "target": "bin", "node_id": 7}], targets=["bin", "umbrella"])
    m = task_metrics(d, TASK, GT, radius=1.0)
    assert not m["success"] and m["report_precision"] == 0.0 and m["false_commit_rate"] == 1.0
    assert m["absent_reported"] == []
