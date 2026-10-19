"""M0 model check for the move subtask (TASK_move_subtask.md §12 M0, settles [VERIFY] V1).

For each case of a cases file, plays the bag segment, keeps a motion-sampled history (§6.2) and
queries the DualVLN S2 server every --every-s seconds. Each reply is saved raw (jsonl) and drawn on
the camera image under the coordinate hypotheses, together with the robot's recorded future path
(next --future-s seconds) projected onto the floor: the hypothesis whose dots fall on that path is
the right one. Images go to the model through the virtual Habitat view (`--view habitat`, view.py) or
unchanged (`--view full`). A `↓` reply is followed by one look-down query on the same frame
(`look_down_mode: same_frame`) to see what S2 does next.

    PYTHONPATH=. $PY scripts/check_dualvln_s2.py --cases eval/move/cases_vr_h3_2.yaml --out runs/m0
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from vlnav.common.models import Pose
from vlnav.io.sources import BagSource, load_bag_config, resolve
from vlnav.move.config import load_move_config
from vlnav.move.history import FrameBuffer, resize_rgb
from vlnav.move.reply import parse_s2_text
from vlnav.move.s2_client import HttpS2Client, S2Request
from vlnav.move.view import ModelView, make_view

COLORS = {"view": (0, 0, 255), "view_from_384": (0, 200, 0), "processed": (255, 0, 0)}     # BGR


def project(K, T_map_cam, pts_map) -> np.ndarray:
    pc = (np.asarray(pts_map, float) - T_map_cam[:3, 3]) @ T_map_cam[:3, :3]
    pc = pc[pc[:, 2] > 0.2]
    return np.stack([pc[:, 0] / pc[:, 2] * K[0, 0] + K[0, 2], pc[:, 1] / pc[:, 2] * K[1, 1] + K[1, 2]], 1)


def hypotheses(xy, res: dict, view: ModelView, size) -> dict[str, tuple[float, float]]:
    """Camera pixel of a reply under each hypothesis: pixel in the view image sent (`view`), in the
    384x384 resize of it (`view_from_384`), or in the processor's smart-resized grid (`processed`)."""
    pw, ph = res["processed_size"]
    return {"view": view.to_camera(xy),
            "view_from_384": view.to_camera((xy[0] * view.out_w / size[0], xy[1] * view.out_h / size[1])),
            "processed": view.to_camera((xy[0] * view.out_w / pw, xy[1] * view.out_h / ph))}


def draw(frame, future_xyz, reply, hyp, text_lines) -> np.ndarray:
    img = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR).copy()
    uv = project(frame.K, frame.T_map_cam, future_xyz) if len(future_xyz) else np.empty((0, 2))
    for a, b in zip(uv[:-1], uv[1:]):
        cv2.line(img, tuple(int(v) for v in a), tuple(int(v) for v in b), (0, 255, 255), 4)
    for name, (u, v) in hyp.items():
        cv2.circle(img, (int(u), int(v)), 16, COLORS[name], -1)
        cv2.putText(img, name, (int(u) + 20, int(v)), 0, 1.0, COLORS[name], 2)
    y = 40
    for line in text_lines:
        cv2.putText(img, line, (20, y), 0, 1.0, (0, 0, 0), 5)
        cv2.putText(img, line, (20, y), 0, 1.0, (255, 255, 255), 2)
        y += 40
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=None, help="comma list of case ids")
    ap.add_argument("--every-s", type=float, default=2.0)
    ap.add_argument("--future-s", type=float, default=10.0)
    ap.add_argument("--view", choices=["habitat", "full"], default="habitat")
    args = ap.parse_args()

    cfg = load_move_config()
    spec = yaml.safe_load(resolve(args.cases).read_text())
    out = resolve(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    client = HttpS2Client(cfg.s2.base_url, cfg.s2.timeout_s)
    health = client.health()
    print("server", health, flush=True)
    size = (cfg.model_input.resize_w, cfg.model_input.resize_h)
    log = (out / "replies.jsonl").open("w")
    only = set(args.only.split(",")) if args.only else None
    for case in spec["cases"]:
        if only and case["id"] not in only:
            continue
        src = BagSource(load_bag_config(resolve(spec["bag_config"]),
                                        {"start_offset_s": case["start_offset_s"],
                                         "end_offset_s": case["end_offset_s"] + args.future_s}))
        frames = list(src)
        t0 = frames[0].stamp
        poses = np.array([[f.stamp, *f.T_map_base[:3, 3]] for f in frames])
        buf = FrameBuffer(cfg.history.step_m, cfg.history.step_rad, size, cfg.model_input.num_history)
        last_q = -np.inf
        view = make_view(frames[0].K, (frames[0].rgb.shape[1], frames[0].rgb.shape[0]), args.view)
        print(f"view {view}", flush=True)
        for f in frames:
            if f.stamp - t0 > case["end_offset_s"] - case["start_offset_s"]:
                break
            pose = Pose.from_matrix(f.T_map_base, f.stamp)
            vimg = view.image(f.rgb)
            buf.add(f.stamp, vimg, pose.position[:2], pose.yaw())
            if f.stamp - last_q < args.every_s:
                continue
            last_q = f.stamp
            hist = buf.history(f.stamp)
            req = S2Request(instruction=case["instruction"], image=vimg, history=[h.rgb for h in hist])
            queries = [("main", req)]
            k = 0
            while k < len(queries):
                tag, rq = queries[k]
                k += 1
                t_req = time.perf_counter()
                res = client.query(rq).model_dump()
                res["wall_ms"] = round(1e3 * (time.perf_counter() - t_req), 1)
                reply = parse_s2_text(res["text"])
                fut = poses[(poses[:, 0] >= f.stamp) & (poses[:, 0] <= f.stamp + args.future_s), 1:]
                rec = {"case": case["id"], "t_rel": round(f.stamp - t0, 2), "stamp": f.stamp, "tag": tag,
                       **res, "reply": reply.model_dump(exclude={"text"})}
                hyp = {}
                if reply.kind == "pixel":
                    hyp = hypotheses(reply.pixel_xy, res, view, size)
                    rec["hypotheses_uv"] = {n: [round(u, 1), round(v, 1)] for n, (u, v) in hyp.items()}
                    uvf = project(f.K, f.T_map_cam, fut)
                    rec["future_dist_px"] = {n: round(float(np.min(np.hypot(*(uvf - np.array(p)).T))), 1)
                                             if len(uvf) else None for n, p in hyp.items()}
                img = draw(f, fut, reply, hyp, [f"{case['id']} t={f.stamp - t0:.1f}s {tag} hist={len(rq.history)}",
                                                case["instruction"][:110], f"S2: {res['text']!r}"])
                name = f"{case['id']}_{f.stamp - t0:05.1f}_{tag}.jpg"
                cv2.imwrite(str(out / "images" / name), cv2.resize(img, (960, 540)))
                rec["image"] = name
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(json.dumps({k2: rec[k2] for k2 in ("case", "t_rel", "tag", "text", "latency_ms", "wall_ms")}
                                 | {"future_dist_px": rec.get("future_dist_px")}), flush=True)
                if reply.kind == "actions" and reply.actions[0] == "look_down" and tag == "main":
                    queries.append(("look_down", S2Request(instruction=rq.instruction, image=rq.image,
                                                           history=rq.history, look_down=True,
                                                           prev_output=res["text"], look_down_image=vimg)))
    log.close()


if __name__ == "__main__":
    main()
