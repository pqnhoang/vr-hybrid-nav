"""Live / recorded visualization of the core pipeline with Rerun (rerun.io). ROS-free.

Sinks: `spawn` (native viewer on this machine), `serve` (web viewer for a browser on another
machine), `save` (.rrd file, open later with `rerun file.rrd`); `serve`/`spawn` can be combined
with `save`. Timelines: `bag_time` (s since start) and `frame`.

Entity layout:
  world/lidar, world/map/chunk_*      LiDAR window (per frame) and accumulated map (grows)
  world/trajectory, world/robot       path and current LiDAR/base pose
  world/camera/image                  pinhole + image with masks; boxes; memory nodes projected
  world/observations                  this frame's 3D observations (centroids + points)
  world/memory/{objects,tentative,best_view_rays,points,updated}
  metrics/{timing,memory,perception}/*, events, memory/latest_best_view
  exploration/view (+ /frontiers, /instances)   top-down exploration debug image; hover = scores
  exploration/log, world/exploration/*          goal changes / events; frontier points, goal, path in 3D
"""
from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from pydantic import BaseModel, ConfigDict

from vlnav.common.config import CONFIG_DIR, deep_merge, load_yaml
from vlnav.common.models import Frame, FrameResult
from vlnav.memory.snapshot import MemorySnapshot
from vlnav.memory.types import MemoryEvent
from vlnav.memory.viz import label_color

STATUS_COLORS = {"lifted": None, "no_3d": (255, 60, 60), "on_dynamic": (150, 150, 150),
                 "dynamic": (255, 255, 255)}
EVENT_COLORS = {"node_created": (40, 220, 40), "node_merged": (255, 200, 0),
                "best_view_updated": (0, 200, 255)}


class VizConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    image_width: int
    jpeg_quality: int
    mask_alpha: float
    lidar_points_per_frame: int
    map_chunk_every_s: float
    map_chunk_points: int
    map_height_band: tuple[float, float]
    node_points_every_frames: int
    node_points_max: int
    project_nodes_max_range_m: float
    map_every_frames: int
    map_max_px: int


def load_viz_config(path: str | Path = CONFIG_DIR / "viz.yaml", overrides=None) -> VizConfig:
    return VizConfig.model_validate(deep_merge(load_yaml(path), overrides))


def blueprint(memory: bool = True, planner: bool = True, exploration: bool = False) -> rrb.Blueprint:
    """Only the views the run logs: `memory` (memory size, latest best view, occupancy + rooms,
    memory events), `planner` (task, planner log), `exploration` (top-down view as the first, active
    tab + its log)."""
    tabs = [rrb.TimeSeriesView(origin="metrics/timing", name="timing (ms)"),
            rrb.TimeSeriesView(origin="metrics/perception", name="detections")]
    logs = []
    if memory:
        tabs = [rrb.TimeSeriesView(origin="metrics/memory", name="memory size"), *tabs,
                rrb.Spatial2DView(origin="memory/latest_best_view", name="latest best view"),
                rrb.Spatial2DView(origin="map", name="occupancy + rooms")]
        logs.append(rrb.TextLogView(origin="events", name="memory events"))
    if planner:
        logs = [rrb.TextDocumentView(origin="task", name="task"), rrb.TextLogView(origin="planner", name="planner"),
                *logs]
    if exploration:
        tabs.insert(0, rrb.Spatial2DView(origin="exploration/view", name="exploration (top-down)"))
        logs.insert(0, rrb.TextLogView(origin="exploration/log", name="exploration"))
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="world", name="3D: map, memory, camera",
                              background=[24, 26, 30],
                              eye_controls=rrb.EyeControls3D(kind=rrb.Eye3DKind.Orbital,
                                                             position=[-7.0, -7.0, 6.0],
                                                             look_target=[0.0, 0.0, 0.0],
                                                             tracking_entity="world/robot")),
            rrb.Vertical(
                rrb.Spatial2DView(origin="world/camera/image", name="camera: detections + memory"),
                rrb.Tabs(*tabs, active_tab=0),
                *([rrb.Tabs(*logs, active_tab=0)] if logs else []),
                row_shares=([2, 4, 2] if exploration else [3, 2, 2])[:2 + bool(logs)],
            ),
            column_shares=[3, 2],
        ),
        rrb.TimePanel(state="expanded", timeline="bag_time"),
        collapse_panels=True,
    )


def lan_ip() -> str:
    """This machine's LAN address (for the web viewer URL printed to the user)."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sk:
        try:
            sk.connect(("10.255.255.255", 1))
            return sk.getsockname()[0]
        except OSError:
            return "127.0.0.1"


def _port_open(port: int) -> bool:
    with socket.socket() as sk:
        sk.settimeout(0.5)
        return sk.connect_ex(("127.0.0.1", port)) == 0


def ensure_web_server(grpc_port: int, web_port: int, memory_limit: str, log_path: Path,
                      timeout_s: float = 30.0) -> None:
    """Run `rerun --serve-web` as a detached process (reused if already up). It buffers the
    stream for browsers that join late and keeps serving after the replay ends."""
    if _port_open(grpc_port):
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.Popen([sys.executable, "-m", "rerun", "--serve-web", "--bind", "0.0.0.0",
                      "--port", str(grpc_port), "--web-viewer-port", str(web_port),
                      "--server-memory-limit", memory_limit],
                     stdout=open(log_path, "w"), stderr=subprocess.STDOUT, start_new_session=True)
    t0 = time.time()
    while not _port_open(grpc_port):
        if time.time() - t0 > timeout_s:
            raise RuntimeError(f"rerun server did not start, see {log_path}")
        time.sleep(0.2)


def start(app_id: str, sinks: list[str], save_path: str | None, web_port: int, grpc_port: int,
          host: str, server_memory_limit: str = "4GB", recording_name: str | None = None,
          memory: bool = True, planner: bool = True, exploration: bool = False) -> str | None:
    """Start the recording with the requested sinks; returns the web viewer URL when serving.
    `recording_name` (e.g. the task instruction) tells recordings apart in one viewer."""
    rr.init(app_id)
    url = None
    targets = []
    if "serve" in sinks:
        ensure_web_server(grpc_port, web_port, server_memory_limit,
                          Path(save_path or ".").parent / "rerun_server.log")
        url = f"http://{host}:{web_port}/?url=rerun+http://{host}:{grpc_port}/proxy"
        targets.append(rr.GrpcSink(f"rerun+http://127.0.0.1:{grpc_port}/proxy"))
    if "spawn" in sinks:
        rr.spawn(port=grpc_port, connect=False)
        targets.append(rr.GrpcSink(f"rerun+http://127.0.0.1:{grpc_port}/proxy"))
    if "save" in sinks:
        if not save_path:
            raise ValueError("save sink needs a path")
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        targets.append(rr.FileSink(save_path))
    bp = blueprint(memory, planner, exploration)
    rr.set_sinks(*targets, default_blueprint=bp)
    if recording_name:
        rr.send_recording_name(recording_name)
    rr.send_blueprint(bp)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    return url


def _jpeg(rgb: np.ndarray, quality: int) -> rr.EncodedImage:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, quality])
    return rr.EncodedImage(contents=buf.tobytes(), media_type="image/jpeg")


def _turbo(v: np.ndarray) -> np.ndarray:
    """Values in [0, 1] -> RGB (turbo: blue low, red high)."""
    v = np.clip(np.asarray(v, dtype=float), 0, 1)
    return cv2.applyColorMap((255 * v).astype(np.uint8)[:, None], cv2.COLORMAP_TURBO)[:, 0, ::-1]


def _height_colors(z: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(z, 2), np.percentile(z, 98)
    return _turbo((z - lo) / max(hi - lo, 1e-6))


class RerunLogger:
    def __init__(self, cfg: VizConfig, seed: int = 0):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.t0: float | None = None
        self.traj: list[np.ndarray] = []
        self.map_buf: list[np.ndarray] = []
        self.map_chunk = 0
        self.last_chunk_t = -np.inf
        self.last_version = -1
        self.reported: list = []
        self._task_fsm: str | None = None
        self._rooms_key: tuple = ()
        self._expl_goal: tuple | None = None

    def _sub(self, pts: np.ndarray, n: int) -> np.ndarray:
        return pts if len(pts) <= n else pts[self.rng.choice(len(pts), n, replace=False)]

    # ------------------------------------------------------------------------------------- frame
    def log_frame(self, k: int, frame: Frame, res: FrameResult, status: list[str],
                  snap: MemorySnapshot | None, events: list[MemoryEvent], timings: dict[str, float]) -> None:
        """`snap=None`: no memory in this run (exploration-only shadow mode)."""
        self.t0 = frame.stamp if self.t0 is None else self.t0
        t = frame.stamp - self.t0
        rr.set_time("frame", sequence=k)
        rr.set_time("bag_time", duration=t)
        self._log_sensors(frame)
        self._log_camera(frame, res, status, snap)
        self._log_observations(res)
        if snap is not None:
            self._log_memory(k, snap, events)
        for name, v in timings.items():
            rr.log(f"metrics/timing/{name}", rr.Scalars(v))
        rr.log("metrics/perception/detections", rr.Scalars(len(res.detections)))
        rr.log("metrics/perception/observations", rr.Scalars(len(res.observations)))

    # -------------------------------------------------------------------------- grid / rooms
    def log_map(self, k: int, grid, snap: MemorySnapshot) -> None:
        """Occupancy grid + room fills/labels as a 2D image (north up); room outlines in 3D."""
        rooms = snap.rooms()
        rooms_key = tuple((r.id, r.label, round(r.area_m2, 1)) for r in rooms)
        if rooms_key != self._rooms_key:
            self._rooms_key = rooms_key
            strips, labels, colors = [], [], []
            for r in rooms:
                poly = snap.room_polygons.get(r.id, [])
                if len(poly) < 3:
                    continue
                z = grid.ground_height(r.centroid_xy)
                z = 0.0 if not np.isfinite(z) else z + 0.05
                strips.append([(x, y, z) for x, y in [*poly, poly[0]]])
                labels.append(f"R{r.id} {r.label} ({r.area_m2:.0f} m2)")
                colors.append(label_color(f"room{r.id}"))
            rr.log("world/rooms", rr.LineStrips3D(strips, colors=colors, radii=0.04, labels=labels)
                   if strips else rr.Clear(recursive=False))
        if k % self.cfg.map_every_frames or 0 in grid.shape:
            return
        north = grid.as_array()[::-1]                 # row 0 = max y (north up)
        img = np.full((*north.shape, 3), 70, np.uint8)
        img[north == 0] = 235
        img[north == 1] = 20
        ox, oy = grid.origin_xy
        res, ny = grid.resolution, north.shape[0]

        def px(xy):                                    # world xy -> pixel (x, y) of `img`
            xy = np.asarray(xy, dtype=float)
            return np.stack([(xy[..., 0] - ox) / res - 0.5, ny - 0.5 - (xy[..., 1] - oy) / res], -1)

        overlay = img.copy()
        cents, names = [], []
        for r in rooms:
            poly = snap.room_polygons.get(r.id, [])
            if len(poly) >= 3:
                cv2.fillPoly(overlay, [np.round(px(poly)).astype(np.int32)], label_color(f"room{r.id}"))
            cents.append(px(r.centroid_xy))
            names.append(f"R{r.id} {r.label}")
        tinted = cv2.addWeighted(overlay, 0.45, img, 0.55, 0)
        img = np.where((north == 0)[..., None], tinted, img)          # tint free cells only
        s = min(1.0, self.cfg.map_max_px / max(img.shape[:2]))
        if s < 1.0:
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
        rr.log("map/occupancy", rr.Image(img))
        rr.log("map/rooms", rr.Points2D(np.asarray(cents).reshape(-1, 2) * s, labels=names, radii=4,
                                         colors=[label_color(f"room{r.id}") for r in rooms])
               if cents else rr.Clear(recursive=False))
        if self.traj:
            rr.log("map/robot", rr.Points2D(px(self.traj[-1][:2])[None] * s, radii=6, colors=(80, 140, 255)))

    # ----------------------------------------------------------------------------- exploration
    def log_exploration(self, img: np.ndarray, view, snap: dict, grid) -> None:
        """Top-down debug image (exploration/viz.py) + hover data for frontier / instance points;
        frontier points, goal and path in 3D at floor height; goal changes and events as text."""
        rr.log("exploration/view", _jpeg(img, 90))
        pts = snap["points"]
        if pts:
            labels = []
            for p in pts:
                sc = p["score"]
                labels.append(f"F{p['id']}  " + ("not selectable" + (" (blacklisted)" if p["blacklisted"] else "")
                                                if sc is None else
                                                f"S_NeSy {sc['s_nesy']:.3f} = S_dist {sc['s_dist']:.2f} (d {sc['dist']:.1f} m)"
                                                f" + VL {sc['s_vl']:.2f} x unk {sc['s_unknown']:.2f}"
                                                + (f" + room {sc['room_bonus']:.2f}" if sc['room_bonus'] else "")))
            rr.log("exploration/view/frontiers", rr.Points2D(view.px([p["xy"] for p in pts]), radii=7,
                                                              colors=(255, 255, 255, 40), labels=labels,
                                                              show_labels=False))
        else:
            rr.log("exploration/view/frontiers", rr.Clear(recursive=False))
        inst = snap["instances"]
        rr.log("exploration/view/instances",
               rr.Points2D(view.px([q["xy"] for q in inst]), radii=8, colors=(255, 255, 255, 40), show_labels=False,
                           labels=[f"I{q['id']} {q['label']} rank {q['rank']:.2f} conf {q['score']:.2f} {q['status']}"
                                   for q in inst]) if inst else rr.Clear(recursive=False))
        robot = snap.get("robot")
        z = grid.ground_height(robot[:2]) if robot else 0.0
        z = (z if np.isfinite(z) else 0.0) + 0.1
        sel = [p for p in pts if p["score"] is not None]
        smax = max([p["score"]["s_nesy"] for p in sel] + [1e-6])
        rr.log("world/exploration/frontiers", rr.Points3D(
            [(*p["xy"], z) for p in sel], radii=0.12, labels=[f"F{p['id']}" for p in sel],
            colors=_turbo([p["score"]["s_nesy"] / smax for p in sel]))
               if sel else rr.Clear(recursive=False))
        path, goal = snap.get("path"), snap.get("goal")
        rr.log("world/exploration/path", rr.LineStrips3D([[(*q, z) for q in path]], colors=(220, 0, 170), radii=0.04)
               if path else rr.Clear(recursive=False))
        rr.log("world/exploration/goal", rr.Points3D([(*goal["xy"], z)], radii=0.25, colors=(220, 0, 170),
                                                      labels=[f"goal {goal['kind']} #{goal['id']}"])
               if goal else rr.Clear(recursive=False))
        key = (snap.get("goal_kind"), goal["id"] if goal else None)
        if key != self._expl_goal:
            self._expl_goal = key
            rr.log("exploration/log", rr.TextLog(f"goal -> {key[0]}" + (f" #{key[1]}" if key[1] is not None else "")))
        for e in snap.get("events", []):
            rr.log("exploration/log", rr.TextLog(f"EVENT {e['type']} {e['payload']}", level="WARN"))
        for n in snap.get("notes", []):
            rr.log("exploration/log", rr.TextLog(n, level="DEBUG"))

    # ----------------------------------------------------------------------------- planner
    def log_task(self, state) -> None:
        """Markdown panel: instruction, FSM state and one row per target (updated over time)."""
        if state is None or state.task_id < 0:
            return
        found = {r.target: r for r in self.reported}
        rows = "\n".join(
            f"| {t.name} | {t.status} | {', '.join(t.detector_prompts)} | "
            f"{('#' + str(found[t.name].node_id) + ' ' + found[t.name].label) if t.name in found else ''} | {t.notes} |"
            for t in state.targets)
        md = (f"## Task\n> {state.instruction}\n\n**State:** `{state.fsm}`"
              + (f" — {state.done_reason}" if state.done_reason else "")
              + (f"\n\n**Current subtask:** {state.current.type} *{state.current.target}*"
                 + (f" → #{state.current.node_id}" if state.current.node_id is not None else "")
                 if state.current else "")
              + "\n\n| target | status | detector prompts | found | notes |\n|---|---|---|---|---|\n" + rows)
        rr.log("task", rr.TextDocument(md, media_type=rr.MediaType.MARKDOWN))

    def log_planner(self, state, dispatched, events, reports, memory_view, decisions=()) -> None:
        """Planner FSM + VLM decisions + subtasks + events as text; goal and reports in 3D."""
        for dec in decisions:
            plan = ", ".join(f"{p['type']} {p['target']!r}" + (f" #{p['node_id']}" if p.get('node_id') is not None else "")
                             for p in dec.get("plan", []))
            lines = [f"VLM {dec['kind'].upper()} ({dec.get('vlm_latency_s')} s): {dec.get('reasoning', '')}",
                     f"  plan: {plan or '-'}"]
            if dec.get("fixes"):
                lines.append(f"  validation fixes: {dec['fixes']}")
            acq = dec.get("acquisition")
            if acq:
                lines.append(f"  acquisition {acq['status']} node={acq['node_id']} candidates={acq['candidates']} "
                             f"conf={acq['vlm_confidence']}: {acq.get('vlm_reason') or ''}")
            if dec.get("done") or dec.get("failed"):
                lines.append(f"  -> {'FAILED' if dec.get('failed') else 'DONE'}: {dec.get('reason')}")
            if dec.get("debug_dir"):
                lines.append(f"  debug: {dec['debug_dir']}")
            rr.log("planner", rr.TextLog("\n".join(lines), level="INFO"))
        for e in events:
            rr.log("planner", rr.TextLog(f"event {e.type} {e.payload}", level="INFO"))
        for st in dispatched:
            msg = f"dispatch {st.type} '{st.target}'" + (f" -> node #{st.node_id}" if st.node_id is not None else "") \
                + (f" prompts {st.detector_prompts}" if st.detector_prompts else "")
            rr.log("planner", rr.TextLog(msg, level="WARN"))
            if st.type == "goto" and st.goal_pose is not None:
                rr.log("world/planner/goal", rr.Points3D([st.goal_pose.position], radii=0.35, colors=(255, 0, 255),
                                                         labels=[f"goto {st.target} #{st.node_id}"]))
            else:
                rr.log("world/planner/goal", rr.Clear(recursive=False))
        for r in reports:
            rr.log("planner", rr.TextLog(f"REPORT {r.target}: #{r.node_id} {r.label} at "
                                         f"({r.position[0]:.1f}, {r.position[1]:.1f})", level="ERROR"))
        if reports:
            self.reported.extend(reports)
            rr.log("world/planner/reports", rr.Points3D([r.position for r in self.reported], radii=0.4,
                                                        colors=(255, 215, 0),
                                                        labels=[f"FOUND {r.target} #{r.node_id}" for r in self.reported]))
        if state is not None and (dispatched or events or reports or decisions or self._task_fsm != state.fsm):
            targets = ", ".join(f"{t.name}={t.status}" for t in state.targets)
            rr.log("planner", rr.TextLog(f"[{state.fsm}] {targets}", level="DEBUG"))
            self.log_task(state)
            self._task_fsm = state.fsm

    def _log_sensors(self, f: Frame) -> None:
        T = f.T_map_base
        rr.log("world/robot", rr.Transform3D(translation=T[:3, 3], mat3x3=T[:3, :3]),
               rr.TransformAxes3D(axis_length=0.5))
        self.traj.append(T[:3, 3].copy())
        rr.log("world/trajectory", rr.LineStrips3D([np.asarray(self.traj)], colors=[(80, 140, 255)],
                                                    radii=0.03))
        p = f.lidar_points_map
        if p is None or not len(p):
            return
        s = self._sub(p, self.cfg.lidar_points_per_frame)
        rr.log("world/lidar", rr.Points3D(s, colors=_height_colors(s[:, 2]), radii=0.02))
        z0 = f.sensor_origin_map[2]
        lo, hi = self.cfg.map_height_band
        band = p[(p[:, 2] > z0 + lo) & (p[:, 2] < z0 + hi)]
        self.map_buf.append(self._sub(band, self.cfg.map_chunk_points // 4))
        if f.stamp - self.last_chunk_t >= self.cfg.map_chunk_every_s:
            chunk = self._sub(np.concatenate(self.map_buf), self.cfg.map_chunk_points)
            rr.log(f"world/map/chunk_{self.map_chunk:05d}",
                   rr.Points3D(chunk, colors=(170, 170, 170), radii=0.025))
            self.map_chunk += 1
            self.map_buf, self.last_chunk_t = [], f.stamp

    def _log_camera(self, f: Frame, res: FrameResult, status: list[str],
                    snap: MemorySnapshot) -> None:
        h, w = f.rgb.shape[:2]
        s = self.cfg.image_width / w
        K = f.K.copy()
        K[:2] *= s
        size = (self.cfg.image_width, int(round(h * s)))
        T = f.T_map_cam
        rr.log("world/camera", rr.Transform3D(translation=T[:3, 3], mat3x3=T[:3, :3]))
        rr.log("world/camera/image", rr.Pinhole(image_from_camera=K, resolution=size,
                                                camera_xyz=rr.ViewCoordinates.RDF,
                                                image_plane_distance=1.0))
        img = cv2.resize(f.rgb, size, interpolation=cv2.INTER_AREA)
        boxes, labels, colors, dyn = [], [], [], []
        for d, st in zip(res.detections, status):
            if st == "dynamic":                      # people: thin unlabeled boxes, own entity
                dyn.append(np.asarray(d.bbox, float) * s)
                continue
            col = STATUS_COLORS.get(st) or label_color(d.label)
            if st in ("lifted", "no_3d"):
                m = cv2.resize(d.mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
                img[m] = ((1 - self.cfg.mask_alpha) * img[m] + self.cfg.mask_alpha * np.array(col)).astype(np.uint8)
            boxes.append(np.asarray(d.bbox, float) * s)
            labels.append(f"{d.label} {d.score:.2f}" + ("" if st == "lifted" else f" [{st}]"))
            colors.append(col)
        rr.log("world/camera/image/rgb", _jpeg(img, self.cfg.jpeg_quality))
        rr.log("world/camera/image/detections",
               rr.Boxes2D(array=np.asarray(boxes).reshape(-1, 4), array_format=rr.Box2DFormat.XYXY,
                          labels=labels, colors=colors) if boxes else rr.Clear(recursive=False))
        rr.log("world/camera/image/people",
               rr.Boxes2D(array=np.asarray(dyn).reshape(-1, 4), array_format=rr.Box2DFormat.XYXY,
                          colors=STATUS_COLORS["dynamic"], radii=0.5) if dyn else rr.Clear(recursive=False))
        # memory objects in front of the camera, projected into the image
        nodes = snap.all_objects() if snap is not None else []
        if nodes:
            c = np.array([n.centroid for n in nodes])
            pc = (c - T[:3, 3]) @ T[:3, :3]
            ok = (pc[:, 2] > 0.3) & (pc[:, 2] < self.cfg.project_nodes_max_range_m)
            uv = pc[ok, :2] / pc[ok, 2:3] @ K[:2, :2].T + K[:2, 2]
            inside = (uv[:, 0] >= 0) & (uv[:, 0] < size[0]) & (uv[:, 1] >= 0) & (uv[:, 1] < size[1])
            sel = [nodes[i] for i in np.flatnonzero(ok)[inside]]
            rr.log("world/camera/image/memory_nodes",
                   rr.Points2D(uv[inside], radii=5, colors=[label_color(n.label) for n in sel],
                               labels=[f"#{n.id} {n.label}" for n in sel]))

    def _log_observations(self, res: FrameResult) -> None:
        if not res.observations:
            rr.log("world/observations", rr.Clear(recursive=True))
            return
        cols = [label_color(o.label) for o in res.observations]
        rr.log("world/observations", rr.Points3D([o.centroid_map for o in res.observations],
                                                 colors=cols, radii=0.1,
                                                 labels=[f"{o.label} {o.score:.2f}" for o in res.observations]))
        pts = np.concatenate([o.points_map for o in res.observations])
        pc = np.concatenate([np.tile(c, (len(o.points_map), 1)) for o, c in zip(res.observations, cols)])
        rr.log("world/observations/points", rr.Points3D(pts, colors=pc, radii=0.02))

    def _log_memory(self, k: int, snap: MemorySnapshot, events: list[MemoryEvent]) -> None:
        for e in events:
            payload = {k: (Path(v).name if k == "image" else round(v, 2) if isinstance(v, float) else v)
                       for k, v in e.payload.items()}
            rr.log("events", rr.TextLog(f"{e.type} {payload}",
                                        level="WARN" if e.type in ("node_pruned", "nodes_deduped") else "INFO"))
        if snap.version() == self.last_version:
            return
        self.last_version = snap.version()
        conf = snap.all_objects()
        tent = [n for n in snap.all_objects(include_tentative=True) if not snap.is_confirmed(n.id)]
        rr.log("metrics/memory/confirmed", rr.Scalars(len(conf)))
        rr.log("metrics/memory/tentative", rr.Scalars(len(tent)))
        if conf:
            rr.log("world/memory/objects", rr.Points3D(
                [n.centroid for n in conf], colors=[label_color(n.label) for n in conf], radii=0.15,
                labels=[f"#{n.id} {n.label} {n.best_score:.2f} x{n.n_obs}" for n in conf]))
            rr.log("world/memory/best_view_rays", rr.LineStrips3D(
                [[n.best_view_pose.position, n.centroid] for n in conf], colors=(200, 200, 200), radii=0.01))
        if tent:
            rr.log("world/memory/tentative", rr.Points3D(
                [n.centroid for n in tent], colors=(120, 120, 120), radii=0.07,
                labels=[f"#{n.id} {n.label}?" for n in tent]))
        else:
            rr.log("world/memory/tentative", rr.Clear(recursive=False))
        changed: dict[int, str] = {}
        best: list[int] = []
        for e in events:
            nid = e.payload.get("id", e.payload.get("kept"))
            if nid is None:
                continue
            if e.type == "best_view_updated":
                best.append(nid)
            if e.type in EVENT_COLORS and changed.get(nid) != "node_created":   # "new" wins
                changed[nid] = e.type
        upd = [(snap.get_object(i), t) for i, t in changed.items() if i in snap.objects]
        rr.log("world/memory/updated", rr.Points3D([n.centroid for n, _ in upd], radii=0.3,
                                                   colors=[EVENT_COLORS[t] for _, t in upd])
               if upd else rr.Clear(recursive=False))
        best = [i for i in best if i in snap.objects]
        if best:
            n = snap.get_object(best[-1])
            try:
                rr.log("memory/latest_best_view", rr.Image(snap.best_view_crop(n.id, 0.15, 320)))
                rr.log("memory/latest_best_view/caption", rr.TextLog(f"#{n.id} {n.label} {n.best_score:.2f}"))
            except (FileNotFoundError, RuntimeError):
                pass
        if k % self.cfg.node_points_every_frames == 0 and conf:
            pts, cols = [], []
            for n in conf:
                p = self._sub(np.asarray(n.points), self.cfg.node_points_max)
                pts.append(p)
                cols.append(np.tile(label_color(n.label), (len(p), 1)))
            rr.log("world/memory/points", rr.Points3D(np.concatenate(pts), colors=np.concatenate(cols), radii=0.025))
