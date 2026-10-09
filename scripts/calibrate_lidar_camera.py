"""LiDAR -> camera extrinsic refinement + gravity alignment for a bag (prerequisite of SYSTEM §4.1).

1. Prior: T_lidar_camlink from CAD (YAML) @ T_camlink_optical from the bag /tf.
2. Gravity: T_gravity_odom from the mean accelerometer direction over the trajectory.
3. Refinement: joint point-to-plane ICP of RealSense depth clouds (camera frame) against local LiDAR
   maps (LiDAR frame at the camera stamp) over many slow-rotation frames, one shared T_lidar_cam.
4. Checks: camera-LiDAR time-offset scan, LiDAR-on-image overlays (CAD vs refined).

Usage:
  python scripts/calibrate_lidar_camera.py --bag-config configs/bags/bkhn_round1.yaml \
      --cad configs/calib/bkhn_round1_cad.yaml --out configs/calib/bkhn_round1.yaml \
      --report-dir runs/calib_bkhn_round1
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import yaml
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from vlnav.io.gravity_align import T_gravity_from_up, up_direction_in_map
from vlnav.io.mcap_bag import McapBag
from vlnav.io.ros_decode import transform_to_matrix
from vlnav.io.sources import (BagSource, load_bag_config, load_calibration, matrix_to_yaml,
                              resolve, transform_points)
from vlnav.io.trajectory import Trajectory


# ---------------------------------------------------------------------------------------- prior
def camlink_to_optical(bag: McapBag, color_frame: str, optical_frame: str,
                       camlink: str) -> np.ndarray:
    edges: dict[str, tuple[str, np.ndarray]] = {}
    for m in bag.messages(["/tf"]):
        for tr in m.msg.transforms:
            edges.setdefault(tr.child_frame_id, (tr.header.frame_id, transform_to_matrix(tr.transform)))
        if optical_frame in edges and color_frame in edges:
            break
    T, frame = np.eye(4), optical_frame
    while frame != camlink:
        parent, T_pc = edges[frame]
        T, frame = T_pc @ T, parent
    return T


# --------------------------------------------------------------------------------------- gravity
def gravity(bag: McapBag, traj: Trajectory, imu_topic: str, every_nth: int,
            max_gap_s: float) -> tuple[np.ndarray, float]:
    stamps, acc = [], []
    for m in bag.messages([imu_topic], start=traj.stamps[0], end=traj.stamps[-1], every_nth=every_nth):
        h = m.msg.header
        stamps.append(h.stamp.sec + h.stamp.nanosec * 1e-9)
        a = m.msg.linear_acceleration
        acc.append([a.x, a.y, a.z])
    up = up_direction_in_map(traj, np.array(stamps), np.array(acc), np.eye(3), max_gap_s)
    tilt = float(np.degrees(np.arccos(np.clip(up[2], -1, 1))))
    return T_gravity_from_up(up), tilt


# ------------------------------------------------------------------------------------- sampling
def rotation_rate(src: BagSource, t: float, dt: float) -> float:
    a, b = src.pose(t - dt), src.pose(t + dt)
    if a is None or b is None:
        return np.inf
    return float(np.linalg.norm(Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).as_rotvec()) / (2 * dt))


def depth_cloud(depth: np.ndarray, K: np.ndarray, stride: int, zmin: float, zmax: float) -> np.ndarray:
    v, u = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
    z = depth[v, u]
    ok = (z > zmin) & (z < zmax)
    z, u, v = z[ok], u[ok], v[ok]
    return np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], axis=1)


def collect(src: BagSource, n_frames: int, max_rot_rate: float, args) -> list[dict]:
    frames = []
    for f in src:
        if rotation_rate(src, f.stamp, args.rate_dt) > max_rot_rate:
            continue
        P = depth_cloud(f.depth, f.K, args.depth_stride, args.zmin, args.zmax)
        if len(P) < args.min_depth_points or not len(f.lidar_points_map):
            continue
        T_map_lidar = src.pose(f.stamp)
        Q = transform_points(np.linalg.inv(T_map_lidar), f.lidar_points_map)
        Q = Q[np.linalg.norm(Q, axis=1) < args.lidar_crop_m]
        frames.append({"t": f.stamp, "P": P, "Q_map": f.lidar_points_map, "rgb": f.rgb, "K": f.K,
                       "T_map_lidar": T_map_lidar, "Q": Q})
    if len(frames) > n_frames:
        frames = [frames[i] for i in np.linspace(0, len(frames) - 1, n_frames).astype(int)]
    return frames


# -------------------------------------------------------------------------------------------- ICP
def target(Q: np.ndarray, voxel: float, knn: int):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Q))
    pcd = pcd.voxel_down_sample(voxel)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn))
    q = np.asarray(pcd.points)
    return q, np.asarray(pcd.normals), cKDTree(q)


def residuals(X, data, dmax):
    out = []
    for P, q, n, tree in data:
        p = transform_points(X, P)
        d, idx = tree.query(p, distance_upper_bound=dmax)
        ok = np.isfinite(d)
        out.append((p[ok], q[idx[ok]], n[idx[ok]], len(P)))
    return out


def icp(X0, data, schedule, huber):
    X = X0.copy()
    log = []
    for dmax, iters in schedule:
        for _ in range(iters):
            H, g, rs, n_in, n_all = np.zeros((6, 6)), np.zeros(6), [], 0, 0
            for p, q, n, total in residuals(X, data, dmax):
                r = ((p - q) * n).sum(1)
                w = np.minimum(1.0, huber / np.maximum(np.abs(r), 1e-12))
                J = np.hstack([np.cross(p, n), n])
                H += (J * w[:, None]).T @ J
                g += (J * w[:, None]).T @ r
                rs.append(r)
                n_in, n_all = n_in + len(p), n_all + total
            delta = -np.linalg.solve(H, g)
            dX = np.eye(4)
            dX[:3, :3] = Rotation.from_rotvec(delta[:3]).as_matrix()
            dX[:3, 3] = delta[3:]
            X = dX @ X
            r = np.concatenate(rs)
            log.append({"dmax": dmax, "median_abs_r": float(np.median(np.abs(r))),
                        "inliers": n_in / n_all, "step": float(np.linalg.norm(delta))})
    eig = np.linalg.eigvalsh(H)
    return X, log, eig


def score(X, data, dmax):
    r = np.concatenate([((p - q) * n).sum(1) for p, q, n, _ in residuals(X, data, dmax)])
    inl = sum(len(p) for p, *_ in residuals(X, data, dmax)) / sum(t for *_, t in residuals(X, data, dmax))
    return float(np.median(np.abs(r))), float(inl)


# --------------------------------------------------------------------------------------- overlays
def overlay(rgb, K, Q_lidar, T_lidar_cam, max_range):
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    pc = transform_points(np.linalg.inv(T_lidar_cam), Q_lidar)
    pc = pc[(pc[:, 2] > 0.2) & (pc[:, 2] < max_range)]
    uv = (pc[:, :2] / pc[:, 2:3]) @ K[:2, :2].T + K[:2, 2]
    ok = (uv[:, 0] >= 0) & (uv[:, 0] < img.shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < img.shape[0])
    colors = cv2.applyColorMap((255 * np.clip(pc[ok, 2] / max_range, 0, 1)).astype(np.uint8)[:, None],
                               cv2.COLORMAP_JET)[:, 0]
    for (u, v), c in zip(uv[ok].astype(int), colors):
        cv2.circle(img, (int(u), int(v)), 2, tuple(int(x) for x in c), -1)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag-config", required=True)
    ap.add_argument("--cad", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report-dir", required=True)
    ap.add_argument("--camlink", default="camera_link")
    ap.add_argument("--color-frame", default="camera_color_frame")
    ap.add_argument("--optical-frame", default="camera_color_optical_frame")
    ap.add_argument("--n-frames", type=int, default=60)
    ap.add_argument("--candidate-rate-hz", type=float, default=0.25)
    ap.add_argument("--max-rot-rate", type=float, default=0.15, help="rad/s")
    ap.add_argument("--rate-dt", type=float, default=0.25)
    ap.add_argument("--lidar-window-s", type=float, default=1.0)
    ap.add_argument("--lidar-crop-m", type=float, default=8.0)
    ap.add_argument("--depth-stride", type=int, default=8)
    ap.add_argument("--zmin", type=float, default=0.4)
    ap.add_argument("--zmax", type=float, default=4.0)
    ap.add_argument("--min-depth-points", type=int, default=2000)
    ap.add_argument("--voxel", type=float, default=0.05)
    ap.add_argument("--knn", type=int, default=20)
    ap.add_argument("--huber", type=float, default=0.05)
    ap.add_argument("--imu-every-nth", type=int, default=20)
    ap.add_argument("--time-offsets", type=str, default="-0.15:0.15:0.025")
    ap.add_argument("--n-overlays", type=int, default=6)
    args = ap.parse_args()

    out_dir = resolve(args.report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bag_cfg = load_bag_config(resolve(args.bag_config))
    bag = McapBag(resolve(bag_cfg.bag_dir), bag_cfg.typestore)
    if bag_cfg.poses.kind != "trajectory":
        raise SystemExit("calibration needs poses.kind: trajectory (a TF bag already has extrinsics)")
    traj = Trajectory.load(resolve(bag_cfg.poses.trajectory), bag_cfg.poses.trajectory_format)

    T_lidar_camlink = load_calibration(resolve(args.cad))["T_lidar_camlink"]
    T_camlink_opt = camlink_to_optical(bag, args.color_frame, args.optical_frame, args.camlink)
    X_cad = T_lidar_camlink @ T_camlink_opt
    T_g, tilt = gravity(bag, traj, bag_cfg.imu_topic, args.imu_every_nth, bag_cfg.max_pose_gap_s)
    print(f"gravity: map frame tilted {tilt:.2f} deg from vertical")

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp:
        yaml.safe_dump({"T_lidar_cam": matrix_to_yaml(X_cad), "T_gravity_odom": matrix_to_yaml(T_g)}, tmp)
    src = BagSource(load_bag_config(resolve(args.bag_config), {
        "poses": {"calibration": tmp.name}, "frame_rate_hz": args.candidate_rate_hz,
        "lidar_window_s": args.lidar_window_s}))
    frames = collect(src, args.n_frames, args.max_rot_rate, args)
    print(f"{len(frames)} calibration frames ({src.stats})")
    data = [(f["P"], *target(f["Q"], args.voxel, args.knn)) for f in frames]

    schedule = [(0.5, 10), (0.25, 10), (0.1, 15)]
    cad_score = score(X_cad, data, 0.1)
    X, log, eig = icp(X_cad, data, schedule, args.huber)
    ref_score = score(X, data, 0.1)
    d = np.linalg.inv(X_cad) @ X
    dt_cm = 100 * np.linalg.norm(X[:3, 3] - X_cad[:3, 3])
    drot = float(np.degrees(np.linalg.norm(Rotation.from_matrix(d[:3, :3]).as_rotvec())))
    rpy_cam = Rotation.from_matrix((np.linalg.inv(T_camlink_opt) @ np.linalg.inv(X_cad) @ X @ T_camlink_opt)[:3, :3]).as_euler("xyz", degrees=True)
    T_lidar_camlink_ref = X @ np.linalg.inv(T_camlink_opt)

    # time offset scan with the refined extrinsic (camera clock - lidar clock)
    lo, hi, step = (float(v) for v in args.time_offsets.split(":"))
    offsets, med = np.arange(lo, hi + 1e-9, step), []
    for off in offsets:
        dd = []
        for f, (P, *_rest) in zip(frames, data):
            T = src.pose(f["t"] + off)
            if T is None:
                continue
            dd.append((P, *target(transform_points(np.linalg.inv(T), f["Q_map"]), args.voxel, args.knn)))
        med.append(score(X, dd, 0.1)[0])
    best_off = float(offsets[int(np.argmin(med))])

    report = {
        "n_frames": len(frames), "gravity_tilt_deg": tilt,
        "cad": {"median_abs_residual_m": cad_score[0], "inliers_10cm": cad_score[1]},
        "refined": {"median_abs_residual_m": ref_score[0], "inliers_10cm": ref_score[1]},
        "refined_minus_cad": {"translation_cm": float(dt_cm), "rotation_deg": drot,
                              "camlink_rpy_deg": [float(v) for v in rpy_cam],
                              "camlink_xyz_m": [float(v) for v in T_lidar_camlink_ref[:3, 3]]},
        "hessian_eigenvalues": [float(e) for e in eig],
        "time_offset_scan": {f"{o:+.3f}": m for o, m in zip(offsets, med)},
        "best_time_offset_s": best_off, "icp_log": log,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k not in ("icp_log",)}, indent=2))

    calib = {
        "T_lidar_cam": matrix_to_yaml(X),
        "T_lidar_cam_cad": matrix_to_yaml(X_cad),
        "T_lidar_camlink": matrix_to_yaml(T_lidar_camlink_ref),
        "T_gravity_odom": matrix_to_yaml(T_g),
        "notes": (f"T_lidar_cam = livox_frame -> camera_color_optical_frame, refined by depth/LiDAR "
                  f"ICP on {len(frames)} frames (median |r| {ref_score[0]*100:.1f} cm vs CAD "
                  f"{cad_score[0]*100:.1f} cm). T_gravity_odom levels the odometry frame "
                  f"(tilt {tilt:.2f} deg). Generated by scripts/calibrate_lidar_camera.py."),
    }
    resolve(args.out).write_text(yaml.safe_dump(calib, sort_keys=False))
    print(f"wrote {args.out}")

    for i in np.linspace(0, len(frames) - 1, min(args.n_overlays, len(frames))).astype(int):
        f = frames[i]
        a = overlay(f["rgb"], f["K"], f["Q"], X_cad, args.zmax * 2)
        b = overlay(f["rgb"], f["K"], f["Q"], X, args.zmax * 2)
        for img, txt in ((a, "CAD prior"), (b, "refined")):
            cv2.putText(img, f"{txt}  t={f['t'] - traj.stamps[0]:.0f}s", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)
        cv2.imwrite(str(out_dir / f"overlay_{i:03d}.jpg"),
                    cv2.resize(np.hstack([a, b]), None, fx=0.5, fy=0.5))


if __name__ == "__main__":
    main()
