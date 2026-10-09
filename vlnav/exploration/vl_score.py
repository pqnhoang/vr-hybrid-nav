"""VL module and VL score (TASK_nesy_exploration.md §5, paper Eq. 3–4).

Per relevant detection k (label ∈ active prompts): μ_k = −atan((u_k − c_x)/f_x) from the mask
centroid column, CCW-positive (left of center > 0, AUDIT pitfall 4); α_k = score; σ_k = sigma_rad.
For a point g with Δθ = wrap(atan2(g − c) − ψ_c) (reference pose per `angle_reference`):

    S_VL(g) = clip_[0,1]( Σ_k α_k exp(−½((Δθ − μ_k)/σ_k)²) · C(Δθ) ),  C = cos²(Δθ/(h_fov/2) · π/2)

and 0 outside the wedge (|Δθ| > h_fov/2 or beyond R).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vlnav.common.geometry import ground_heading, wrap_angle
from vlnav.common.models import Frame, FrameResult


@dataclass(frozen=True)
class RefPose:
    x: float
    y: float
    psi: float


@dataclass(frozen=True)
class Mixture:
    mu: np.ndarray
    alpha: np.ndarray
    sigma: np.ndarray
    labels: tuple[str, ...] = ()


def reference_pose(frame: Frame, angle_reference: str) -> RefPose:
    """`robot` (paper): base position + heading; `camera`: camera position + optical z axis on the
    ground plane (AUDIT 2.1, pitfall 2)."""
    if angle_reference == "robot":
        T, axis = frame.T_map_base, 0
    else:
        T, axis = frame.T_map_cam, 2
    return RefPose(float(T[0, 3]), float(T[1, 3]), ground_heading(T[:3, :3], axis))


def bearing(u, K: np.ndarray):
    """Angle of an image column relative to the optical axis, CCW-positive."""
    return -np.arctan((np.asarray(u, dtype=float) - K[0, 2]) / K[0, 0])


def mixture(res: FrameResult, sigma_rad: float) -> Mixture:
    dets = [res.detections[i] for i in res.relevant_idx]
    mu = bearing([d.mask_centroid_uv[0] for d in dets], res.frame.K) if dets else np.zeros(0)
    return Mixture(mu=np.asarray(mu, float), alpha=np.array([d.score for d in dets], float),
                   sigma=np.full(len(dets), sigma_rad), labels=tuple(d.label for d in dets))


def view_confidence(dtheta, hfov: float) -> np.ndarray:
    dtheta = np.asarray(dtheta, dtype=float)
    c = np.cos(dtheta / (hfov / 2) * np.pi / 2) ** 2
    return np.where(np.abs(dtheta) <= hfov / 2, c, 0.0)


def relative_bearing(xy: np.ndarray, ref: RefPose) -> tuple[np.ndarray, np.ndarray]:
    d = np.asarray(xy, dtype=float).reshape(-1, 2) - (ref.x, ref.y)
    return wrap_angle(np.arctan2(d[:, 1], d[:, 0]) - ref.psi), np.hypot(d[:, 0], d[:, 1])


def s_vl(xy: np.ndarray, ref: RefPose, hfov: float, mix: Mixture,
         max_range: float | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(S_VL, C(Δθ), in_wedge) per point. `max_range=None`: no range gate (instance ranking)."""
    dtheta, dist = relative_bearing(xy, ref)
    inside = np.abs(dtheta) <= hfov / 2
    if max_range is not None:
        inside &= dist <= max_range
    c = view_confidence(dtheta, hfov)
    if len(mix.mu):
        g = np.sum(mix.alpha * np.exp(-0.5 * ((dtheta[:, None] - mix.mu) / mix.sigma) ** 2), axis=1)
    else:
        g = np.zeros(len(dtheta))
    return np.where(inside, np.clip(g * c, 0.0, 1.0), 0.0), c, inside


def assign(p, s: float, c: float, subtask_id: int, mode: str) -> None:
    """§5.3 per-subtask validity + `vl_update_mode` on a FrontierPoint."""
    if p.s_vl_subtask_id != subtask_id:
        p.s_vl, p.s_vl_subtask_id, p.n_vl_updates, p.vl_weight = float(s), subtask_id, 1, float(c)
        p.vl_xy = p.xy
        return
    if mode == "first":
        return
    if mode == "max":
        if float(s) > p.s_vl:
            p.s_vl, p.vl_xy = float(s), p.xy
    else:                                            # conf_weighted (VLFM-style)
        w = p.vl_weight + float(c)
        if w > 0:
            p.s_vl = (p.s_vl * p.vl_weight + float(s) * float(c)) / w
        p.vl_weight, p.vl_xy = w, p.xy
    p.n_vl_updates += 1
