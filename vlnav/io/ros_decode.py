"""Decoders for the ROS messages used by the bag sources (pure numpy / OpenCV)."""
from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

COMPRESSED_DEPTH_HEADER_BYTES = 12   # image_transport compressed_depth ConfigHeader
PNG_SIGNATURE = b"\x89PNG"
MM_PER_M = 1000.0

# sensor_msgs/PointField datatype -> numpy dtype
_POINTFIELD_DTYPES = {1: np.int8, 2: np.uint8, 3: np.int16, 4: np.uint16, 5: np.int32,
                      6: np.uint32, 7: np.float32, 8: np.float64}


def stamp_s(header) -> float:
    return header.stamp.sec + header.stamp.nanosec * 1e-9


def decode_compressed_image(msg) -> np.ndarray:
    """sensor_msgs/CompressedImage (jpeg/png) -> HxWx3 uint8 RGB (AUDIT pitfall 9)."""
    bgr = cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"cannot decode image with format {msg.format!r}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


_RAW_TO_RGB = {"bgra8": cv2.COLOR_BGRA2RGB, "bgr8": cv2.COLOR_BGR2RGB, "rgba8": cv2.COLOR_RGBA2RGB,
               "rgb8": None, "mono8": cv2.COLOR_GRAY2RGB}
_RAW_CHANNELS = {"bgra8": 4, "bgr8": 3, "rgba8": 4, "rgb8": 3, "mono8": 1}


def decode_raw_image(msg) -> np.ndarray:
    """sensor_msgs/Image (8-bit color / mono encodings) -> HxWx3 uint8 RGB."""
    enc = msg.encoding
    if enc not in _RAW_CHANNELS:
        raise ValueError(f"unsupported image encoding {enc!r}")
    ch = _RAW_CHANNELS[enc]
    buf = np.frombuffer(bytes(msg.data), np.uint8).reshape(msg.height, msg.step)
    img = buf[:, :msg.width * ch].reshape(msg.height, msg.width, ch) if ch > 1 else buf[:, :msg.width]
    code = _RAW_TO_RGB[enc]
    return np.ascontiguousarray(img) if code is None else cv2.cvtColor(img, code)


def decode_image(msg) -> np.ndarray:
    """CompressedImage or Image -> RGB."""
    return decode_raw_image(msg) if hasattr(msg, "encoding") else decode_compressed_image(msg)


def decode_compressed_depth(msg) -> np.ndarray:
    """`16UC1; compressedDepth` (12-byte header + PNG of uint16 mm) -> HxW float32 meters."""
    data = bytes(msg.data)
    if not msg.format.startswith("16UC1") or data[COMPRESSED_DEPTH_HEADER_BYTES:][:4] != PNG_SIGNATURE:
        raise ValueError(f"unsupported compressedDepth format {msg.format!r}")
    raw = cv2.imdecode(np.frombuffer(data[COMPRESSED_DEPTH_HEADER_BYTES:], np.uint8),
                       cv2.IMREAD_UNCHANGED)
    return raw.astype(np.float32) / MM_PER_M


def decode_pointcloud_xyz(msg) -> np.ndarray:
    """sensor_msgs/PointCloud2 -> Nx3 float64 xyz, dropping NaN and all-zero points."""
    fields = {f.name: f for f in msg.fields}
    n = msg.width * msg.height
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(n, msg.point_step)
    cols = []
    for name in ("x", "y", "z"):
        f = fields[name]
        dt = np.dtype(_POINTFIELD_DTYPES[f.datatype]).newbyteorder(">" if msg.is_bigendian else "<")
        cols.append(buf[:, f.offset:f.offset + dt.itemsize].copy().view(dt).ravel())
    xyz = np.stack(cols, axis=1).astype(np.float64)
    ok = np.isfinite(xyz).all(axis=1) & (np.abs(xyz).sum(axis=1) > 0)
    return xyz[ok]


def camera_info_K_D(msg) -> tuple[np.ndarray, np.ndarray, int, int]:
    return (np.asarray(msg.k, dtype=float).reshape(3, 3), np.asarray(msg.d, dtype=float),
            int(msg.width), int(msg.height))


def transform_to_matrix(tr) -> np.ndarray:
    """geometry_msgs/Transform -> 4x4."""
    T = np.eye(4)
    q = tr.rotation
    T[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    T[:3, 3] = [tr.translation.x, tr.translation.y, tr.translation.z]
    return T
