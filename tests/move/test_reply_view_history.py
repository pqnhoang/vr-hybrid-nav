"""Move M1: S2 text parsing (InternNav rules), the virtual Habitat view, history sampling."""
import numpy as np
from move_helpers import K

from vlnav.move.history import FrameBuffer, history_ids
from vlnav.move.reply import parse_s2_text
from vlnav.move.view import make_view


def test_parse_pixel_actions_and_garbage():
    r = parse_s2_text("(123, 45)")
    assert r.kind == "pixel" and r.pixel_xy == (123, 45)
    assert parse_s2_text("123,45").pixel_xy == (123, 45)
    assert parse_s2_text("318 90").pixel_xy == (318, 90)
    assert parse_s2_text("STOP").is_stop
    assert parse_s2_text("←←").actions == ["left", "left"]
    assert parse_s2_text("↓").actions == ["look_down"]
    assert parse_s2_text("↑→").actions == ["forward", "right"]
    assert parse_s2_text("go somewhere").kind == "invalid"
    assert parse_s2_text("7").kind == "invalid"                 # one number is not a pixel


def test_habitat_view_crop_matches_training_camera():
    Kc = np.array([[739.74, 0, 964.65], [0, 739.74, 547.01], [0, 0, 1]])       # vr_h3_2 ZED 1920x1080
    v = make_view(Kc, (1920, 1080), "habitat")
    Kv = v.K(Kc)
    assert (v.out_w, v.out_h) == (640, 480)
    assert abs(np.degrees(2 * np.arctan(320 / Kv[0, 0])) - 79.0) < 0.01       # hfov 79 deg
    assert abs(Kv[0, 2] - 320) < 1e-6 and abs(Kv[1, 2] - 240) < 1e-6           # principal point at the center
    for uv in ((964.65, 547.01), (400.0, 200.0), (1500.0, 900.0)):
        assert np.allclose(v.to_camera(v.to_view(uv)), uv)
    img = np.zeros((1080, 1920, 3), np.uint8)
    u, w = int(round(v.to_camera((100, 50))[0])), int(round(v.to_camera((100, 50))[1]))
    img[w - 3:w + 4, u - 3:u + 4] = 255
    out = v.image(img)
    assert out.shape == (480, 640, 3)
    y, x = np.unravel_index(np.argmax(out[..., 0]), out.shape[:2])
    assert abs(x - 100) <= 1 and abs(y - 50) <= 1


def test_view_clamps_to_the_image_and_full_is_identity():
    v = make_view(K, (640, 480), "habitat")                    # 77 deg camera: narrower than 79 deg
    assert v.x0 >= 0 and v.y0 >= 0 and v.x0 + v.w <= 640 + 1e-9 and v.y0 + v.h <= 480 + 1e-9
    assert abs(v.w / v.h - 4 / 3) < 1e-9
    f = make_view(K, (640, 480), "full")
    img = np.random.default_rng(0).integers(0, 255, (480, 640, 3), np.uint8)
    assert f.image(img) is img and f.to_camera((10, 20)) == (10, 20)


def test_history_ids_match_internnav():
    assert history_ids(0, 8) == []
    assert history_ids(1, 8) == [0]
    assert history_ids(5, 8) == [0, 1, 2, 3, 4]
    assert history_ids(30, 8) == np.unique(np.linspace(0, 29, 8, dtype=np.int32)).tolist()


def test_frame_buffer_keeps_frames_by_motion():
    buf = FrameBuffer(step_m=0.25, step_rad=np.radians(15), size_wh=(384, 384), num_history=8)
    img = np.zeros((480, 640, 3), np.uint8)
    assert buf.add(0.0, img, (0, 0), 0.0)
    assert not buf.add(0.5, img, (0.1, 0), 0.0)                # moved 0.1 m
    assert buf.add(1.0, img, (0.3, 0), 0.0)                    # moved 0.3 m
    assert buf.add(1.5, img, (0.3, 0), np.radians(20))         # turned 20 deg
    assert buf.kept[0].rgb.shape == (384, 384, 3)
    assert [k.stamp for k in buf.history(1.5)] == [0.0, 1.0]  # strictly before the query frame
