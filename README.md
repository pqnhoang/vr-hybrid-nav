# vr-hybrid-vln: VL-Nav re-implementation

Cài đặt lại hệ thống **VL-Nav** (arXiv:2502.00931): một **NeSy task planner** (Qwen3-VL tách câu lệnh
thành các subtask `explore` / `goto`, tra bộ nhớ, xác minh mục tiêu bằng ảnh) chạy trên một **unified
memory** (scene graph gồm object node và room node, kèm ảnh best-view), cùng một **NeSy exploration**
(frontier + instance point, chấm điểm theo Eq. 1–8 của paper). Hiện tại hệ thống chạy offline trên
ROS 2 bag (run mode A), không cần cài ROS.

Spec nằm trong `plans/`. `plans/SYSTEM.md` là nguồn chuẩn; mọi lựa chọn không có trong paper được ghi
trong `docs/design_choices.md`.

## Trạng thái

| Package | Nội dung | Trạng thái |
|---|---|---|
| `vlnav/io` | đọc bag `.mcap` / `.db3`, TF, trajectory, ghép ảnh + LiDAR + pose | xong (S1) |
| `vlnav/perception` | YOLO-World + FastSAM, đưa mask lên 3D bằng depth hoặc LiDAR chiếu vào ảnh | xong (S2) |
| `vlnav/mapping` | occupancy grid 2D từ LiDAR / depth (5 cm, log-odds, ray casting), export bản đồ ROS | xong (S3) |
| `vlnav/navigation` | `GridAStar` (đường đi trên grid); controller mới chỉ có interface | xong (S4) |
| `vlnav/memory` | object node, room (phân vùng + gán nhãn bằng VLM), snapshot, checkpoint | xong (M1–M5); thiếu `RemoteMemoryMirror` (M6) |
| `vlnav/acquisition` | khớp nhãn bằng CLIP, chọn top-k, VLM xác minh bằng best-view | xong (A1–A6) |
| `vlnav/planner` | FSM, decompose / replan bằng Qwen3-VL, validation, decision log | xong (P4–P7) |
| `vlnav/exploration` | frontier, VL score, instance point, curiosity, chọn goal, debug view | xong E1–E4 + bag shadow mode; thiếu Habitat (E5–E6) |
| `vlnav/runtime` | vòng lặp full system mode A (bag) | xong (S5) |
| — | mode B (Habitat), ROS 2 node + robot (mode C) | chưa làm (S6, S7) |

## Cấu trúc thư mục

```
vlnav/            code (io, perception, mapping, navigation, memory, acquisition, planner, exploration, runtime, sim, viz)
configs/          <package>.yaml = mặc định; bags/ = cấu hình từng bag; runs/ = override cho từng lần chạy; vocab/; calib/
scripts/          chạy trên bag (run_*_bag.py), xem log (inspect_*.py), export bản đồ, kiểm tra extrinsic
eval/             đánh giá acquisition và planner (open-loop)
tests/            unit test (không cần GPU / mạng)
docs/             running.md, visualization.md, design_choices.md
plans/            spec
checkpoint/       model weights (không track bằng git)
sample-data/      bag mẫu
runs/             kết quả chạy (không track bằng git)
```

---

## 1. Cài môi trường

### 1.1 Trên máy `vrb-rtx5-pd` (đã có sẵn)

Dùng env `fastvggt` có sẵn (Python 3.10, torch 2.11 + CUDA 13.0, 2 × RTX 4090). Không cần cài gì thêm.
Mọi lệnh chạy **từ thư mục gốc repo** với `PYTHONPATH=.`:

```bash
cd ~/hoangpqn/vr-hybrid-vln
export PY=~/miniconda3/envs/fastvggt/bin/python      # nên thêm dòng này vào ~/.zshrc
PYTHONPATH=. $PY -m pytest -q tests/                  # ~195 test, khoảng 3 s
```

> Nếu gặp `zsh: permission denied: scripts/...py` thì nghĩa là `$PY` chưa được đặt: lệnh đã chạy
> thẳng file `.py` thay vì gọi qua Python.

### 1.2 Cài mới trên máy khác

Yêu cầu: Linux, GPU NVIDIA (khuyến nghị 2 × 24 GB: một GPU cho perception, một cho Qwen3-VL-8B;
một GPU 24 GB cũng chạy được nếu chia model, xem mục 2), Python 3.10, `ffmpeg` (không bắt buộc,
dùng để xuất video H.264).

```bash
conda create -n vlnav python=3.10 -y
conda activate vlnav
# torch: chọn bản CUDA hợp với driver (xem `nvidia-smi`); máy hiện tại dùng cu130 (driver 580)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
# vlnav + các extra: test, đọc bag, perception, VLM, Rerun (+ calib = open3d, chỉ cho script calibrate)
pip install -e ".[dev,io,perception,vlm,viz]"
sudo apt install ffmpeg            # không bắt buộc
```

Phiên bản đang dùng: torch 2.11, ultralytics 8.4.37, transformers 5.5.4 (Qwen3-VL cần ≥ 4.57),
accelerate 1.13, clip (openai/CLIP từ git), rosbags 0.9.23, mcap 1.3.1, scikit-image 0.25,
pydantic 2.12, numpy 2.2. **rerun-sdk phải đúng 0.30.2**: viewer chỉ mở được file `.rrd` ghi bởi
cùng phiên bản.

### 1.3 Model weights → `checkpoint/`

Đường dẫn khai báo trong `configs/checkpoints.yaml` và `configs/vlm.yaml`:

```
checkpoint/detector/yolov8x-worldv2.pt          YOLO-World (detector)
checkpoint/segmenter/FastSAM-x.pt               FastSAM (mask)
checkpoint/text_encoder/clip/ViT-B-32.pt        CLIP ViT-B/32 (text encoder của YOLO-World + khớp nhãn)
checkpoint/vlm/Qwen3-VL-8B-Instruct/            VLM (bản Instruct, KHÔNG dùng Thinking), ~17 GB
```

```bash
mkdir -p checkpoint/{detector,segmenter,text_encoder/clip,vlm}
# YOLO-World + FastSAM: ultralytics tự tải về thư mục hiện tại
(cd checkpoint/detector && $PY -c "from ultralytics import YOLO; YOLO('yolov8x-worldv2.pt')")
(cd checkpoint/segmenter && $PY -c "from ultralytics import FastSAM; FastSAM('FastSAM-x.pt')")
# CLIP ViT-B/32
$PY -c "import clip; clip.load('ViT-B/32', device='cpu', download_root='checkpoint/text_encoder/clip')"
# YOLO-World đọc CLIP từ ~/.cache/clip: tạo symlink
mkdir -p ~/.cache/clip && ln -sf "$PWD/checkpoint/text_encoder/clip/ViT-B-32.pt" ~/.cache/clip/ViT-B-32.pt
# Qwen3-VL-8B-Instruct (Hugging Face)
hf download Qwen/Qwen3-VL-8B-Instruct --local-dir checkpoint/vlm/Qwen3-VL-8B-Instruct
```

### 1.4 Dữ liệu

Bag mẫu nằm trong `sample-data/`; mỗi bag có một file cấu hình trong `configs/bags/`:

| Bag | Config | Ghi chú |
|---|---|---|
| `sample-data/vr_h3_2_calib_lidar_rgb_20261007_093649_calib_1` (`.db3`) | `configs/bags/vr_h3_2_calib_1.yaml` | humanoid, văn phòng 98 s, ZED (không có depth) + Livox MID-360, pose từ TF |
| `sample-data/bkhn_round1` (`.mcap`) | `configs/bags/bkhn_round1.yaml` | campus ngoài trời, RealSense D455 (có depth) + Livox, pose từ file odometry |

### 1.5 Kiểm tra cài đặt

```bash
PYTHONPATH=. $PY -m pytest -q tests/                        # không cần GPU
# chạy thử GPU trên 10 s đầu của bag (perception + grid + exploration, không cần VLM)
PYTHONPATH=. $PY scripts/run_exploration_bag.py --bag-config configs/bags/vr_h3_2_calib_1.yaml \
  --run-config configs/runs/vr_h3_2_exploration.yaml --run-dir runs/smoke --target sofa --end-offset 10
```

---

## 2. Cấu hình

Cấu hình chồng theo 3 lớp, lớp sau đè lớp trước:

1. `configs/<package>.yaml`: mặc định của từng package (`perception`, `mapping`, `navigation`,
   `memory`, `exploration`, `acquisition`, `planner`, `vlm`, `sim`, `viz`).
2. `configs/runs/*.yaml` (`--run-config`): override cho từng lần chạy, viết dưới key tên package.
   Ví dụ:
   ```yaml
   exploration:
     scoring: {w_vl: 6.0}
     frontier: {min_cluster_cells: 10}
   vlm: {cache_dir: runs/vlm_cache}
   ```
3. Tham số dòng lệnh (`--instruction`, `--start-offset`, các cờ ablation, ...).

`--bag-config` (trong `configs/bags/`) mô tả bag: topic, nguồn pose, extrinsic, tần số frame.

**GPU:** perception chạy trên `cuda:0` (`configs/perception.yaml` → `detector.device`). Qwen3-VL chạy
trên GPU 1 (`configs/vlm.yaml` → `local.devices: [1]`); đặt `[0, 1]` để chia model qua 2 GPU. Có thể
dùng server vLLM thay cho bản chạy trong tiến trình: `vllm serve checkpoint/vlm/Qwen3-VL-8B-Instruct
--port 8000` rồi đặt `backend: openai`.

---

## 3. Chạy

Mọi lệnh: `PYTHONPATH=. $PY <script> ...` từ thư mục gốc repo. Phần "Lệnh" chỉ ghi script và tham số.

| Muốn làm gì | Lệnh | Cần VLM? |
|---|---|---|
| **Exploration** (bag shadow mode, một subtask explore cố định) | `scripts/run_exploration_bag.py --bag-config configs/bags/vr_h3_2_calib_1.yaml --run-config configs/runs/vr_h3_2_exploration.yaml --run-dir runs/expl --target sofa --prompts sofa,armchair --rerun save` | không |
| **Memory** (perception + grid + scene graph + room) | `scripts/run_memory_bag.py --bag-config configs/bags/vr_h3_2_calib_1.yaml --run-config configs/runs/vr_h3_2_memory.yaml --run-dir runs/mem --label-rooms --rerun save` | chỉ khi có `--label-rooms` |
| **Full system** (thêm planner, acquisition, exploration shadow) | `scripts/run_system_bag.py --bag-config configs/bags/vr_h3_2_calib_1.yaml --run-config configs/runs/vr_h3_2_system.yaml --run-dir runs/sys --rerun save --instruction "I need to print a document, then sit somewhere and watch TV."` | có |
| Đánh giá planner open-loop (5 task có ground truth) | `eval/planner/run_openloop.py --bag-config configs/bags/vr_h3_2_calib_1.yaml --run-config configs/runs/vr_h3_2_system.yaml --gt eval/planner/gt_vr_h3_2.yaml --out runs/openloop` | có |
| Tạo fixture + đánh giá acquisition | `eval/acquisition/make_fixture.py --from-scene-graph runs/mem/memory/final --image-root runs/mem/memory/images --out runs/fixtures/mem --confirmed-only --link`, rồi `eval/acquisition/eval_acquisition.py --fixture runs/fixtures/mem/memory.json --cases eval/acquisition/cases_vr_h3_2.yaml --out runs/eval_acq/default.csv` | có (`--no-vlm` để bỏ) |
| Xuất bản đồ từ một run đã có | `scripts/export_map.py --run-dir runs/mem` | không |
| Kiểm tra extrinsic LiDAR ↔ camera (ảnh overlay) | `scripts/check_extrinsics.py --bag-config configs/bags/vr_h3_2_calib_1.yaml --out-dir runs/extr` | không |

Tham số dùng chung:
- `--start-offset 15 --end-offset 30`: chỉ chạy đoạn bag từ giây 15 đến 30.
- `--realtime 1.0`: phát theo tốc độ thật của bag (dùng khi demo).
- Cờ ablation cho exploration: `--no-ibtp`, `--no-curiosity`, `--nearest-frontier`,
  `--vl-update-mode {first,max,conf_weighted}`, `--room-bonus`.

**Cache:** `configs/runs/*` đặt `detector.cache_dir`, nên detection được lưu lên đĩa theo (frame,
vocabulary). Chạy lại sẽ nhanh; thêm từ mới vào prompt thì frame đó phải detect lại. `vlm.cache_dir`
lưu câu trả lời VLM theo prompt, nên chạy lại cho kết quả giống hệt.

### Kết quả trong `--run-dir`

| File / thư mục | Có ở | Nội dung |
|---|---|---|
| `summary.json` / `stats.json` | mọi run | thời gian từng module, room, report, thống kê |
| `recording.rrd` | khi có `--rerun save` | bản ghi Rerun |
| `exploration/` | exploration, system | `shadow.mp4` (video debug top-down), `frames/`, `final.png`, `shadow_log.jsonl`, `stats.json` |
| `map/` | memory, system | `occupancy.png`, `occupancy.pgm` + `.yaml` (định dạng ROS map_server / Nav2), `rooms.png`, `trajectory.csv` |
| `memory/` | memory, system | checkpoint scene graph (`graph.json`, `grid.npz`, ...), ảnh best-view, `final/` |
| `topdown/`, `gallery.jpg`, `events.jsonl` | memory | bản đồ top-down theo thời gian, ảnh best-view, sự kiện memory |
| `decisions.jsonl`, `planner_debug/`, `acquisition_debug.jsonl`, `reports.json` | system | toàn bộ quyết định của planner / VLM |

### Xem bằng Rerun

```bash
rerun runs/expl/recording.rrd                    # mở file đã ghi (viewer cùng phiên bản 0.30.2)
rerun runs/sys_a/recording.rrd runs/sys_b/recording.rrd   # so sánh nhiều run trong một viewer
```

- `--rerun save` ghi file `.rrd`.
- `--rerun spawn` mở viewer trên máy đang chạy.
- `--rerun serve,save --realtime 1.0` cho phép xem trực tiếp từ trình duyệt máy khác: mở URL được
  in ra (`http://<ip>:9090/?url=rerun+http://<ip>:9876/proxy`). Khi xong, tắt server bằng
  `pkill -f "rerun --serve-web"`.

Mỗi run chỉ hiện các tab nó thực sự có dữ liệu:

| Tab | Nội dung |
|---|---|
| 3D | LiDAR, bản đồ, camera, object node, room, frontier / goal / đường đi |
| camera | detection, mask, object node chiếu vào ảnh |
| exploration (top-down) | xem mục 4.1 |
| occupancy + rooms | bản đồ phòng |
| latest best view | ảnh best-view mới nhất |
| task / planner | trạng thái task và suy luận của VLM |
| log | log exploration và memory |

Chi tiết: `docs/visualization.md`.

---

## 4. Debug

### 4.1 Exploration

Tab **exploration (top-down)** (cũng chính là nội dung `exploration/shadow.mp4`):

- **Bản đồ:**
  - Ô trắng / xám đậm / xám nhạt: trống / vật cản / chưa biết.
  - Ô xanh nhạt: frontier.
  - Quạt vàng: tầm nhìn camera.
  - Chấm tròn: frontier point; màu theo điểm S_NeSy (xanh thấp → đỏ cao), vòng xanh lá = có bằng
    chứng VL, xám = không được chọn, × = bị blacklist.
  - Hình thoi: instance point (cam active, xanh lá visited, đỏ unreachable); ô vuông nhỏ = approach
    point.
  - Vòng hồng tím: goal; đường hồng tím: đường A* tới goal.
  - Chấm xanh dương + mũi tên: robot.
- **Bảng bên phải:**
  - **TASK:** câu lệnh, trạng thái planner, và kế hoạch (danh sách target: `[x]` đã report, `[>]`
    target hiện tại, `[ ]` chưa làm, `[!]` thất bại). Khi chạy exploration riêng (không có planner),
    phần này hiện subtask cố định; dùng `--instruction "..."` nếu muốn ghi kèm câu lệnh.
  - subtask, goal, các phát hiện VL của frame (`nhãn conf@góc`, góc dương = bên trái), và bảng ứng
    viên `S_NeSy = 1·S_dist + 3·S_VL·S_unk`.

Trong Rerun, rê chuột lên một điểm để xem chi tiết điểm số của nó.

Quy trình debug, từ ngoài vào trong:

```bash
# 1. dòng thời gian: mỗi lần đổi goal + lý do (bị lộ / không có đường / bị vượt điểm / instance ưu tiên)
PYTHONPATH=. $PY scripts/inspect_exploration.py runs/expl
# 2. một bước: goal, bảng xếp hạng, instance; vì sao frontier 110 không được chọn
PYTHONPATH=. $PY scripts/inspect_exploration.py runs/expl --step 40 --why 110
# 3. lịch sử của một frontier point / một instance
PYTHONPATH=. $PY scripts/inspect_exploration.py runs/expl --point 202
PYTHONPATH=. $PY scripts/inspect_exploration.py runs/expl --instance 1
# 4. dừng replay ở bước 40 và xem trạng thái thật (pdb; gõ `c` để chạy tiếp)
PYTHONPATH=. $PY scripts/run_exploration_bag.py ... --end-offset 30 --stop-at 40
#    trong pdb: expl.frontiers.points[202], expl.planner.plan(grid, frame.T_map_base[:2, 3], (9.0, 4.3)),
#               res.detections, expl._mix, snap
```

Muốn thử tham số thì sửa phần `exploration:` trong một bản sao của run config (đầu tiên là
`scoring.w_vl` so với `w_dist`), chạy lại một đoạn ngắn rồi so `stats.json`. Để tái hiện trong cảnh tổng
hợp, dùng `tests/exploration/expl_helpers.py`.

Lưu ý: ở shadow mode robot đi theo bag chứ không theo goal, nên goal thường đổi khi LiDAR "nhìn thấy"
luôn vùng frontier đó. Đây là bản đồ thay đổi thật, không phải lỗi.

### 4.2 Planner / VLM

Mỗi lần gọi VLM được lưu trong `<run-dir>/planner_debug/NNN_<kind>/`: `prompt.txt`, `images/`,
`reply_0.txt`, `input.json`, `result.json`, `memory/`.

```bash
PYTHONPATH=. $PY scripts/inspect_decisions.py runs/sys                    # dòng thời gian quyết định
PYTHONPATH=. $PY scripts/inspect_decisions.py runs/sys --show 2 --prompt  # một quyết định đầy đủ
# chạy lại đúng một lần gọi VLM với prompt / config hiện tại (~3 s, không replay bag)
PYTHONPATH=. $PY scripts/replay_planner_query.py runs/sys/planner_debug/003_replan --show-prompt
```

- Prompt planner: `vlnav/planner/prompts/*.j2`.
- Prompt xác minh mục tiêu: `vlnav/acquisition/prompts/verify_target.j2`.
- Prompt gán nhãn phòng: `vlnav/memory/prompts/room_label.j2`.
- Log xác minh mục tiêu: `acquisition_debug.jsonl`.
- Trong Rerun: tab task và planner.

### 4.3 Memory, perception, bản đồ

- **Memory:**
  - `topdown/*.png` cho bản đồ object theo từng checkpoint;
  - `gallery.jpg` cho ảnh best-view của từng node;
  - `events.jsonl` cho node tạo / merge / dedup / room;
  - `stats.json` cho các cặp cùng nhãn nằm gần nhau, tức nghi trùng.
- **Perception:**
  - trong Rerun, tab camera: màu theo nhãn = đã lên 3D, đỏ `[no_3d]` = không có depth, xám
    `[on_dynamic]` = nằm trên người;
  - `scripts/check_extrinsics.py` vẽ overlay LiDAR lên ảnh để kiểm tra calib.
- **Bản đồ:** `map/occupancy.png`; tham số trong `configs/mapping.yaml` (độ phân giải, dải chiều cao
  vật cản, mô hình mặt sàn).

### 4.4 Lỗi thường gặp

| Hiện tượng | Nguyên nhân / cách sửa |
|---|---|
| `zsh: permission denied: scripts/...py` | `$PY` chưa đặt: `export PY=~/miniconda3/envs/fastvggt/bin/python` |
| `FileNotFoundError: checkpoint missing ...` | thiếu weights: mục 1.3 |
| `YOLO-World loads its CLIP text encoder from ~/.cache/clip/...` | tạo symlink `~/.cache/clip/ViT-B-32.pt` (mục 1.3) |
| CUDA out of memory khi load Qwen3-VL | GPU đang bị tiến trình khác dùng (`nvidia-smi`); hoặc đặt `vlm: {local: {devices: [0, 1]}}` |
| `ValueError: ... angle_reference 'robot' ...` | camera lệch hướng > 2° so với robot, hoặc có nhiều camera: đặt `exploration: {angle_reference: camera}` |
| Tab Rerun trống / layout cũ | bấm "Reset to default blueprint" ở panel blueprint; viewer phải là 0.30.2 (`rerun --version`) |
| Port 9090 / 9876 đang bị chiếm | `pkill -f "rerun --serve-web"` hoặc đổi `--web-port` / `--grpc-port` |
| Chạy lần đầu chậm (~300 ms/frame perception) | bình thường; các lần sau dùng detection cache |
| Kết quả planner khác giữa các lần chạy | đặt `vlm.cache_dir`; để nhãn phòng cũng tất định: `memory: {labeling: {worker_thread: false}}` (chậm hơn) |

---

## 5. Thêm một bag mới

1. Sao chép một file trong `configs/bags/` rồi sửa: `bag_dir`, topic camera / LiDAR / depth,
   `poses` (`kind: tf` với `map_frame` / `base_frame`, hoặc `kind: trajectory` với file odometry),
   `extra_static_tf` nếu thiếu optical frame, `frame_rate_hz`.
2. Kiểm tra calib: `scripts/check_extrinsics.py --bag-config <file> --out-dir runs/extr`. Các điểm
   LiDAR phải khớp cạnh vật trong ảnh. Nếu lệch, xem `scripts/calibrate_lidar_camera.py`.
3. Tạo run config trong `configs/runs/`:
   - `perception.vocabulary.generic_vocab_file`: `configs/vocab/office.txt` cho trong nhà,
     `campus.txt` cho ngoài trời;
   - `detector.cache_dir`;
   - `memory.rooms.enabled`: tắt khi ở ngoài trời.
4. Chạy memory trước (`run_memory_bag.py`), xem `map/` và `topdown/`, rồi mới chạy exploration và
   full system.

## 6. Tài liệu thêm

- `docs/running.md`: bảng lệnh đầy đủ, output, debug (tiếng Anh).
- `docs/visualization.md`: Rerun, ý nghĩa từng view.
- `docs/design_choices.md`: mọi quyết định thiết kế ngoài paper và số đo trên bag thật.
- `plans/`: spec (`SYSTEM.md`, `TASK_*.md`, `AUDIT_AND_PITFALLS.md`).
