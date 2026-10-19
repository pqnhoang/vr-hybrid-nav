# Plan: Agentic Code-as-Policy cho VLN

> Mở rộng từ hệ VL-Nav-style hiện có: NeSy planner (subtask `explore` / `goto` / `stop`) → goal pose trên
> occupancy grid → `GridAStar` (path nội bộ) → action `vr_common_msgs/NavigateToPose` (`vr_navigate_to_pose`)
> của robot. Repo **không có FAR Planner**; SYSTEM.md §7 chỉ nhắc tới nó như một adapter dự kiến.
> Coding agent **không nằm trong vòng điều khiển**. Agent loop ước tính mất 10–30 s (chưa đo), nên chỉ dùng ở
> tầng deliberative để sinh, kiểm chứng và sửa program.

**Thuật ngữ.** "Scene graph" trong plan này là **unified memory** hiện có (`vlnav/memory/`): object node
(nhãn, `label_votes`, tâm 3D, điểm, best-view pose và ảnh, `room_id`, `n_obs`) và room node (polygon, nhãn,
phòng kề nhau). Nó **không** phải 3D scene graph phân tầng kiểu Hydra (không có tầng places, floors,
buildings), không có hướng của vật, không có thuộc tính (màu, hình dạng), và không có cạnh quan hệ giữa
các vật (chỉ có truy vấn `neighbors`). Snapshot lấy qua `MemoryView.snapshot(t)`, JSON qua `memory/serialize.py`.

---

## 1. Ý tưởng chính

Coding agent sinh program từ instruction và snapshot scene graph. Trước khi submit, agent tự **validate và
dry-run** program trên một world model symbolic dựng từ scene graph và occupancy grid. Khi thực thi thật mà
fail, runtime gửi trace về để agent **sửa phần program còn lại**. Mọi action đều quy về **goal pose** cho
nav stack hiện có.

**Đóng góp có thể claim (cần làm rõ so với công trình 2026):**
- Program synthesis có kiểm chứng **trước khi thực thi** bằng dry-run trên scene-graph world model.
  ⚠ ASENA (arXiv:2609.39207, NVIDIA, 9/2026) đã có coding agent cho điều hướng humanoid, sửa lỗi theo
  feedback khi chạy và lưu skill qua nhiều lần chạy. Graph-as-Policy (graph-robots, 7/2026) đã có đồ thị skill
  có kiểu, kiểm tra tĩnh và checkpoint (cho manipulation). Điểm khác cần nhấn mạnh: dry-run trên world model
  symbolic trước khi robot di chuyển, và tầng phản xạ NeSy vẫn chạy realtime khi agent đang nghĩ.
- Repair dựa trên trace khi thực thi. Không còn là điểm mới riêng, vì ASENA đã làm.
- Thống nhất route-following VLN (R2R-style) và VLN dựa trên suy luận trong cùng một program.
  (Cần kiểm chứng lại tham khảo "TIAMAT" trước khi dùng tên này trong bài.)
- Prior cho exploration do LLM soạn, cộng vào NeSy score. Code đã có một dạng prior hẹp: `room_bonus`
  (`configs/exploration.yaml`, mặc định tắt). $S_\text{prior}$ là bản tổng quát của nó.

---

## 2. Kiến trúc

| Tầng | Thành phần | Chạy ở đâu (hiện tại) | Tần số |
|---|---|---|---|
| Deliberative | **Planner Agent**: synthesize, validate, dry-run, submit, repair | Cloud API | Mỗi episode và mỗi lần fail |
| Executive | **Program Runtime**: coroutine executor, monitor predicate, hot-swap program | Máy 2×RTX 4090 (`vrb-rtx5-pd`) | Theo nhịp frame, khoảng 2 Hz |
| Perception/Memory | Scene graph, image memory, YOLO-World + FastSAM (đã có), **VLM verify** (Qwen3-VL-8B, chỉ dùng như một tool; vẫn gán nhãn phòng) | Máy 4090: GPU 0 perception, GPU 1 Qwen | Khoảng 2 Hz |
| Reactive | Occupancy grid, NeSy exploration, `GridAStar` (**giữ nguyên**); robot nhận goal qua `NavigateToPose` | Máy 4090; nav stack phía robot | Theo frame; robot tự điều khiển |

Số đo thật (bag `vr_h3_2`, `runs/memory_rooms_vr_h3_2/stats.json`): frame 2 Hz, perception p50 154 ms /
p95 292 ms, memory update p50 42 ms / p95 133 ms. Mục tiêu 5–10 Hz trước đây là **không khả thi** với
YOLO-World-x + FastSAM-x. Chưa có bản chạy trên Orin hay Jetson.

**Fallback:** khi API timeout hoặc mất mạng, chuyển về `PlannerCore` hiện có (Qwen3-VL local). Đây là điểm
cộng cho deploy thực tế và cũng có sẵn để làm baseline.

---

## 3. API primitive (mọi action đều quy về pose trên occupancy grid)

Cột "Hiện trạng" đối chiếu với code ngày 15/10/2026.

| Primitive | Ngữ nghĩa | Hiện trạng |
|---|---|---|
| `sg.find(target, room=None, near=None, nearest_to=None)` → `[Obj]` | Khớp `target` với nhãn node | Có `find_by_text` và `SymbolicFilter`. **Phải khớp cả `detector_prompts` và ảnh crop**: CLIP chữ-chữ chỉ cho "green plant" ↔ "potted plant" = 0.896, sát ngưỡng 0.88; "leafy plant" 0.877 thì trượt; bàn bida bị gán nhãn `desk`. **Không có `attrs`**, vì node không lưu thuộc tính. |
| `sg.rel(a, b, viewer)` → `{"left_of", "right_of", "in_front_of", "near", ...}` | Tính bằng hình học, theo hướng nhìn của `viewer` | Chưa có. Có tâm 3D và `neighbors`. Quan hệ "trước/sau" theo **hướng của vật** không tính được, vì memory không có hướng vật. |
| `seen(target, min_conf=.5, min_views=3)` → `Obj \| None` | Predicate có hysteresis | Gần có: `n_obs`, `best_score`, tentative/confirmed; `candidate_found` trong `core.py`. |
| `vlm.verify(obj, question, image="current" \| "best_view")` → `bool` | Neural, đắt (khoảng 2.4 s/lần) | Verifier hiện tại chỉ chọn một ứng viên theo chữ cái. Câu hỏi yes/no tự do cần prompt mới. |
| `front_of(obj, d=0.8)` | Điểm đứng cách vật `d` m **về phía robot** (vật không có hướng) | Có sẵn logic standoff: `_goto_approach` trong `explorer.py`. |
| `between(a, b)`, `ray(heading, dist)`, `room(label)` | Điểm/vùng trên grid | `room`: có polygon và nhãn phòng. `ray`: cần viết (chiếu tia trên grid, dừng ở ô bị chặn hoặc ô chưa biết). |
| `fov_wedge(heading, deg)`, `frontiers(region)` | Vùng / tập frontier | Có code wedge và FrontierPool trong `exploration/`. |
| `go_to(p, until=None, timeout=60)` | Goal pose | Có nhánh goto theo pose (`selector.goto`). Thiếu `until`. **Goal trong vùng chưa biết luôn `nav_failed`**, vì `snap_goal` chỉ nhận ô free đã biết trong 0.5 m. |
| `follow(heading, step=2.0, until=None, max_dist=30)` | Receding horizon dọc theo tia, goal cố định trong map frame | Chưa có (chính là subtask `move`). |
| `explore(target, region=None, prior=None)` | Wrap NeSy exploration; trả về `Obj \| None` | Có, nhưng chưa nhận `region`/`prior`. |
| `rotate(deg)` | Dương = trái | Cần bật `goto_yaw_tol_rad` (hiện là `null`, nên xoay tại chỗ sẽ coi là đến nơi ngay). |
| `stop()`, `report(obj)` | | Có trong planner. |

**Prior cho exploration:**

$$S = w_d S_\text{dist} + w_{VL} S_{VL} S_\text{unk} + w_p\, S_\text{prior}(g)$$

trong đó $S_\text{prior}$ được tạo từ các spatial op (`room("kitchen")`, `fov_wedge(...)`, ...).

**Ví dụ:** *"Go forward till you see the green plant, turn left then stop in front of the table."*

```python
def task(api):
    start = api.robot.pose()                                # frame tham chiếu của câu lệnh
    plant = yield from api.follow(heading=start.yaw, until=lambda: api.seen("green plant"),
                                  max_dist=30)
    yield from api.rotate(+90)                              # dương = trái
    here = api.robot.pose()
    table = api.sg.find("table", nearest_to=here, ahead_of=here) \
            or (yield from api.explore("table", region=api.fov_wedge(here.yaw, 60)))
    if table is None:
        raise api.TaskFailed("no table")
    yield from api.go_to(api.front_of(table, d=0.8))
    if not api.vlm.verify(table, "Is a table directly ahead, within about 1 m?", image="current"):
        raise api.TaskFailed("table not in view on arrival")
    yield from api.stop()
```

Quy ước: mọi quan hệ (`ahead_of`, `left`) được định nghĩa trong **frame robot tại thời điểm bắt đầu
primitive**, và goal của `follow`/`rotate` chỉ tính **một lần**, rồi giữ cố định trong map frame.
`seen()` dùng target cùng `detector_prompts` của nó, và runtime đặt prompt đó cho detector trong lúc
primitive chạy. Không dùng `assert`, vì Python bỏ qua `assert` khi chạy với `-O`.

---

## 4. Tool cấp cho agent

Không cấp Bash hay quyền ghi file. Chỉ dùng tool custom:

| Tool | Chức năng |
|---|---|
| `api_docs()` | Spec của khoảng 15 primitive: signature, ngữ nghĩa, frame tham chiếu |
| `sg_query(target, room)`, `sg_summary()` | Chỉ đọc, dữ liệu là snapshot scene graph tại thời điểm gọi |
| `validate(program)` | Parse AST, chỉ cho phép tên trong `api`, cấm `import`, thuộc tính `__dunder__`, `exec`/`eval`; bắt buộc mọi vòng lặp và `follow`/`explore` có `max_dist`/`timeout` |
| `dry_run(program)` | Chạy program trên **mock world** dựng từ scene graph và occupancy grid. Object đã có thì `seen()` trả về true khi nằm trong FoV giả lập (ray-cast trên grid để xét che khuất); object chưa có thì sinh nhánh explore. Trả về trace goal pose và các điểm có thể fail (goal rơi vào ô chưa biết hoặc ô bị chặn) |
| `submit(program)` | Kết thúc vòng lặp |

Agent chỉ dùng tool tự định nghĩa, nên có hai lựa chọn:
- **Messages API + Tool Runner** (`anthropic`, `client.beta.messages.tool_runner` với `@beta_tool`): không có
  tool built-in nào để phải chặn, ít phụ thuộc hơn. **Khuyến nghị.**
- **Claude Agent SDK** (`claude_agent_sdk`): có sẵn harness của Claude Code. Phải tự loại tool built-in.

Khung cho Agent SDK (đã đối chiếu với code.claude.com/docs/en/agent-sdk/custom-tools ngày 15/10/2026):

```python
from claude_agent_sdk import query, ClaudeAgentOptions, tool, create_sdk_mcp_server

@tool("dry_run", "Simulate a task program on the scene-graph world model", {"program": str})
async def dry_run(args):
    trace = mock_world.run(args["program"], sg_snapshot)
    return {"content": [{"type": "text", "text": trace.to_json()}]}

nav = create_sdk_mcp_server(name="nav", version="0.1.0",
                            tools=[api_docs, sg_query, validate, dry_run, submit])
opts = ClaudeAgentOptions(
    mcp_servers={"nav": nav},
    tools=[],                                   # bỏ toàn bộ tool built-in (Bash, Write, ...) khỏi context
    allowed_tools=["mcp__nav__*"],              # chỉ để khỏi hỏi quyền; KHÔNG chặn được tool khác
    system_prompt=SYSTEM_PROMPT,
    max_turns=8,
)
```

`allowed_tools` chỉ là danh sách được duyệt sẵn. Muốn loại tool built-in phải dùng `tools=[]` hoặc
`disallowed_tools`.

---

## 5. Hợp đồng program và runtime

- **Định dạng:** `def task(api):` dưới dạng generator. Mỗi primitive `yield` một `Goal(pose, until, timeout)`.
- **Monitor:** predicate `until` được đánh giá trên scene graph theo nhịp frame (khoảng 2 Hz), không gọi VLM.
- **Safety:** không có primitive điều khiển vận tốc. Code chỉ được xuất pose. Nav stack của robot (server
  `NavigateToPose`) lo tránh vật cản, còn `GridAStar` chỉ kiểm tra có đường đi hay không.
- **Hot-swap:** program mới chỉ được thay vào ở ranh giới giữa hai primitive. Trong lúc agent đang repair,
  robot đi tới điểm an toàn gần nhất hoặc chạy explore cục bộ, không đứng yên.
- **Trace gửi về khi fail:** gồm dòng code fail, loại lỗi (timeout, `TaskFailed`, predicate không bao giờ
  đúng, `nav_failed`), pose, diff của scene graph so với snapshot cũ, và ảnh best-view liên quan.

---

## 6. Ngân sách latency và chi phí

| Pha | Model gợi ý | Mục tiêu |
|---|---|---|
| Synthesis lần đầu | `claude-opus-5-5` | ≤ 30 s, trong lúc chờ robot vẫn explore theo NeSy scoring |
| Repair | `claude-haiku-5-5` | ≤ 10 s |
| `max_turns` | 6–8 | Giới hạn chặn vòng lặp tool |

Cache `api_docs` trong system prompt (prompt caching). Log token mỗi episode để báo cáo, vì reviewer chắc
chắn sẽ hỏi về chi phí. Các con số latency trên là mục tiêu, cần đo trong tuần 4.

---

## 7. Lộ trình (khoảng 11–13 tuần)

Điều kiện tiên quyết: repo **chưa có Habitat** (milestone E5–E6 và S6 chưa làm). Mọi bước "trong sim" đều
phụ thuộc vào việc này.

| Tuần | Mục tiêu | Deliverable / Gate |
|---|---|---|
| 1 | Spec: chốt API primitive (frame, đơn vị, ngữ nghĩa `until`), viết `api_docs`, skeleton runtime, mock world | Tài liệu API, runtime chạy với program viết tay trên bag `vr_h3_2` (open-loop) |
| 2–3 | Primitive trên stack cũ: `follow`/`rotate` (subtask `move`), `go_to(until)`, spatial op, `seen` có hysteresis, khớp nhãn theo `detector_prompts` + ảnh, `explore(prior)` | Unit test trên grid tổng hợp; kiểm tra goal trên bản đồ bag thật |
| 4–6 | **Habitat harness** (E5): env wrapper, `HabitatSource` (RGB-D + pose, không có LiDAR → mapping từ depth), controller theo waypoint | Chạy closed-loop một scene |
| 7 | Tích hợp agent, eval offline khoảng 300 instruction (R2R-CE val-unseen + tập suy luận tự viết). Dry-run cần scene graph của từng scene MP3D, nên phải chạy perception + memory trên episode Habitat trước | Tỷ lệ parse, tỷ lệ dùng API hợp lệ, tỷ lệ pass dry-run, số turn và token. **Gate: pass dry-run ≥ 90%** |
| 8–9 | Closed-loop trong sim: R2R-CE/RxR-CE (subset khoảng 100 episode), HM3D-OVON hoặc GOAT-Bench; bật repair loop | Bảng kết quả chính |
| 10–11 | Ablation, robot thật: khoảng 20–30 task trên **humanoid `vr_h3_2`** (ZED + Livox MID-360), có cả route lẫn suy luận | Bảng ablation, video demo |
| 12–13 | Viết bài, release toàn bộ program và trace | Draft paper |

---

## 8. Thí nghiệm

**Baseline:**
- (a) Hệ hiện tại (subtask `explore`/`goto` cố định), cùng perception và cùng planner. Đây là so sánh được
  kiểm soát tốt nhất.
- (b) CaP sinh một lần bằng Messages API, không có tool.
- (c) CaP dùng Qwen3-VL-8B (local) sinh code.
- (d) Các method ngoài: VLFM, ApexNav, và một method zero-shot trên VLN-CE như InstructNav hoặc CA-Nav.
  Mỗi method đều cần chạy được trong Habitat của mình.

**Ablation:**
- Bỏ `dry_run`, chỉ giữ validate. **Đây là ablation quan trọng nhất**, vì nó chứng minh giá trị của agentic loop.
- Bỏ repair.
- Bỏ $S_\text{prior}$.
- Thay spatial op bằng best-view pose.
- Thay code bằng JSON plan graph (kiểu Graph-as-Policy: node `explore`/`goto`/`move`/`check`, cạnh có điều kiện).

**Metric:**
- SR, SPL, nDTW (cho route), MTUR.
- Tỷ lệ program thực thi không lỗi, số lần repair mỗi episode, latency và token mỗi episode.
- Variance qua 3 seed, vì agent không deterministic.

---

## 9. Rủi ro

| Rủi ro | Giảm thiểu |
|---|---|
| Mock world quá đơn giản khiến dry-run pass nhưng chạy thật vẫn fail | Đo định lượng tỷ lệ pass dry-run nhưng fail thật. Con số này tự nó đã là một kết quả đáng báo cáo |
| Habitat chưa có; lộ trình phụ thuộc vào nó | Làm harness ở tuần 4–6; nếu trễ, eval trên bag (open-loop) và robot thật |
| Khớp nhãn yếu (paraphrase, detector gán sai nhãn) làm `seen`/`sg.find` trượt | Khớp theo `detector_prompts` + embedding ảnh crop; thêm case paraphrase vào eval |
| Memory không có hướng vật và thuộc tính | `front_of` định nghĩa theo phía robot; thuộc tính hỏi qua `vlm.verify` |
| Goal trong vùng chưa biết bị `nav_failed` | `follow`/`go_to` dùng target trung gian là ô free xa nhất trên tia, cập nhật mỗi frame |
| Predicate `seen` dao động do detector | Yêu cầu nhiều view, có hysteresis, đặt khoảng cách tối thiểu |
| Lệch frame cho các quan hệ `ahead_of`, `left` | Mọi quan hệ định nghĩa trong frame robot tại thời điểm bắt đầu primitive |
| Trùng ý tưởng với ASENA / Graph-as-Policy | Định vị đóng góp quanh dry-run trước khi chạy và tầng NeSy realtime; so sánh trực tiếp trong related work |
| Phụ thuộc cloud và chi phí | Có fallback về planner cũ, có báo cáo chi phí |
| Agent dùng quá nhiều turn | Giới hạn `max_turns`, prompt caching |
| Reproducibility | Cố định model version, log toàn bộ transcript |

---

## Tham khảo

- VL-Nav: Neuro-Symbolic Reasoning-based Vision-Language Navigation — https://arxiv.org/abs/2502.00931
- ASENA: Self-evolving Agents for Embodied Navigation — https://arxiv.org/abs/2609.39207
- Graph-as-Policy — https://github.com/graph-robots/graph-as-policy
- Claude Agent SDK, custom tools — https://code.claude.com/docs/en/agent-sdk/custom-tools
