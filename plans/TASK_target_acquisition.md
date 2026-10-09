# TASK: Implement the Target Acquisition module (coarse-to-fine verification)

Reference: VL-Nav (arXiv:2502.00931v7), Sec. III-A.2, "Task Decomposition and Replanning".
This module is one component of the NeSy Task Planner. **Read `SYSTEM.md` first** (shared models,
repo layout, global milestone order) and `TASK_nesy_task_planner.md` §5.6 (how the planner calls
this module). It must be implementable and testable **standalone**, before the real scene graph /
perception pipeline exists (fixture-backed `InMemoryMemory`).

---

## 1. Goal

Given a target name (e.g. `"rain jacket"`), the original abstract instruction
(e.g. `"Today's weather report indicates rain. Help Rob find an umbrella, a jacket and shoes."`),
and the current symbolic memory, decide **which object node (if any) is the target** and return the
saved best-view pose of that node as the navigation goal for a `GOTO` subtask.

Two stages:

1. **Symbolic Filtering (coarse)** — query the scene graph for nodes semantically related to the
   target, rank by detection confidence, keep top-k (default k=3). No VLM call.
2. **Neural Verification (fine)** — send the best-view images + neighbor/room context of the top-k
   candidates to a VLM (Qwen3-VL-8B via vLLM). The VLM selects the candidate that best matches the
   abstract instruction, or rejects all.

Output: chosen `node_id` + its `best_view_pose`, or `None` (caller will issue an `EXPLORE` subtask).

## 2. Non-goals (do NOT implement in this task)

- Building the scene graph, detection, data association, room segmentation, LIO.
- Task decomposition / replanning prompts (other planner component).
- Exploration, path planning.
- ROS 2 wiring (handled by `planner_node`, SYSTEM milestone S7). This package is pure Python, ROS-free.

## 3. Tech stack & constraints

- Python 3.10+, type hints everywhere, `dataclasses` or `pydantic` v2 for models.
- VLM: Qwen3-VL-8B-Instruct served by vLLM with the OpenAI-compatible API
  (`/v1/chat/completions`), images sent as base64 data URLs. Use `openai` Python client.
- Text embeddings for label matching: `open_clip` (SigLIP or ViT-L/14) **or**
  `sentence-transformers`; hide behind an interface so it can be swapped.
- Image ops: `opencv-python` / `Pillow`. Config: YAML + pydantic settings.
- Tests: `pytest`. No network / GPU required for unit tests (mock VLM + mock embedder).
- Do not hardcode paths, thresholds or prompts in code — everything goes through config or
  prompt template files.

## 4. Repository layout

```
vlnav/
├── common/models.py         # Pose, ObjectNode, RoomNode (SYSTEM §3) — imported, not redefined
├── acquisition/
│   ├── __init__.py
│   ├── models.py            # Candidate, AcquisitionResult (acquisition-only models)
│   ├── memory_api.py        # MemoryView Protocol + InMemoryMemory (mock / fixture-backed)
│   ├── label_matcher.py     # LabelMatcher: target text -> matching node labels
│   ├── symbolic_filter.py   # SymbolicFilter: top-k candidates
│   ├── context_builder.py   # builds per-candidate visual + textual context
│   ├── vlm_client.py        # VLMClient Protocol + OpenAIVLMClient + MockVLMClient
│   ├── verifier.py          # NeuralVerifier: prompt, call, parse, validate
│   └── acquirer.py          # TargetAcquirer: orchestration, blacklist, fallback
│   └── prompts/
│       └── verify_target.j2 # Jinja2 prompt template
├── configs/
│   └── acquisition.yaml     # included by configs/system.yaml
├── eval/acquisition/
│   ├── make_fixture.py      # build a fixture memory from images + JSON, or from a saved SceneGraph
│   └── eval_acquisition.py  # offline evaluation on labeled cases
└── tests/
    ├── fixtures/acquisition/
    └── test_acquisition_*.py
```

## 5. Data models

`Pose`, `ObjectNode`, `RoomNode` come from `common.models` (SYSTEM §3; `ObjectNode` there has extra
fields such as `label_votes`, `points`, `best_view_cam_pose` — this package must only rely on the
fields it needs: `id, label, centroid, best_score, best_view_pose, best_view_image_path,
best_view_bbox, room_id, n_obs, first_seen_t`). Fixtures may leave the other fields at defaults.

Acquisition-only models (`acquisition/models.py`):

```python
class Candidate(BaseModel):
    node: ObjectNode
    match_score: float               # label/text similarity to target
    rank_score: float                # score used for top-k ranking
    forced: bool = False             # force-included (proposed by the planner VLM)

class AcquisitionResult(BaseModel):
    status: Literal["selected", "rejected_all", "no_candidates", "error"]
    node_id: int | None
    goal_pose: Pose | None           # = node.best_view_pose when selected
    candidates: list[int]            # ids that were sent to the VLM
    vlm_reason: str | None
    vlm_confidence: float | None
    latency_s: dict[str, float]      # {"filter": ..., "vlm": ..., "total": ...}
```

## 6. Memory interface (`memory_api.py`)

`MemoryView` is defined here and implemented by `memory.SceneGraph`, `memory.RemoteMemoryMirror`
(planner spec §4) and the test-only `InMemoryMemory`. Code **only** against the Protocol:

```python
class MemoryView(Protocol):
    def all_objects(self, until_t: float | None = None) -> list[ObjectNode]: ...
    def get_object(self, node_id: int) -> ObjectNode: ...
    def get_room(self, room_id: int) -> RoomNode | None: ...
    def neighbors(self, node_id: int, radius_m: float, same_room: bool) -> list[ObjectNode]: ...
```

- `until_t` supports time snapshots: only nodes with `first_seen_t <= until_t` are visible.
  This is required for open-loop evaluation on ROS bag replays.
- Provide `InMemoryMemory` loading from a JSON fixture (schema in Section 12) for tests and eval.
- `neighbors`: Euclidean distance between centroids `<= radius_m`, excluding the node itself,
  optionally restricted to the same room. Sorted by distance, capped at `max_neighbors` (config).

## 7. Stage 1 — Symbolic Filtering

### 7.1 LabelMatcher (`label_matcher.py`)

Problem: the target name produced by the planner (`"rain jacket"`) often does not equal the detector
label (`"jacket"`, `"coat"`, `"raincoat"`). String match is **not acceptable**.

```python
class LabelMatcher:
    def score(self, target: str, labels: list[str]) -> dict[str, float]  # cosine sim in [0, 1]
```

- Encode target and labels with a text encoder; cosine similarity; cache label embeddings
  (labels repeat across nodes). The embedder is a shared object (`TextEmbedder` protocol in
  `acquisition/label_matcher.py`), also injected into `memory.ObjectMap` association and the planner
  serializer — instantiate it once in `runtime/`.
- Optional synonym table in config (`synonyms: {"rain jacket": ["raincoat", "jacket", "coat"]}`);
  a synonym hit gets similarity `1.0`.
- A label matches if `sim >= label_sim_threshold` (config, default 0.75 — must be tuned, expose it).
- Prompt both sides with a template, e.g. `"a photo of a {}"`, configurable.

### 7.2 SymbolicFilter (`symbolic_filter.py`)

```python
class SymbolicFilter:
    def topk(self, target: str, memory: MemoryView, k: int,
             exclude: set[int], until_t: float | None) -> list[Candidate]
```

1. Collect nodes whose label matches (7.1), excluding `exclude` (blacklist).
2. Drop nodes with `best_score < min_det_score` (config).
3. Rank by `rank_score`. Implement a strategy switch in config:
   - `"confidence"` (paper default): `rank_score = best_score`
   - `"confidence_x_similarity"`: `rank_score = best_score * match_score`
4. Return top-k. Tie-break by `n_obs` desc, then `id` asc (deterministic).

## 8. Stage 2 — Neural Verification

### 8.1 ContextBuilder (`context_builder.py`)

For each candidate build:

- **Crop image**: bbox expanded by `crop_margin` (config, default 0.25 of bbox size), clamped to
  the image, resized so the long side `<= crop_max_px` (default 512).
- **Full-frame image** with the bbox drawn (thickness/color configurable), resized so long side
  `<= frame_max_px` (default 768). Make full-frame optional via config (`include_full_frame`).
- **Text context**: `node_id`, detector label, `best_score` (2 decimals), room label,
  neighbor labels with distances, e.g. `"on/near: black box (0.4 m), desk (0.9 m)"`.

Use **candidate letters** (`A`, `B`, `C`) in the prompt, not raw node ids, and keep a mapping
letter -> node_id. This prevents the VLM from inventing ids.

### 8.2 VLMClient (`vlm_client.py`)

```python
class VLMClient(Protocol):
    def chat(self, messages: list[dict], json_schema: dict | None,
             timeout_s: float) -> str: ...
```

- `OpenAIVLMClient`: OpenAI-compatible endpoint (base_url, model name from config). Structured
  output via `structured_output`: `response_format` (default, OpenAI-standard
  `response_format={"type": "json_schema", "json_schema": {...}}`) or `guided_json`
  (`extra_body={"guided_json": schema}`, older vLLM). The parameter names differ across vLLM versions
  — add an integration test against the installed version (AUDIT pitfall 23). Always also parse
  robustly. `temperature=0`. Cache responses by prompt hash for offline evaluation (AUDIT 27).
- `MockVLMClient`: returns scripted responses (for tests), can simulate timeouts and malformed JSON.

### 8.3 Prompt (`prompts/verify_target.j2`)

Must contain, in this order:
1. Role: the robot must find an object that satisfies the instruction.
2. The original instruction **verbatim** and the target name.
3. Explicit criteria: the object must match the intended meaning of the instruction (e.g. rain
   jacket = waterproof outerwear, not any jacket); use visual attributes and spatial context
   (room, nearby objects); detector labels may be wrong.
4. For each candidate letter: crop image, (optional) full-frame image, text context.
5. Instruction to answer `"none"` if no candidate is a convincing match.
6. Output only JSON matching the schema below.

### 8.4 Output schema

```json
{
  "type": "object",
  "properties": {
    "reasoning": {"type": "string"},
    "choice": {"type": "string", "enum": ["A", "B", "C", "none"]},
    "confidence": {"type": "number", "minimum": 0, "maximum": 1}
  },
  "required": ["reasoning", "choice", "confidence"]
}
```

The enum must be generated dynamically from the actual number of candidates.

### 8.5 NeuralVerifier (`verifier.py`)

```python
class NeuralVerifier:
    def verify(self, instruction: str, target: str,
               candidates: list[Candidate], memory: MemoryView) -> VerifyOutput
```

- Parse JSON robustly (strip code fences; on failure retry once with a "return valid JSON only"
  message; on second failure return error).
- Validate `choice` is one of the offered letters; anything else → treat as error.
- If `choice != "none"` but `confidence < min_vlm_confidence` (config) → treat as `none`.
- Log full prompt (minus base64), raw response, parsed output to a JSONL debug log.

## 9. Orchestration — TargetAcquirer (`acquirer.py`)

```python
class TargetAcquirer:
    def acquire(self, instruction: str, target: str, memory: MemoryView,
                until_t: float | None = None,
                force_include: list[int] | None = None) -> AcquisitionResult
    def blacklist(self, target: str, node_id: int) -> None   # planner: after nav_failed / goto timeout
    def reset_blacklist(self, target: str | None = None) -> None
```

`force_include`: node ids proposed by the planner VLM (planner spec §5.6). Valid, non-blacklisted
ids are added to the round-0 candidate set (marked `forced`, may make it k+1 candidates); they are
still verified like any other candidate.

Algorithm:

```
blacklist = self._blacklist[target]          # per-target set of rejected node ids
for round in range(max_rounds):              # config, default 2  (k, then next k)
    cands = SymbolicFilter.topk(target, memory, k, exclude=blacklist, until_t)
    if not cands:
        return status="no_candidates" (round 0) or "rejected_all" (round > 0)
    out = NeuralVerifier.verify(instruction, target, cands, memory)
    if out.error: return status="error"
    if out.choice is a node:
        return status="selected", goal_pose = node.best_view_pose
    blacklist |= {c.node.id for c in cands}
return status="rejected_all"
```

Rules:
- Goal is `best_view_pose`, **not** the centroid (the pose is known reachable and the target is
  visible from it).
- Blacklist is per target and persists across calls until `reset_blacklist`. A node rejected for
  `"rain jacket"` can still be a candidate for `"umbrella"`.
- Must be deterministic given the same memory, config and VLM responses.
- Record latencies for filter, VLM and total.

## 10. Configuration (`configs/acquisition.yaml`)

```yaml
symbolic:
  k: 3
  rank_strategy: confidence            # confidence | confidence_x_similarity
  min_det_score: 0.15
  label_sim_threshold: 0.75
  text_encoder: siglip                 # siglip | clip_vitb32 | sentence_transformers
  text_encoder_path: checkpoints/text_encoder/siglip-base-patch16-224
  text_template: "a photo of a {}"
  synonyms: {}
context:
  crop_margin: 0.25
  crop_max_px: 512
  include_full_frame: true
  frame_max_px: 768
  neighbor_radius_m: 1.5
  neighbor_same_room: false
  max_neighbors: 6
vlm:
  base_url: http://localhost:8000/v1
  model: checkpoints/vlm/Qwen3-VL-8B-Instruct   # must be an *Instruct* model, not Thinking
  timeout_s: 30
  structured_output: response_format   # response_format (OpenAI json_schema) | guided_json
  min_vlm_confidence: 0.5
acquirer:
  max_rounds: 2
logging:
  debug_jsonl: logs/acquisition_debug.jsonl
```

## 11. Milestones (A-IDs referenced in SYSTEM §9; each ends with green tests)

1. **A1 MemoryView + InMemoryMemory + JSON fixture loader** (models from `common.models`). Tests:
   snapshot filtering by `until_t`, neighbor query correctness.
2. **A2 LabelMatcher + SymbolicFilter.** Stub embedder (deterministic vectors): synonym hit,
   threshold, blacklist exclusion, both rank strategies, deterministic tie-break, `force_include`.
3. **A3 ContextBuilder.** Crop clamping at borders, resize limits, bbox drawing, letter mapping,
   neighbor text formatting.
4. **A4 VLMClient + NeuralVerifier** with `MockVLMClient`: valid choice, `none`, low confidence →
   none, malformed JSON → retry → success, malformed twice → error, invalid letter → error, timeout.
5. **A5 TargetAcquirer.** Selected on round 0, rejected → next k, no candidates, blacklist
   persistence/reset, external `blacklist()`, returned pose is `best_view_pose`.
6. **A6 Integration (GPU)**: `eval/acquisition/eval_acquisition.py` against a live vLLM server on
   (a) a hand-made fixture, then (b) a `SceneGraph` saved from a real bag (planner milestone P3),
   loaded through `make_fixture.py --from-scene-graph`. `MemoryView` conformance tests are shared
   with `memory/` (same test module parametrized over implementations).

## 12. Fixtures & evaluation

### Fixture memory JSON

```json
{
  "rooms": [{"id": 0, "label": "bedroom"}],
  "objects": [{
    "id": 17, "label": "jacket", "centroid": [3.1, 4.2, 1.0], "best_score": 0.62,
    "best_view_pose": {"position": [2.0, 4.0, 0.0], "orientation_xyzw": [0,0,0,1], "stamp": 120.5},
    "best_view_image_path": "images/000231.jpg", "best_view_bbox": [210, 80, 390, 400],
    "room_id": 0, "n_obs": 4, "first_seen_t": 118.0
  }]
}
```

### Eval cases (`eval_cases.yaml`)

```yaml
- instruction: "Today's weather report indicates rain. Help Rob find a jacket."
  target: "rain jacket"
  gt_node_ids: [17]          # empty list = target not present, correct answer is no selection
  until_t: null
```

### Metrics (eval script prints a table + writes CSV)

- `symbolic_recall@k`: GT node in top-k candidates.
- `verification_acc`: given GT in top-k, VLM picks it.
- `end_to_end_acc`: selected node in `gt_node_ids` (or correctly no selection when empty).
- `false_commit_rate`: selected a node when `gt_node_ids` is empty or selected a wrong node.
- Mean / p95 latency of VLM call.
- Ablation flags on CLI: `--k`, `--rank-strategy`, `--no-full-frame`, `--no-neighbors`,
  `--no-vlm` (top-1 of symbolic filter is taken as the answer).

## 13. Acceptance criteria

- All unit tests pass without network or GPU (`pytest -q`).
- No module imports ROS 2; no module imports `exploration/`.
- Every threshold/prompt is configurable; no magic numbers in code.
- The VLM can never cause selection of a node that was not among the offered candidates.
- `eval_acquisition.py` runs end-to-end on a fixture against a live vLLM server and outputs the
  metrics above.
- Debug JSONL log lets a human reproduce each decision (candidates, context text, raw VLM output).

## 14. Open decisions (keep configurable, do not silently pick one)

- Rank strategy (`confidence` vs `confidence_x_similarity`).
- Neighbor definition (radius vs same room).
- Whether to send full-frame images in addition to crops (VRAM / latency trade-off).
- Behavior after all rounds rejected: currently return `rejected_all`; caller decides to explore.
