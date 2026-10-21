# TASK: Code-as-policy multi-agent system (Planner / Coder / Observer) for VLN

**Read first:** `SYSTEM.md` for the shared models and runtimes, and `agentic_cap_vln_plan.md` §3 for the
primitive list. This task is the concrete build plan for that idea. It uses the role split of
**GraspMAS** (Fsoft-AIC, IROS 2025) in place of the single tool-using agent of `agentic_cap_vln_plan.md` §4.
Choices that GraspMAS does not fix are marked **[DESIGN]**. Facts that must be measured before relying
on them are marked **[VERIFY]**.

---

## 1. What GraspMAS does (read from `github.com/Fsoft-AIC/GraspMAS`, 20/10/2026)

| Part | GraspMAS | File |
|---|---|---|
| Loop | `for round in range(max_round)`: Planner → Coder → `exec` → Observer; stop when the plan is "Return to user" | `agents/graspmas.py` |
| Planner | Text LLM (gpt-4o, T=0.8). Input: query, previous plan, **Observer summary only**. Output `<thought>…</thought><plan>Step 1… Step n</plan>`. Few-shot examples are whole multi-round episodes (plan → code → observer summary → next plan) | `agents/planner.py`, `agents/prompt/planner_prompt.py` |
| Coder | Text LLM (T=0.1). Input: plan + docstring of the tool class + examples. Output `def execute_command(image):` | `agents/coder.py`, `agents/prompt/coder_prompt.py` |
| Tools | One class `ImagePatch`: `find`, `find_part`, `exists`, `verify_property`, `simple_query`, `llm_query`, `compute_depth`, `grasp_detection`, plus box attributes for spatial logic | `image_patch.py` |
| Observer | VLM. Input: result JSON (grasp, error log) + **visualisation image**. Output JSON `{verdict, checklist{target_match, semantic_alignment, fragile_overlap, collision_risk}, error_logs, summary}`; only `summary` goes back to the Planner | `agents/observer.py`, `agents/prompt/observer_prompt.py` |

**What to keep:**
- Separate roles with separate prompts. The Planner reasons in words, the Coder only translates, the
  Observer only judges from what it sees.
- The Observer checklist: a fixed set of yes/no/uncertain checks, then a short summary for the Planner.
- Few-shot examples written as whole episodes that include a failed round and its repair.
- The tool class as the only interface, documented by its docstring inside the Coder prompt.

**What not to copy:**
- `exec(code, globals())` runs outside any `try`, with the whole module namespace and no checks.
- State lives on the agent object (`self.plan`, `self.observation`) and is never reset between
  queries. `main_batch.py` runs 4 queries concurrently on **one** instance (`asyncio.gather`), so
  they overwrite each other's plan and observation.
- The code is parsed by dropping the first and last line of the reply, and the observation is parsed
  with `split` and `json.loads`. One bad reply crashes the episode.
- The execution is free and repeatable: one static image, no side effects. **A robot is not.**
  Executing a program moves the robot, takes minutes, and changes what the robot sees. See §2.

## 2. Main difference: two loops **[DESIGN]**

GraspMAS can run the program each round because running it costs nothing. We split its loop in two:

```
             ┌────────────── deliberation loop (no robot motion, ≤ max_round) ──────────────┐
instruction ─► Planner ─► Coder ─► validate ─► dry_run (mock world) ─► Observer ─► VALID? ──┼─► accepted program
             └────────────▲─────────────────────────────────────────────────────────┘ no   │
                          │                                                                 ▼
                          │ summary         execution loop (robot, ~2 Hz frames)    Program Runtime
                          └──────── Observer ◄── checkpoint / failure event + trace ◄── (primitives → goals)
```

- **Deliberation loop**: this is GraspMAS's loop, with `exec` replaced by `dry_run`. The program runs on a
  symbolic world built from the current scene graph and occupancy grid (`agentic_cap_vln_plan.md` §4). The
  Observer judges the dry-run trace and a top-down render of the planned goals.
- **Execution loop**: the accepted program runs on the real stack. The Observer is called only at
  **checkpoints** (an explicit `api.check(...)` in the program, the end of the program) and on
  **failures** (`nav_failed`, `timeout`, `TaskFailed`, an exception). Its summary starts a repair: a
  new deliberation loop for the **rest** of the program, from the current state.
- While the agents deliberate, the robot keeps running the NeSy layer: explore the current target, or
  hold. It never waits idle on an LLM.

## 3. Shared state (blackboard) **[DESIGN]**

One read-only snapshot per decision. Agents never write to it; they return proposals. Only the Program
Runtime commits.

```python
@dataclass(frozen=True)
class WorldSnapshot:
    version: int
    stamp: float
    task: TaskState            # instruction, targets, reported / failed, done
    robot: Pose
    nav: NavStatus | None      # Nav2 action state for the current goal; None in bag mode
    grid: GridView             # read-only reference + version (OccupancyGrid2D), not a copy
    memory: MemoryView         # UnifiedMemory.snapshot(t)
    program: ProgramState      # source, current primitive (line), trace so far
    pending: Proposal | None   # next program: validated + dry-run, waiting for a swap
    events: tuple[Event, ...]  # since the previous snapshot

@dataclass(frozen=True)
class Proposal:
    program: str
    base_version: int          # snapshot the agents reasoned on
    rounds: list[RoundLog]     # plan / code / dry-run / observation per round
```

| Field | Written by | Read by |
|---|---|---|
| `grid`, `memory` | mapping, perception + memory (every frame) | primitives, `dry_run`, agents (as text) |
| `robot`, `nav` | ROS node / `NavigationAction`, bag source | runtime, Observer |
| `task`, `program` | Program Runtime only | agents |
| `pending` | agent orchestrator (`mas.py`) | runtime: swap or reject |
| `events` | runtime, perception, nav | orchestrator: which agent to call |

**Swap rule:** a proposal is swapped in only at a primitive boundary. If events since `base_version`
invalidate it (a target was reported, a goal cell is now blocked), it is dry-run again on the new snapshot
before the swap. If that dry-run fails, the proposal goes back to the agents.

**One episode = one state object.** Agents are stateless functions of `(snapshot, history)`. This removes
the GraspMAS concurrency bug, and lets an eval run several tasks in parallel.

## 4. Agents

All three agents use `VLMClient.chat` (`vlnav/acquisition/vlm_client.py`). That gives local Qwen3-VL and
OpenAI-compatible backends, the response cache and logging. `vlnav/agent/llm.py` (the copy of GraspMAS's
client) is removed.

**Backends, set per agent [DESIGN]:**

| Agent | Default | Why |
|---|---|---|
| Planner | cloud API model | Implicit targets, conditions and fallbacks are where Qwen3-VL-8B is weak (planner evals: armchair, TV) |
| Coder | cloud API model | A program that does not validate wastes a round; API models are far more reliable at writing code from a docstring alone |
| Observer | cloud API at M5, then try local Qwen | Gets camera images; local keeps images on the machine |
| `check()` in programs | local Qwen | Many calls, images only, short yes/no answers |

```yaml
# configs/agent.yaml (sketch)
planner:  {vlm: {backend: openai, base_url: <provider>, model: <pinned version>, api_key_env: OPENAI_API_KEY}}
coder:    {vlm: {backend: openai, base_url: <provider>, model: <pinned version>, api_key_env: OPENAI_API_KEY}}
observer: {vlm: {backend: local}}            # overrides on configs/vlm.yaml
max_round: 3
fallback: nesy                               # API error / timeout / no network -> PlannerCore for this task
```

- **Key:** `VLMConfig.api_key_env` names the environment variable that holds the key (done). There is
  no key file in the repo, unlike GraspMAS's `api.key`.
- **Reproducibility:** pin the exact model version; temperature 0; `cache_dir` on; log every transcript.
- **Fallback:** an API error or timeout after one retry switches the task to the NeSy `PlannerCore`, which
  runs fully local. A robot without network must still work.
- **Data leaving the machine:** the Planner and Coder send text only (instruction, scene-graph labels and
  positions). The Observer and `check()` send camera images, so they stay local wherever images must not
  leave the site.
- **Local ablation:** the same system with Qwen3-VL-8B as Planner and Coder is always part of the eval
  (§9).

Prompts live in `prompt/` as Jinja strings, the same way as `prompt/planner_prompt_v1.py`: a system turn,
few-shot turns, then the request. Every reply is checked against a JSON schema or tag format. On a bad
reply the agent retries once with the parse error, then falls back (§4.4).

### 4.1 Planner (`agent/planner.py`, `prompt/cap_planner_v1.py`)
- **Input:** instruction; scene-graph text (the same rendering as the NeSy planner prompt); robot room and
  pose; the previous plan; the Observer summary; the steps already done (from `program.trace`).
- **Output (JSON):** `{"thought": str, "plan": [str, ...], "done": bool}`. Each step names one skill
  (`find`, `explore`, `go_to`, `follow`, `rotate`, `check`, `report`, `stop`). `done` means "return to user".
- **Rules, as in GraspMAS:** decide from the Observer summary; do not repeat a failed step unchanged; on a
  repair, plan only the remaining part.
- Temperature 0 by default **[DESIGN]**. GraspMAS uses 0.8, but we need reproducible evals, and the
  response cache needs determinism.

### 4.2 Coder (`agent/coder.py`, `prompt/cap_coder_v1.py`)
- **Input:** the plan, the `NavAPI` docstring (generated from the code; §5), and code examples.
- **Output:** one fenced block containing a plain function `def task(api):`, with no generators and no
  `yield` (§5.1). Each plan step is a `# Step k` comment. The comments let the runtime map a line of the
  trace back to a plan step.
- Parsing: extract the fenced block, `ast.parse`, then `validate` (§6). A validation error goes back to
  the Coder once, with the message, before the round counts as failed.

### 4.3 Observer (`agent/observer.py`, `prompt/cap_observer_v1.py`)
- **Input, deliberation:** the instruction; the dry-run trace (goals, the predicates assumed true, failure
  points); a top-down render of the planned path and goals over the grid, with the scene-graph objects
  labelled. Use `exploration/viz.py` for the render.
- **Input, execution:** the trace since the last checkpoint; the current camera image annotated with the
  detections (the annotated images already exist); the top-down map with the driven path; the error or
  event.
- **Output (JSON):**
  ```
  {"verdict": "VALID|INVALID",
   "checklist": {"target_match": "yes|no|uncertain",      // the right object / place
                 "spatial_relation": "yes|no|partial",    // left/right/near/in front of as asked
                 "route_progress": "yes|no|uncertain",    // follows the route part of the instruction
                 "goal_reachable": "yes|no",              // goals on known free cells, path exists
                 "stuck_or_loop": "yes|no"},
   "error_logs": "...", "summary": "..."}
  ```
  Only `summary` and `verdict` go to the Planner, as in GraspMAS.
- **The verdict is not trusted alone.** Hard checks come from code: the goal is reachable on the grid, the
  target node exists. Code can flip a VALID to INVALID, never the reverse.

### 4.4 Orchestrator (`agent/mas.py`)
`deliberate(snapshot, failure=None) -> Proposal | None` runs up to `max_round` rounds of Planner → Coder
→ validate → dry_run → Observer. It stops at the first VALID round, or when the Planner says `done`.
- After `max_round` without a VALID round, it returns the best round (fewest hard-check failures).
- If no program validates, it returns `None`. The runtime then falls back to the NeSy `PlannerCore` for
  this task; this is also a baseline.

Every round is logged (`rounds.jsonl`) with prompts, replies, code, trace, observation, tokens and latency.

## 5. `NavAPI`: the tool class (counterpart of `ImagePatch`)

There are two implementations with the **same signatures and docstrings**: `RealNavAPI` (runtime) and
`MockNavAPI` (dry-run). The Coder prompt shows the docstring of the shared base class, so the two cannot
drift apart. The list refines `agentic_cap_vln_plan.md` §3. It is kept to about 12 calls, because the
Coder learns the whole API from one docstring (GraspMAS's `ImagePatch` has 8 methods).

### 5.1 Rules [DESIGN]
1. **Queries vs actions.** Queries read the snapshot and return at once; the robot does not move. Actions
   move or look, take time, and can fail. `MockNavAPI` reuses the real queries and only fakes the actions,
   which keeps dry-run honest.
2. **Plain blocking calls, not generators.** With generators, `api.go_to(x)` written without `yield from`
   creates a generator object and does nothing, with no error. An 8B Coder makes this mistake easily.
   Instead, the program runs in its own thread, and each action blocks until the runtime posts that
   action's end event (§6).
3. **Typed handles, never raw ids.** Queries return `Obj` / `Room`, and actions take them. The model
   cannot invent node ids; the planner already has to repair invented `#ids`.
4. **Failure is a return value** (`None` / `False`), handled with an `if`. Exceptions are only for
   `BudgetExceeded` and `Cancelled` (swap, task cancelled); the program must not catch them.
5. **Directions are relative to the robot at the time of the call.** `side()` and `turn()` use the
   heading when they are called.

### 5.2 Data (read-only)
- `Obj`: `name`, `room` (`Room | None`), `xy`, `height`, `score`, `n_obs`, `seen_s_ago`.
- `Room`: `label`, `xy`, `area`, `objects`.

### 5.3 Queries (instant)

| Call | Returns | Real implementation | Mock |
|---|---|---|---|
| `find(name, room=None, near=None, radius=3.0)` | `list[Obj]`, best match first | `MemoryView.find_by_text` + `SymbolicFilter`, matched on the target's detector prompts | same, on the snapshot |
| `rooms(label=None)` | `list[Room]` | `MemoryView.rooms` | same |
| `here()` | `xy`, `yaw`, `room` | robot pose + room lookup | simulated pose |
| `distance(a, b=None)` | metres along a path on the grid, `None` if no path; `b=None` → from the robot | `GridAStar.cost_to` | same |
| `side(x)` | `"left"`, `"right"`, `"ahead"` or `"behind"` | **new**: bearing in the robot frame | same, simulated pose |

### 5.4 Actions (move or look; can fail)

| Call | Returns | Real implementation | Mock |
|---|---|---|---|
| `explore(name, prompts=None, room=None, timeout=120)` | `Obj \| None` | `Subtask(explore)` → `ExplorationCore` + candidate check + verifier | known object of that name → found; else traced as "unknown" |
| `go_to(target, timeout=60)` | `bool`; `target` is an `Obj`, a `Room` or `here()` | `Subtask(goto)`; standoff `_goto_approach`. `Room` → **new**: a free cell inside the room | A* on the grid; fails on an unknown or blocked goal |
| `follow(instruction, timeout=120)` | `bool` | `Subtask(move)` → `MoveCore` (DualVLN) | success, marked "unverified" |
| `turn(deg)` | `bool`, positive = left | goal pose with yaw; **needs** `goto_yaw_tol_rad` set | pose update |
| `look_around()` | `list[Obj]` newly seen | **new**: 4 turns of 90° | known objects within range of the simulated FoV |
| `check(question, obj=None)` | `True \| False \| None` | **new** yes/no VLM prompt on the current view, or the best view of `obj`; the only VLM call in a program; checkpoint for the Observer (§2) | `None` → both branches traced |
| `report(obj)` | — | `TargetReport` | trace entry |
| `fail(reason)` | — | task failed with the reason | trace entry |

Python does the rest: sorting, filtering by `side`, conditions, counting.

### 5.5 Left out on purpose
- Velocity or raw pose commands: all motion goes through the nav stack, which is the safety boundary.
- Pixels and images: route following from images is DualVLN's job, behind `follow`.
- Writes to the map or memory: only perception writes (§3).
- Open-ended reasoning in code ("watch football" → TV): that is the Planner's job, in the plan text.

### 5.6 Example (`eval/planner/gt_vr_h3_2_move.yaml`)
```python
def task(api):
    # Step 1: follow the route; if DualVLN fails, go to the landmark object instead
    if not api.follow("walk past the beige sofa on your left and stop near the high bar chairs"):
        chairs = api.find("bar chair")
        if not chairs or not api.go_to(chairs[0]):
            return api.fail("bar chairs not reached")
    # Step 2: find an armchair (memory first, then explore)
    seats = api.find("armchair")
    seat = seats[0] if seats else api.explore("armchair", prompts=["armchair", "sofa chair"])
    if seat is None:
        return api.fail("no armchair")
    # Step 3: go there and report
    if api.go_to(seat):
        api.report(seat)
```
Falling back from a failed route to its landmark cannot be expressed in the current JSON plans.

## 6. Safe execution (`agent/validate.py`, `agent/runtime.py`)
- **Validate (AST):** only `def task(api)`; the only names allowed are `api`, the literals, and the
  builtins `len, min, max, sorted, abs, range, enumerate, any, all`; no `import`, attributes starting with
  `_`, `exec/eval/open/getattr`, `while True`, or `lambda` outside `until=`. Every loop has a literal bound.
- **Run:** `exec` in a fresh namespace with `{"__builtins__": SAFE_BUILTINS}`, **never** `globals()`.
  `task(api)` runs in its own thread. Each action posts a request to the runtime and blocks on a queue
  until the runtime posts the result: the action's end event (`reached`, `nav_failed`, `timeout`,
  verified candidate…). The runtime thread stays the only one that touches the planner, exploration and
  move cores. Queries read the snapshot of the current frame.
- **Cancel and swap:** the runtime answers a blocked action with `Cancelled`, which raises in the
  program thread. Swaps happen only between two actions (§3). Any other exception → failure event with
  the line number.
- **Mock:** `MockNavAPI` runs the same function synchronously in the agent process, with a step limit.
- **Budget:** `max_primitives`, `max_task_s` and `max_rounds_per_task` in `configs/agent.yaml`.

## 7. Package layout
```
vlnav/agent/
  state.py        WorldSnapshot, Proposal, ProgramState, Event (pydantic, frozen)
  api.py          NavAPI base, Obj / Room, Coder-prompt docstring built from the code
  real_api.py     RealNavAPI on PlannerCore / ExplorationCore / MoveCore / acquisition
  mock_world.py   MockNavAPI + FoV ray cast + A* on the grid
  validate.py     AST checks
  runtime.py      ProgramRuntime: program thread, action queue, events, cancel and swap
  planner.py  coder.py  observer.py  mas.py
  config.py       AgentConfig (pydantic, extra="forbid")
prompt/cap_planner_v1.py  prompt/cap_coder_v1.py  prompt/cap_observer_v1.py
configs/agent.yaml
scripts/run_cap_bag.py     bag replay, shadow execution (BagEventSim semantics), Rerun tab "agent"
eval/cap/                  task set + metrics
tests/agent/
```
`vlnav/agent/cap.py` and `vlnav/agent/planner.py` (both empty) are replaced. `vlnav/agent/llm.py` is removed.

## 8. Milestones

| M | Content | Gate |
|---|---|---|
| M0 | `state.py`, `api.py` (signatures and docstrings only), `validate.py`; 10 hand-written programs from `eval/planner/gt_*.yaml` | the docstring renders; validation accepts the 10 programs and rejects a list of bad programs |
| M1 | `mock_world.py`: dry-run on a saved snapshot of `vr_h3_2` (memory JSON + grid) | the hand-written programs give the expected traces; unknown-cell goals are flagged |
| M2 | `runtime.py` + `real_api.py`, run in `run_cap_bag.py` with **hand-written programs, no LLM** | on the bag, each program dispatches the same subtasks as the NeSy planner does for the same task |
| M3 | Coder alone: hand-written plans → code, with an API model and with local Qwen3-VL-8B | API model: ≥ 90 % of the programs validate and dry-run like the hand-written ones. The local rate decides whether the local model is only an ablation or also the fallback Coder |
| M4 | Planner + Coder, open loop (one round, no Observer) | task success on the bag ≥ the NeSy planner on `eval/planner` tasks |
| M5 | Observer + deliberation loop (`max_round` 3) | success improves over M4; record rounds per task, tokens, latency |
| M6 | Execution checkpoints, failure repair, proposal swap | repaired tasks counted; no swap in the middle of a primitive (test) |
| M7 | ROS node integration (needs `NavStatus`); robot runs | 10 real tasks with logs |

## 9. Evaluation
- **Tasks:** `eval/planner/gt_vr_h3_2.yaml` and `gt_vr_h3_2_move.yaml`, plus about 10 new tasks that need
  program structure (spatial relations, conditions, counting). Example: "if the kitchen has a kettle, go
  there, else find a cup". No task copies a few-shot example (test, as in `test_prompt_examples.py`).
- **Metrics:** success, target precision and order (`eval/planner/metrics.py`); validate rate; dry-run pass
  rate; rounds per task; repairs per task; tokens and latency per agent.
- **Baselines:** NeSy `PlannerCore` (few-shot v1); CaP with one round and no Observer (M4).
- **Ablations:** no dry-run (Observer judges the code only); no Observer image; Planner and Coder merged
  into one agent; local Qwen3-VL-8B as Planner and Coder (fully offline).

## 10. Risks
| Risk | Mitigation |
|---|---|
| An 8B local model writes poor code | API model by default for the Planner and Coder (§4); local measured at M3 |
| API outage, rate limit or no network on the robot | One retry, then the NeSy `PlannerCore` takes the task (§4) |
| Camera images sent to a provider | Only the Observer and `check()` see images; keep them local where needed |
| API model changes between runs | Pin the version; response cache; log transcripts |
| The Observer says VALID on a wrong plan | Hard checks from code override it (§4.3); report the Observer's false-VALID rate on dry-runs |
| Deliberation is too slow for a moving robot | The NeSy layer keeps the robot busy (§2); log latency per round; cap `max_round` |
| The mock world is too optimistic | Report "dry-run passed, real run failed" as its own metric |
| One snapshot is used for 30 s of reasoning | `base_version` + re-dry-run before the swap (§3) |

## 11. Implementation status (21/10/2026) and deviations

Built: `vlnav/agent/` (config, api, validate, state, real_api, mock_world, chat, planner, coder, observer,
mas, runtime), `prompt/cap_{planner,coder,observer}_v1.py`, `configs/agent.yaml`,
`configs/runs/vr_h3_2_ros2_cap.yaml`, `eval/planner/run_openloop.py --agent`, `tests/agent/`.
Covers M0, M1, M2 and M5/M6 logic with scripted agents. Not measured yet: M3/M4/M5 gates on the bag.

**Main rule: three task types.** Every part of an instruction becomes one of three subtasks:
- movement command or route ("go forward", "turn right at the box") → `follow` → `move` (DualVLN);
- object already in the scene graph and confirmed by the VLM → `locate` returns it → `goto`, no explore;
- object unknown (or every known match rejected) → `locate` explores → `explore`, then verify, then `goto`.

Deviations from §4-§7:
| Plan | Built | Why |
|---|---|---|
| `explore` + Python `find` to choose memory vs explore (§5.6) | new action `locate(name, prompts, room, timeout)`: verify scene-graph matches with the acquirer, else explore with the candidate check → verify loop; `explore` stays (ignores known objects) | the goto / explore choice is one call, and a known object is never used unverified |
| `turn(deg)`, `look_around()` | not built | `turn` needs `goto_yaw_tol_rad` in the nav stack (no Nav2 / ROS changes for now); `follow("Turn left.")` covers it |
| `check` on the current view; mock traces both branches | `check` on the object's best view; the mock assumes "yes" (the success path) | no current-view image in the runtime yet; both-branch tracing later |
| Observer gets a top-down render and camera images | text only: dry-run trace + hard checks | M5 first with text; images are an ablation |
| NeSy layer keeps exploring during deliberation (§2) | the robot holds (no active subtask) while agents deliberate or verify | needs a second subtask owner; later |
| Proposal swap during execution + re-dry-run on `base_version` (§3) | no swap; repair only after a program failure (`max_repairs`) | M6 swap later; `base_version` is logged |
| `Obj` fields (§5.2) | plus `target` (what it was searched as, used by `report`) and `verified`; `Place` from `here()` | report and metrics need the target name |
| `max_task_s`, `max_rounds_per_task` | `planner.task_timeout_s`, `max_round`, `max_repairs`, `max_primitives` | reuse the planner timeout |
| `scripts/run_cap_bag.py`, `eval/cap/` | `eval/planner/run_openloop.py --agent` (same metrics: plan order from the dry-run actions) | one eval path for both planners |
| `vlnav/agent/chat.py` | added: shared prompt loading + JSON repair retry for the three agents | |

Hard checks (`mock_world.dry_run`): exception (with line), action budget, no action, `follow` with move
disabled, a known goal with no path, `report` before a successful `go_to`, `fail` reached on the success
path, a movement command in the instruction (regex) with no `follow` call.

Runtime: `ProgramRuntime` has PlannerCore's public interface, so SystemRuntime, LiveRuntime and the ROS
node run it unchanged (`agent.enabled` in the run config, `vlnav/runtime/builder.py`). No valid program on
the first deliberation → the NeSy PlannerCore takes the task (`fallback: nesy`).
