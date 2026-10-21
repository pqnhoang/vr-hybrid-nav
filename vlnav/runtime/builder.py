"""Build shared perception, mapping, memory and planning components from run-config overrides.

`build_components` supports live adapters; `build_system` adds a bag source and BagEventSim for
offline mode A. Also shared by exploration replay and planner open-loop evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from vlnav.acquisition.acquirer import TargetAcquirer
from vlnav.agent.config import AgentConfig, load_agent_config
from vlnav.acquisition.config import load_acquisition_config
from vlnav.acquisition.label_matcher import LabelMatcher, make_text_embedder
from vlnav.acquisition.vlm_client import VLMClient, load_vlm_config, make_vlm_client
from vlnav.common.config import deep_merge
from vlnav.exploration import ExplorationCore, load_exploration_config
from vlnav.io.sources import BagSource, load_bag_config, resolve
from vlnav.mapping.config import MappingConfig, load_mapping_config
from vlnav.mapping.occupancy import OccupancyGrid2D
from vlnav.memory import ImageStore, RoomLabeler, UnifiedMemory, load_memory_config
from vlnav.move import CachedS2Client, HttpS2Client, MoveCore, S2Client, load_move_config
from vlnav.navigation import GridAStar, NavigationConfig, load_navigation_config
from vlnav.perception.config import load_checkpoint_paths, load_perception_config
from vlnav.perception.detector import YoloWorldFastSam
from vlnav.perception.pipeline import PerceptionPipeline, VocabularyManager, load_vocab_file
from vlnav.planner.config import load_planner_config
from vlnav.planner.config import PlannerConfig
from vlnav.planner.core import PlannerCore
from vlnav.planner.serializer import GraphSerializer
from vlnav.runtime.system_runtime import SystemRuntime
from vlnav.sim.bag_event_sim import BagEventSim, load_sim_config


@dataclass
class System:
    source: BagSource
    perception: PerceptionPipeline
    memory: UnifiedMemory
    matcher: LabelMatcher
    acquirer: TargetAcquirer
    planner: PlannerCore                     # or agent.runtime.ProgramRuntime (agent.enabled), same interface
    sim: BagEventSim
    vlm: VLMClient
    grid: OccupancyGrid2D
    mapping: MappingConfig
    exploration: ExplorationCore
    move: MoveCore | None = None             # move.enabled

    def runtime(self, **kw) -> SystemRuntime:
        return SystemRuntime(self.source, self.perception, self.memory, self.planner, self.sim,
                             self.grid, self.mapping.depth_points, self.exploration, move=self.move, **kw)


@dataclass
class Components:
    """Shared pipeline without a bag source or simulated navigation events."""
    perception: PerceptionPipeline
    memory: UnifiedMemory
    matcher: LabelMatcher
    acquirer: TargetAcquirer
    planner: PlannerCore                     # or agent.runtime.ProgramRuntime (agent.enabled), same interface
    vlm: VLMClient
    grid: OccupancyGrid2D
    mapping: MappingConfig
    exploration: ExplorationCore
    move: MoveCore | None = None             # move.enabled


def build_perception(rc: dict, detector: YoloWorldFastSam | None = None) -> PerceptionPipeline:
    pcfg = load_perception_config(overrides=rc.get("perception"))
    vocab = VocabularyManager(load_vocab_file(resolve(pcfg.vocabulary.generic_vocab_file)), pcfg.dynamic.labels)
    det = detector or YoloWorldFastSam(pcfg.detector, load_checkpoint_paths())
    return PerceptionPipeline(det, vocab, pcfg.lifting, pcfg.dynamic)


def room_labels_near(memory: UnifiedMemory):
    """Exploration's injected room lookup (`scoring.room_bonus`): label of the room at xy plus the
    labels of its adjacent rooms; exploration itself never imports memory."""
    def lookup(xy) -> set[str]:
        rid = memory.tracker.room_at(xy, memory.cfg.rooms.room_assign_fallback_m)
        rooms = {r.id: r for r in memory.rooms()}
        if rid not in rooms:
            return set()
        r = rooms[rid]
        return {r.label} | {rooms[a].label for a in r.adjacent_room_ids if a in rooms}
    return lookup


def build_exploration(rc: dict, grid: OccupancyGrid2D, memory: UnifiedMemory | None = None) -> ExplorationCore:
    ecfg = load_exploration_config(overrides=rc.get("exploration"))
    ncfg = load_navigation_config(overrides=rc.get("navigation"))
    rooms = room_labels_near(memory) if memory is not None and ecfg.scoring.room_bonus.enabled else None
    return ExplorationCore(ecfg, grid, GridAStar(ncfg), ncfg.robot_radius_m, rooms)


def build_move(rc: dict, grid: OccupancyGrid2D, s2: S2Client | None = None) -> MoveCore:
    """`s2` defaults to the HTTP client of `move.s2` (cached when `cache_dir` is set)."""
    mcfg = load_move_config(overrides=rc.get("move"))
    if s2 is None:
        s2 = HttpS2Client(mcfg.s2.base_url, mcfg.s2.timeout_s)
        if mcfg.s2.cache_dir:
            s2 = CachedS2Client(s2, resolve(mcfg.s2.cache_dir))
    return MoveCore(mcfg, grid, GridAStar(load_navigation_config(overrides=rc.get("navigation"))), s2)


def build_matcher(acfg) -> LabelMatcher:
    sym = acfg.symbolic
    return LabelMatcher(make_text_embedder(sym.text_encoder, sym.text_encoder_path, sym.device),
                        sym.text_template, sym.label_sim_threshold, sym.synonyms, sym.label_rel_margin)


def build_system(bag_config: str | Path, rc: dict, run_dir: Path, bag_overrides: dict | None = None,
                 vlm: VLMClient | None = None, matcher: LabelMatcher | None = None,
                 detector: YoloWorldFastSam | None = None) -> System:
    """`vlm`, `matcher` and `detector` can be passed in to reuse loaded models across runs."""
    src = BagSource(load_bag_config(resolve(bag_config), {**rc.get("bag", {}), **(bag_overrides or {})}))
    c = build_components(rc, run_dir, vlm, matcher, detector)
    sim = BagEventSim(load_sim_config(overrides=rc.get("sim")))
    return System(src, c.perception, c.memory, c.matcher, c.acquirer, c.planner, sim,
                  c.vlm, c.grid, c.mapping, c.exploration, c.move)


def build_components(rc: dict, run_dir: Path, vlm: VLMClient | None = None,
                     matcher: LabelMatcher | None = None,
                     detector: YoloWorldFastSam | None = None) -> Components:
    """Build the same models and shared map for offline replay or live sensor adapters."""
    perception = build_perception(rc, detector)
    mapping = load_mapping_config(overrides=rc.get("mapping"))
    grid = OccupancyGrid2D.from_config(mapping)       # shared: memory (rooms), exploration
    vlm = vlm or make_vlm_client(load_vlm_config(overrides=rc.get("vlm")))
    mcfg = load_memory_config(overrides=rc.get("memory"))
    store = ImageStore(run_dir / "memory" / "images", mcfg.images.image_jpeg_quality, mcfg.images.crop_cache_size)
    labeler = RoomLabeler(mcfg.labeling, vlm) if mcfg.rooms.enabled else None   # planner's VLM, text only
    memory = UnifiedMemory(mcfg, grid, store, labeler=labeler,
                           checkpoint_dir=run_dir / "memory" / "checkpoints")
    acfg = load_acquisition_config(overrides={**rc.get("acquisition", {}),
                                              "logging": {"debug_jsonl": str(run_dir / "acquisition_debug.jsonl")}})
    matcher = matcher or build_matcher(acfg)
    acquirer = TargetAcquirer(acfg, matcher, vlm)
    move_on = load_move_config(overrides=rc.get("move")).enabled
    pcfg = load_planner_config(overrides=rc.get("planner"))
    planner = PlannerCore(pcfg, memory, acquirer, vlm, matcher,
                          decision_log=run_dir / "decisions.jsonl", move_enabled=move_on)
    exploration = build_exploration(rc, grid, memory)
    acfg_agent = load_agent_config(overrides=rc.get("agent"))
    if acfg_agent.enabled:                            # code-as-policy agents; PlannerCore is the fallback
        ncfg = load_navigation_config(overrides=rc.get("navigation"))
        planner = build_agent_runtime(acfg_agent, rc, run_dir, pcfg, ncfg, memory, grid, acquirer, vlm,
                                      matcher, planner, lambda: exploration.robot, move_on)
    move = build_move(rc, grid) if move_on else None
    return Components(perception, memory, matcher, acquirer, planner, vlm, grid, mapping, exploration, move)


def build_agent_runtime(cfg: AgentConfig, rc: dict, run_dir: Path, pcfg: PlannerConfig, ncfg: NavigationConfig,
                        memory: UnifiedMemory, grid: OccupancyGrid2D, acquirer: TargetAcquirer, vlm: VLMClient,
                        matcher: LabelMatcher, fallback: PlannerCore, robot_fn, move_on: bool):
    """ProgramRuntime (TASK_cap_mas.md) with Planner / Coder / Observer; each role uses the shared VLM
    client unless `agent.<role>.vlm` gives overrides on configs/vlm.yaml (one client per distinct config)."""
    from vlnav.agent.coder import CoderAgent
    from vlnav.agent.mas import Orchestrator
    from vlnav.agent.observer import ObserverAgent
    from vlnav.agent.planner import PlannerAgent
    from vlnav.agent.runtime import ProgramRuntime
    clients: dict[str, VLMClient] = {}

    def client(role) -> VLMClient:
        if role.vlm is None:
            return vlm
        key = repr(sorted(role.vlm.items()))
        if key not in clients:
            clients[key] = make_vlm_client(load_vlm_config(overrides=deep_merge(rc.get("vlm") or {}, role.vlm)))
        return clients[key]
    t = cfg.vlm_timeout_s
    log_dir = Path(cfg.log_dir) if cfg.log_dir else run_dir
    rt: ProgramRuntime | None = None

    def log(kind, **fields):
        if rt is not None:
            rt.log(kind, **fields)
    mas = Orchestrator(cfg, PlannerAgent(client(cfg.planner), cfg.prompts.planner, t),
                       CoderAgent(client(cfg.coder), cfg.prompts.coder, t),
                       ObserverAgent(client(cfg.observer), cfg.prompts.observer, t),
                       acquirer.filter, GraphSerializer(pcfg, matcher), ncfg, log, log_dir / "rounds.jsonl")
    rt = ProgramRuntime(cfg, pcfg, ncfg, memory, grid, acquirer, vlm, mas, fallback, robot_fn,
                        decision_log=run_dir / "decisions.jsonl", move_enabled=move_on)
    return rt
