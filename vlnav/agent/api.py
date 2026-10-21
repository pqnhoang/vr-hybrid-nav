"""NavAPI: the only interface of a task program (TASK_cap_mas.md §5).

`RealNavAPI` (real_api.py) and `MockNavAPI` (mock_world.py) subclass `NavAPI` and keep its
signatures; the Coder prompt shows `api_doc()`, built from the docstrings below, so the prompt and
the two implementations cannot drift apart.

Rules (§5.1): queries return at once and never move the robot; actions block until they end and
report failure as a return value; programs get typed handles (`Obj`, `Room`, `Place`), never ids;
directions are relative to the robot when the call is made. Motion commands ("go forward", "turn
left at the box") go to `follow` (DualVLN); objects go through `locate` (scene graph first, explore
when unknown) and `go_to`.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field


class Cancelled(Exception):
    """The runtime cancelled the program (new task, abort). Programs must not catch it."""


class BudgetExceeded(Exception):
    """More actions than `max_primitives`. Programs must not catch it."""


class TaskFailed(Exception):
    """Raised by `fail(reason)`: ends the program as failed."""


@dataclass(frozen=True)
class Room:
    label: str
    xy: tuple[float, float]
    area: float
    objects: tuple[str, ...]          # object labels in the room
    _id: int = field(default=-1, repr=False, compare=True)


@dataclass(frozen=True)
class Obj:
    name: str                         # memory label, e.g. "couch"
    target: str                       # what it was looked for as, e.g. "sofa" (report() uses it)
    room: Room | None
    xy: tuple[float, float] | None    # None: not seen yet (dry-run placeholder)
    height: float
    score: float
    n_obs: int
    seen_s_ago: float
    verified: bool = False            # confirmed by the VLM (locate / explore / go_to)
    _id: int = field(default=-1, repr=False, compare=True)


@dataclass(frozen=True)
class Place:
    xy: tuple[float, float]
    yaw: float                        # radians, map frame
    room: Room | None


class NavAPI:
    """Robot API for task programs. Write `def task(api):` and call only these methods.

    Queries (instant, the robot does not move):
      find, rooms, here, distance, side
    Actions (block until done, may fail; failure is a return value):
      follow, locate, explore, go_to, check, report, fail

    Choosing the action for each part of the instruction:
      - A movement command or route ("go forward", "turn right at the box", "go through the door",
        "follow the corridor", "stop near the plant") -> follow(<that part as one sentence>).
      - An object to find or reach -> obj = locate(name); then go_to(obj) and report(obj).
        locate() uses the scene graph when a matching object is already known and verified (the
        robot then goes straight to it), and explores when it is unknown.
    """

    # ------------------------------------------------------------------------------ queries
    def find(self, name: str, room: Room | str | None = None, near: Obj | None = None,
             radius: float = 3.0) -> list[Obj]:
        """Known objects matching `name` in the scene graph, best first; [] if none. Not verified:
        use it for conditions ("if the kitchen has a kettle"), not as proof that the target is there.
        `room`: a Room or a room label; `near`: only objects within `radius` m of that object."""
        raise NotImplementedError

    def rooms(self, label: str | None = None) -> list[Room]:
        """Known rooms, largest first; with `label`, only rooms of that type (e.g. "kitchen")."""
        raise NotImplementedError

    def here(self) -> Place:
        """The robot's position, heading and room now."""
        raise NotImplementedError

    def distance(self, a: Obj | Room | Place, b: Obj | Room | Place | None = None) -> float | None:
        """Path length in m from `b` (default: the robot) to `a` on the map; None if no known path."""
        raise NotImplementedError

    def side(self, x: Obj | Room | Place) -> str:
        """Where `x` is relative to the robot now: "ahead", "left", "right" or "behind"."""
        raise NotImplementedError

    # ------------------------------------------------------------------------------ actions
    def follow(self, instruction: str, timeout: float | None = None) -> bool:
        """Drive a route or movement command with the vision-language navigation model, e.g.
        follow("Go forward to the cartoon box and turn right."). One call per route part, written
        as one imperative sentence; objects in it are only landmarks. True when the model says the
        route is done, False on failure or timeout."""
        raise NotImplementedError

    def locate(self, name: str, prompts: list[str] | None = None, room: Room | str | None = None,
               timeout: float | None = None) -> Obj | None:
        """Find a verified object `name` without driving to it: a known match in the scene graph
        if the VLM confirms one, otherwise explore until one is seen and confirmed. `prompts`: 1-3
        detector phrases (e.g. ["armchair", "lounge chair"]); `room`: where to search first.
        None if not found within `timeout` s."""
        raise NotImplementedError

    def explore(self, name: str, prompts: list[str] | None = None, room: Room | str | None = None,
                timeout: float | None = None) -> Obj | None:
        """Explore unknown space until a verified `name` is seen. Like locate() but never uses
        objects already in the scene graph (use it when the known ones are wrong)."""
        raise NotImplementedError

    def go_to(self, target: Obj | Room | Place, timeout: float | None = None) -> bool:
        """Drive to an object (stops next to it), a room (its centre) or a Place. An object that is
        not verified yet is checked by the VLM first. True when reached."""
        raise NotImplementedError

    def check(self, question: str, obj: Obj | None = None) -> bool | None:
        """Ask the VLM a yes/no question about the best image of `obj`, e.g.
        check("Is this sofa orange?", sofa). None if it cannot tell."""
        raise NotImplementedError

    def report(self, obj: Obj) -> None:
        """Report `obj` as the found target (call after go_to(obj) returned True)."""
        raise NotImplementedError

    def fail(self, reason: str) -> None:
        """End the task as failed with a short reason. The program stops here."""
        raise NotImplementedError


QUERIES = ("find", "rooms", "here", "distance", "side")
ACTIONS = ("follow", "locate", "explore", "go_to", "check", "report", "fail")


def api_doc(move: bool = True) -> str:
    """Docstring of NavAPI as the Coder sees it; without `move`, follow() is left out."""
    names = QUERIES + tuple(a for a in ACTIONS if move or a != "follow")
    doc = inspect.getdoc(NavAPI)
    if not move:
        doc = doc.replace("      follow, locate", "      locate")
        doc = "\n".join(l for l in doc.splitlines() if "follow(" not in l and "movement command" not in l)
    out = [doc, ""]
    for n in names:
        fn = getattr(NavAPI, n)
        sig = str(inspect.signature(fn)).replace("(self, ", "(").replace("(self)", "()")
        body = "\n".join("    " + l for l in inspect.getdoc(fn).splitlines())
        out.append(f"api.{n}{sig}\n{body}")
    out.append("Obj fields: name, target, room, xy, height, score, n_obs, seen_s_ago, verified. "
               "Room fields: label, xy, area, objects. Place fields: xy, yaw, room.")
    return "\n\n".join(out)
