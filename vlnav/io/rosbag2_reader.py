"""ROS 2 bags through `rosbags` (sqlite3 `.db3`, or mcap with valid metadata). Same interface as
`McapBag`; `open_bag` picks the right reader for a bag directory."""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

from vlnav.io.mcap_bag import BagMessage, McapBag


class Rosbag2Bag:
    def __init__(self, bag_dir: str | Path, typestore: str = "ROS2_HUMBLE"):
        self.bag_dir = Path(bag_dir)
        self.typestore = get_typestore(getattr(Stores, typestore))
        with Reader(self.bag_dir) as r:
            self.start_time = r.start_time * 1e-9
            self.end_time = r.end_time * 1e-9
            self.topics = {c.topic: c.msgtype for c in r.connections}

    def messages(self, topics: Iterable[str], start: float | None = None,
                 end: float | None = None, every_nth: int = 1) -> Iterator[BagMessage]:
        """Messages of `topics` with log time in [start, end), in log-time order."""
        topics = set(topics)
        count: dict[str, int] = {}
        with Reader(self.bag_dir) as r:
            conns = [c for c in r.connections if c.topic in topics]
            if not conns:
                return
            kw = {}
            if start is not None:
                kw["start"] = int(start * 1e9)
            if end is not None:
                kw["stop"] = int(end * 1e9)
            for c, t, raw in r.messages(connections=conns, **kw):
                k = count[c.topic] = count.get(c.topic, -1) + 1
                if k % every_nth:
                    continue
                yield BagMessage(c.topic, t * 1e-9, self.typestore.deserialize_cdr(raw, c.msgtype))


def open_bag(bag_dir: str | Path, typestore: str = "ROS2_HUMBLE"):
    """Decompressed `.mcap` files -> McapBag (reads files directly), otherwise rosbags."""
    d = Path(bag_dir)
    if any(d.glob("*.mcap")):
        return McapBag(d, typestore)
    if any(d.glob("*.db3")) or (d / "metadata.yaml").exists():
        return Rosbag2Bag(d, typestore)
    raise FileNotFoundError(f"no ROS 2 bag in {d}")
