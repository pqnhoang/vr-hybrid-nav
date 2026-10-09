"""ROS 2 MCAP bag reading without a ROS install (SYSTEM.md §4.1).

Reads every `*.mcap` file of a bag directory (decompress `.mcap.zstd` files first with
`zstd -d`), messages in log-time order across files, deserialized with the `rosbags` typestore.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcap.reader import make_reader
from rosbags.typesys import Stores, get_typestore


@dataclass(frozen=True)
class BagMessage:
    topic: str
    log_time: float          # s, recorder clock
    msg: Any                 # deserialized rosbags message


@dataclass(frozen=True)
class _File:
    path: Path
    start: float
    end: float


class McapBag:
    def __init__(self, bag_dir: str | Path, typestore: str = "ROS2_HUMBLE"):
        self.bag_dir = Path(bag_dir)
        paths = sorted(self.bag_dir.glob("*.mcap"))
        if not paths:
            zst = sorted(self.bag_dir.glob("*.mcap.zstd"))
            hint = f" (found {len(zst)} .mcap.zstd files: decompress with `zstd -d`)" if zst else ""
            raise FileNotFoundError(f"no .mcap files in {self.bag_dir}{hint}")
        self.typestore = get_typestore(getattr(Stores, typestore))
        self.files: list[_File] = []
        for p in paths:
            with open(p, "rb") as f:
                st = make_reader(f).get_summary().statistics
            self.files.append(_File(p, st.message_start_time * 1e-9, st.message_end_time * 1e-9))
        self.files.sort(key=lambda f: f.start)

    @property
    def start_time(self) -> float:
        return self.files[0].start

    @property
    def end_time(self) -> float:
        return self.files[-1].end

    def messages(self, topics: Iterable[str], start: float | None = None,
                 end: float | None = None, every_nth: int = 1) -> Iterator[BagMessage]:
        """Messages of `topics` with log time in [start, end), in log-time order.

        `every_nth` > 1 keeps only every n-th message of each topic (skips deserialization).
        """
        topics = list(topics)
        count: dict[str, int] = {}
        for f in self.files:
            if (end is not None and f.start >= end) or (start is not None and f.end < start):
                continue
            with open(f.path, "rb") as fh:
                reader = make_reader(fh)
                kw = {}
                if start is not None:
                    kw["start_time"] = int(start * 1e9)
                if end is not None:
                    kw["end_time"] = int(end * 1e9)
                for schema, channel, message in reader.iter_messages(topics=topics,
                                                                     log_time_order=True, **kw):
                    k = count[channel.topic] = count.get(channel.topic, -1) + 1
                    if k % every_nth:
                        continue
                    yield BagMessage(channel.topic, message.log_time * 1e-9,
                                     self.typestore.deserialize_cdr(message.data, schema.name))
