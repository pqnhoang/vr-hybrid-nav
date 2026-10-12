"""Save the ROS occupancy map and its RViz overlays as simple PNG files."""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np


class MapImageWriter:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.count = max((int(p.stem[6:]) for p in self.directory.glob("frame_*.png")
                          if p.stem[6:].isdigit()), default=0)

    def write(self, map_msg, frontiers, robot, output) -> Path | None:
        """Save the same map, frontiers, path and goal published for RViz."""
        width, height = map_msg.info.width, map_msg.info.height
        if not width or not height:
            return None
        cells = np.asarray(map_msg.data, dtype=np.int8).reshape(height, width)
        image = np.full((height, width, 3), 200, dtype=np.uint8)
        image[cells == 0] = (255, 255, 255)
        image[cells > 0] = (35, 35, 35)
        image = cv2.flip(image, 0)

        # Keep small maps readable without changing the map's cell boundaries.
        scale = max(1, math.ceil(600 / max(width, height)))
        if scale > 1:
            image = cv2.resize(image, (width * scale, height * scale), interpolation=cv2.INTER_NEAREST)
        ox = map_msg.info.origin.position.x
        oy = map_msg.info.origin.position.y
        resolution = map_msg.info.resolution

        def pixel(xy):
            x, y = xy[:2]
            return (round((x - ox) / resolution * scale - .5),
                    round((height - (y - oy) / resolution) * scale - .5))

        for point in frontiers:
            color = (128, 128, 128) if point["blacklisted"] else (255, 217, 13)
            cv2.circle(image, pixel(point["xy"]), max(3, round(.125 / resolution * scale)),
                       color, -1, cv2.LINE_AA)

        active = output.goal is not None and output.goal_kind != "hold" and output.path
        if active:
            points = np.asarray([pixel(xy) for xy in output.path], dtype=np.int32)
            if len(points) > 1:
                cv2.polylines(image, [points], False, (255, 200, 30), 2, cv2.LINE_AA)

        if robot is not None:
            center = pixel(robot.position)
            tip = pixel((robot.position[0] + .5 * math.cos(robot.yaw()),
                         robot.position[1] + .5 * math.sin(robot.yaw())))
            cv2.circle(image, center, max(4, round(.2 / resolution * scale)),
                       (255, 80, 50), -1, cv2.LINE_AA)
            cv2.arrowedLine(image, center, tip, (255, 80, 50), 2, cv2.LINE_AA, tipLength=.35)
            cv2.putText(image, "ROBOT", (center[0] + 9, center[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 80, 50), 2, cv2.LINE_AA)

        if active:
            goal = output.goal
            center = pixel(goal.position)
            tip = pixel((goal.position[0] + .7 * math.cos(goal.yaw()),
                         goal.position[1] + .7 * math.sin(goal.yaw())))
            cv2.circle(image, center, max(5, round(.3 / resolution * scale)),
                       (13, 166, 255), -1, cv2.LINE_AA)
            cv2.arrowedLine(image, center, tip, (13, 166, 255), 2, cv2.LINE_AA, tipLength=.35)

        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise OSError("OpenCV could not encode the map PNG")
        data = encoded.tobytes()
        self.count += 1
        frame = self.directory / f"frame_{self.count:06d}.png"
        for destination in (frame, self.directory / "latest.png"):
            temporary = self.directory / f".{destination.name}.tmp"
            temporary.write_bytes(data)
            temporary.replace(destination)
        return frame
