"""PNG snapshots use the published ROS map and current navigation markers."""
import cv2
import numpy as np

from vlnav.common.models import Pose
from vlnav.exploration.explorer import ExplorationOutput
from vlnav.mapping.occupancy import OccupancyGrid2D
from vlnav.runtime.live_runtime import hold
from vlnav_ros.map_images import MapImageWriter
from vlnav_ros.messages import header, occupancy_message


def test_map_pngs_show_ros_map_and_current_goal(tmp_path):
    cells = np.full((20, 20), -1, np.int8)
    cells[4:16, 4:16] = 0
    cells[8:10, 8:10] = 1
    grid = OccupancyGrid2D.from_array(cells, .2, (-2., -2.))
    map_msg = occupancy_message(grid, header("odom_lidar", 10.))
    frontiers = [{"xy": (-.6, .7), "blacklisted": False}]
    robot = Pose(position=(-1., -1., 0.), orientation_xyzw=(0., 0., 0., 1.), stamp=10.)
    goal = Pose(position=(.5, .5, 0.), orientation_xyzw=(0., 0., 0., 1.), stamp=10.)
    output = ExplorationOutput(goal=goal, goal_kind="frontier", goal_id=4,
                               path=[(-1., -1.), (.5, .5)], events=[])
    writer = MapImageWriter(tmp_path)

    assert writer.write(map_msg, frontiers, robot, hold()) == tmp_path / "frame_000001.png"
    assert writer.write(map_msg, frontiers, robot, output) == tmp_path / "frame_000002.png"
    image = cv2.imread(str(tmp_path / "latest.png"))
    assert image.shape == (600, 600, 3)
    assert np.array_equal(image[585, 15], (200, 200, 200))  # unknown
    assert np.array_equal(image[285, 135], (255, 255, 255))  # free
    assert np.array_equal(image[315, 255], (35, 35, 35))     # occupied
    assert np.any(np.all(image == (13, 166, 255), axis=2))   # goal
    assert np.any(np.all(image == (255, 217, 13), axis=2))   # frontier
    assert np.any(np.all(image == (255, 80, 50), axis=2))    # robot

    writer.write(map_msg, frontiers, robot, hold())
    cleared = cv2.imread(str(tmp_path / "latest.png"))
    assert not np.any(np.all(cleared == (13, 166, 255), axis=2))
    assert len(list(tmp_path.glob("frame_*.png"))) == 3
    assert MapImageWriter(tmp_path).write(map_msg, frontiers, robot, hold()) == tmp_path / "frame_000004.png"
