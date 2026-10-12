from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    defaults = {
        "instruction": "",
        "use_sim_time": "false",
        "rviz": "true",
        "save_map_images": "false",
        "map_image_every_s": "1.0",
        "map_image_dir": "",
        # Resolve defaults in the node's model environment, not the system ROS CLI.
        "sensor_config": "",
        "run_config": "",
        "run_dir": "",
    }
    args = [DeclareLaunchArgument(k, default_value=v) for k, v in defaults.items()]
    params = {k: ParameterValue(LaunchConfiguration(k), value_type=str)
              for k in ("instruction", "sensor_config", "run_config", "run_dir", "map_image_dir")}
    params["use_sim_time"] = ParameterValue(LaunchConfiguration("use_sim_time"), value_type=bool)
    params["save_map_images"] = ParameterValue(LaunchConfiguration("save_map_images"), value_type=bool)
    params["map_image_every_s"] = ParameterValue(LaunchConfiguration("map_image_every_s"), value_type=float)
    rviz_config = Path(get_package_share_directory("vlnav_ros")) / "rviz" / "vlnav.rviz"
    return LaunchDescription(args + [
        Node(package="vlnav_ros", executable="vlnav_node", name="vlnav", output="screen", parameters=[params]),
        Node(package="rviz2", executable="rviz2", arguments=["-d", str(rviz_config)],
             parameters=[{"use_sim_time": params["use_sim_time"]}],
             condition=IfCondition(LaunchConfiguration("rviz"))),
    ])
