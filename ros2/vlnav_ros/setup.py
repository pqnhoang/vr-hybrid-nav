from glob import glob
from setuptools import find_packages, setup

setup(
    name="vlnav_ros", version="0.0.1", packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/vlnav_ros"]),
        ("share/vlnav_ros", ["package.xml"]),
        ("share/vlnav_ros/launch", glob("launch/*.launch.py")),
        ("share/vlnav_ros/rviz", glob("rviz/*.rviz")),
    ],
    install_requires=["setuptools"],
    entry_points={"console_scripts": ["vlnav_node = vlnav_ros.node:main"]},
)
