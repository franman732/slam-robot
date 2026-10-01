import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # 1. Define the main Turtlesim simulation node
    turtlesim_node = Node(
        package='turtlesim',
        executable='turtlesim_node',
        name='sim_node',
        output='screen'
    )

    # 2. Define the Keyboard Teleop node to run at the same time
    teleop_node = Node(
        package='turtlesim',
        executable='turtle_teleop_key',
        name='teleop_node',
        output='screen',
        prefix='xterm -e'  # Opens this node in a separate terminal window so it can capture keyboard inputs
    )

    # 3. Pack them into the LaunchDescription and return it
    return LaunchDescription([
        turtlesim_node,
        teleop_node
    ])