from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
import os
from launch.actions import TimerAction

def generate_launch_description():
    gazebo_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    get_package_share_directory('turtlebot3_gazebo'),
                    'launch',
                    'turtlebot3_world.launch.py'
                )
            )
        )
    
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz_node',
        output='screen'
    )

    EKF_Node = Node(
        package='my_slam',
        executable='EKF_prototype',
        name='EKF',
        output='screen'
    )

    PID_node = Node(
        package='my_slam',
        executable='PID_Node',
        name='PID_Node',
        output='screen'
    )

    voxel_construction_node = Node(
        package='my_slam',
        executable='construct_voxel_grid',
        name='voxel_grid',
        output='screen'
    )

    map_matching_node = Node(
        package='my_slam',
        executable='map_matching',
        name='mapMatching',
        output='screen'
    )

    occupancy_node = Node(
        package='my_slam',
        executable='construct_occupancy_grid',
        name='occupancy_Node',
        output='screen'
    )

    visualization_node = Node(
        package='my_slam',
        executable='visualization',
        name='visualization_Node',
        output='screen'
    )

    frontier_node = Node(
        package='my_slam',
        executable='frontier_location',
        name='frontier_node',
        output='screen'
    )

    path_node = Node(
        package='my_slam',
        executable='pathPlanning',
        name='path_node',
        output='screen'
    )

    other_nodes = TimerAction(
        period=3.0,
        actions=[
            rviz_node,
            EKF_Node,
            voxel_construction_node,
            map_matching_node,
            PID_node,
            visualization_node,
            occupancy_node,
            frontier_node,
            path_node
            ]
        )

    # 3. Pack them into the LaunchDescription and return it
    return LaunchDescription([
        gazebo_launch,
        other_nodes
    ])