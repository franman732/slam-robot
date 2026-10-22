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

    curve_node = Node(
        package='my_slam',
        executable='curve_smoothing',
        name='curve_node',
        output='screen'
    )

    velocity_calculator_node = Node(
        package='my_slam',
        executable='linear_velocity_calculator',
        name='velocity_calculator_node',
        output='screen'
    )
    path_checker_node = Node(
        package='my_slam',
        executable='path_checker',
        name='path_checker_node',
        output='screen'
    )

    drive_controller_node = Node(
        package='my_slam',
        executable='drive_controller',
        name='drive_controller_node',
        output='screen'
    )

    other_nodes = TimerAction(
        period=10.0,
        actions=[
            rviz_node,
            EKF_Node,
            voxel_construction_node,
            map_matching_node,
            visualization_node,
            occupancy_node,
            frontier_node,
            curve_node,
            velocity_calculator_node,
            path_checker_node,
            drive_controller_node
            ]
        )

    path_planner_node = TimerAction(
        period=10.0,
        actions=[
            path_node,
            ]
        )

    # 3. Pack them into the LaunchDescription and return it
    return LaunchDescription([
        gazebo_launch,
        other_nodes,
        path_planner_node
    ])