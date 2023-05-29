from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():

    simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([get_package_share_directory('pacote_de_exemplos'), '/launch/simulation.launch.py']),
           launch_arguments={
                'world_path': [get_package_share_directory('pacote_de_exemplos'), '/simulation/worlds/simple_room_with_fixed_boxes.world'],
            }.items(),
    )

    robot = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([get_package_share_directory('pacote_de_exemplos'), '/launch/load.launch.py']),
           launch_arguments={
                'rvizconfig': [get_package_share_directory('pacote_de_exemplos'), '/config/rviz/mapping_cartographer.rviz'],
            }.items(),
    )

    cartographer = Node(
        package='cartographer_ros',
        executable='cartographer_node',
        name='cartographer_node',
        output='screen',
        remappings=[
            ('scan', 'scan'),
            ('imu', 'imu'),
        ],
        parameters=[{
            'use_sim_time': True
        }],
        arguments=[
            '-configuration_directory', [get_package_share_directory('pacote_de_exemplos'), '/config/nav/'],
            '-configuration_basename', 'cartographer.lua'
        ]
    )

    occupacy_grid = Node(
        package='cartographer_ros',
        executable='cartographer_occupancy_grid_node',
        name='occupancy_grid_node',
        output='screen',
        parameters=[{
            'use_sim_time': True
        }],
        arguments=[
            '-resolution', '0.05', 
            '-publish_period_sec', '1.0'
        ]
    )

    ld = LaunchDescription()
    ld.add_action(simulation)
    ld.add_action(robot)
    ld.add_action(cartographer)
    ld.add_action(occupacy_grid)

    return ld
