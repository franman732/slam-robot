import rclpy

from rclpy.node import Node
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

from slam_messages.msg import VoxelGrid

class VoxelGridVisualizer(Node):
    def __init__(self):
        super().__init__('voxel_grid_visualizer')

        self.subscription = self.create_subscription(
            VoxelGrid,
            '/NDTVoxelMap',
            self.constructVizMap,
            10
        )

        self.publisher = self.create_publisher(
            MarkerArray,
            '/voxel_grid_visualization',
            10
        )

        self.voxelSize = 0.2

    def constructVizMap(self, message):
        markerArray = MarkerArray()
        
        for i, voxel in enumerate(message.voxels):
            marker = Marker()

            marker.header.frame_id = 'map'

            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            marker.pose.position.x = voxel.voxel_x * self.voxelSize
            marker.pose.position.y = voxel.voxel_y * self.voxelSize
            marker.pose.position.z = 0.0

            marker.scale.x = self.voxelSize
            marker.scale.y = self.voxelSize
            marker.scale.z = 0.01

            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 1.0

            marker.lifetime.sec = 0

            markerArray.markers.append(marker)

        self.publisher.publish(markerArray)


def main(args = None):
    rclpy.init(args = args)

    newNode = VoxelGridVisualizer()

    rclpy.spin(newNode)

    newNode.destroy_node()
    rclpy.shutdown()