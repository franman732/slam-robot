import rclpy
import numpy as np

from rclpy.node import Node
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import ColorRGBA

from slam_messages.msg import VoxelGrid

class VoxelGridVisualizer(Node):
    def __init__(self):
        super().__init__('voxel_grid_visualizer')

        self.voxelSubscription = self.create_subscription (
            VoxelGrid,
            '/NDTVoxelMap',
            self.constructVizMap,
            10
        )

        self.occupancySubscription = self.create_subscription (
            Float64MultiArray,
            '/occupancyGrid',
            self.constructOccupancyGrid,
            1
        )

        self.voxelPublisher = self.create_publisher (
            MarkerArray,
            '/voxel_grid_visualization',
            10
        )

        self.occupancyPublisher = self.create_publisher(
            Marker,
            '/occupancyGridVisualization',
            10
        )

        self.voxelSize = 0.2

    def constructVizMap(self, message):
        markerArray = MarkerArray()
        counter = 0

        meanRadius = 0.0125 # radius of the circle representing the mean
        num_points = 50 # number of points used to construct each elipse
        t = np.linspace(0, 2 * np.pi, num_points)

        for i, voxel in enumerate(message.voxels):
            xMin = voxel.voxel_x * self.voxelSize
            xMax = (voxel.voxel_x + 1) * self.voxelSize
            yMin = voxel.voxel_y * self.voxelSize
            yMax = (voxel.voxel_y + 1) * self.voxelSize

            corners = [
                (xMin, yMin),
                (xMax, yMin),
                (xMax, yMax),
                (xMin, yMax)
            ]

            for j in range(4):
                Line = Marker()
                Line.header.frame_id = 'map'

                Line.id = counter
                Line.type = Marker.LINE_LIST
                Line.action = Marker.ADD

                Line.scale.x = 0.01
                Line.scale.y = 0.01
                Line.scale.z = 0.01

                p1 = Point()
                p1.x = corners[j][0]
                p1.y = corners[j][1]
                p1.z = 0.0

                p2 = Point()
                p2.x = corners[(j + 1) % 4][0]
                p2.y = corners[(j + 1) % 4][1]
                p2.z = 0.0

                Line.points.append(p1)
                Line.points.append(p2)

                Line.color.r = 0.0
                Line.color.g = 0.0
                Line.color.b = 1.0
                Line.color.a = 1.0

                Line.lifetime.sec = 0

                markerArray.markers.append(Line)

                counter += 1

            mean = Marker()
            mean.header.frame_id = 'map'

            mean.id = counter
            mean.type = Marker.CYLINDER
            mean.action = Marker.ADD

            mean.pose.position.x = voxel.x_mean
            mean.pose.position.y = voxel.y_mean
            mean.pose.position.z = 0.0

            mean.scale.x = meanRadius * 2
            mean.scale.y = meanRadius * 2
            mean.scale.z = 0.01

            mean.color.r = 1.0
            mean.color.g = 0.0
            mean.color.b = 0.0
            mean.color.a = 1.0

            mean.lifetime.sec = 0
            
            markerArray.markers.append(mean)

            counter += 1



            covariance = Marker()
            covariance.header.frame_id = "map"
            covariance.type = Marker.LINE_STRIP
            covariance.action = Marker.ADD

            covariance.id = counter

            covariance.scale.x = 0.01

            covariance.color.r = 1.0
            covariance.color.g = 0.0
            covariance.color.b = 0.0
            covariance.color.a = 1.0

            covariance.lifetime.sec = 0

            elipseAngle = (1/2) * np.arctan2(2 * voxel.xy_cov, voxel.x_var - voxel.y_var)
            eigenvalues, eigenvectors = np.linalg.eigh([[voxel.x_var, voxel.xy_cov], [voxel.xy_cov, voxel.y_var]])

            xScale = np.sqrt(eigenvalues[1]) * 4
            yScale = np.sqrt(eigenvalues[0]) * 4

            a = xScale / 2
            b = yScale / 2

            x = a * np.cos(t)
            y = b * np.sin(t)

            x_rot = (x * np.cos(elipseAngle) - y * np.sin(elipseAngle))

            y_rot = (x * np.sin(elipseAngle) + y * np.cos(elipseAngle))

            x_final = x_rot + voxel.x_mean
            y_final = y_rot + voxel.y_mean

            for i in range(num_points):

                point = Point()

                point.x = float(x_final[i])
                point.y = float(y_final[i])
                point.z = 0.01

                covariance.points.append(point)

            covariance.points.append(covariance.points[0])

            markerArray.markers.append(covariance)

            counter += 1

        self.voxelPublisher.publish(markerArray)


    def constructOccupancyGrid(self, message):
        gridList = np.array(message.data)

        square = Marker()
        square.header.frame_id = 'map'
        square.id = 0
        
        square.type = Marker.CUBE_LIST
        square.action = Marker.ADD

        square.scale.x = .03
        square.scale.y = .03
        square.scale.z = 0.1

        x = 0
        y = 0

        for i, value in enumerate(gridList):
            
            identifier = (i + 1) % 3

            if identifier == 1:
                x = value
            elif identifier == 2:
                y = value
            else:
                p = Point()
                p.x = x * .03 + .015
                p.y = y * .03 + .015
                p.z = 0.01

                c = ColorRGBA()

                if value > 0:
                    c.r = 1.0
                    c.g = 0.0
                    c.b = 0.0
                    c.a = 1.0
                else:
                    c.r = 0.0
                    c.g = 1.0
                    c.b = 0.0
                    c.a = 1.0

                square.points.append(p)
                square.colors.append(c)

        self.occupancyPublisher.publish(square)
        print("markerArray")

def main(args = None):
    rclpy.init(args = args)

    newNode = VoxelGridVisualizer()

    rclpy.spin(newNode)

    newNode.destroy_node()
    rclpy.shutdown()