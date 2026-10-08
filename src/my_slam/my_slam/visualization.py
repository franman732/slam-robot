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

        self.INFLATEDXOFFSET = 3.0
        self.OCCUPANCYXOFFSET = -3.0
        self.VOXELYOFFSET = -6.0

        self.voxelSubscription = self.create_subscription (
            VoxelGrid,
            '/NDTVoxelMap',
            self.constructVizMap,
            10
        )

        self.occupancySubscription = self.create_subscription (
            Float64MultiArray,
            '/occupancyGrid',
            self.occupancyGrid,
            1
        )

        self.inflatedOccupancySubscription = self.create_subscription (
            Float64MultiArray,
            '/inflatedOccupancyGrid',
            self.inflatedOccupancyGrid,
            1
        )

        self.stateSubscriber = self.create_subscription (
            Float64MultiArray,
            '/corrected',
            self.addCar,
            1
        )

        self.frontierSubscriber = self.create_subscription (
            Point,
            '/frontier',
            self.addFrontier,
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

        self.inflatedOccupancyPublisher = self.create_publisher(
            Marker,
            '/inflatedGridVisualization',
            10
        )

        self.carPublisher = self.create_publisher(
            Marker,
            '/carVisualization',
            1
        )

        self.frontierPublisher = self.create_publisher (
            Marker,
            '/frontierVisualization',
            1
        )

        self.VOXELSIZE = 0.2
        self.OCCUPANCYSIZE = 0.03

    def occupancyGrid(self, message):
        self.constructOccupancyGrid(message, 0)

    def inflatedOccupancyGrid(self, message):
        self.constructOccupancyGrid(message, 1)

    def constructVizMap(self, message):
        markerArray = MarkerArray()
        counter = 0

        meanRadius = 0.0125 # radius of the circle representing the mean
        num_points = 50 # number of points used to construct each elipse
        t = np.linspace(0, 2 * np.pi, num_points)

        for i, voxel in enumerate(message.voxels):
            xMin = voxel.voxel_x * self.VOXELSIZE
            xMax = (voxel.voxel_x + 1) * self.VOXELSIZE
            yMin = voxel.voxel_y * self.VOXELSIZE
            yMax = (voxel.voxel_y + 1) * self.VOXELSIZE

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
                p1.x = corners[j][0] + self.VOXELYOFFSET
                p1.y = corners[j][1]
                p1.z = 0.0

                p2 = Point()
                p2.x = corners[(j + 1) % 4][0] + self.VOXELYOFFSET
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

            mean.pose.position.x = voxel.x_mean + self.VOXELYOFFSET
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

                point.x = float(x_final[i]) + self.VOXELYOFFSET
                point.y = float(y_final[i])
                point.z = 0.01

                covariance.points.append(point)

            covariance.points.append(covariance.points[0])

            markerArray.markers.append(covariance)

            counter += 1

        self.voxelPublisher.publish(markerArray)

    def constructOccupancyGrid(self, message, type):
        gridList = np.array(message.data)

        occupancy = Marker()
        occupancy.header.frame_id = 'map'
        occupancy.id = 0
        
        occupancy.type = Marker.CUBE_LIST
        occupancy.action = Marker.ADD

        occupancy.scale.x = self.OCCUPANCYSIZE
        occupancy.scale.y = self.OCCUPANCYSIZE
        occupancy.scale.z = 0.1

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
                p.x = x * self.OCCUPANCYSIZE - self.OCCUPANCYSIZE / 2
                p.y = y * self.OCCUPANCYSIZE - self.OCCUPANCYSIZE / 2 + (self.INFLATEDXOFFSET if type != 0 else self.OCCUPANCYXOFFSET)
                p.z = 0.00125

                c = ColorRGBA()

                prob = np.exp(value) / (1 + np.exp(value))

                if prob > .50:
                    c.r = 1.0
                    c.g = 0.0
                    c.b = 0.0
                    c.a = 1.0
                else:
                    c.r = 0.0
                    c.g = 1.0
                    c.b = 0.0
                    c.a = 1.0

                occupancy.points.append(p)
                occupancy.colors.append(c)

        if type == 0:
            self.occupancyPublisher.publish(occupancy)
        else:
            self.inflatedOccupancyPublisher.publish(occupancy)
        #print("markerArray")

    def addCar(self, message):
        stateVector = np.array(message.data)

        car = Marker()
        car.header.frame_id = 'map'

        car.id = 0
        car.type = Marker.CYLINDER
        car.action = Marker.ADD

        car.pose.position.x = stateVector[0]
        car.pose.position.y = stateVector[1]
        car.pose.position.z = 0.126

        car.scale.x = .178
        car.scale.y = .138
        car.scale.z = 0.01

        car.color.r = 0.0
        car.color.g = 0.0
        car.color.b = 1.0
        car.color.a = 1.0

        car.lifetime.sec = 0
        
        self.carPublisher.publish(car)


    def addFrontier(self, message):
        xPos = message.x
        yPos = message.y

        frontier = Marker()
        frontier.header.frame_id = 'map'

        frontier.id = 0
        frontier.type = Marker.CUBE
        frontier.action = Marker.ADD

        frontier.pose.position.x = xPos * self.OCCUPANCYSIZE - self.OCCUPANCYSIZE / 2
        frontier.pose.position.y = yPos * self.OCCUPANCYSIZE - self.OCCUPANCYSIZE / 2 + self.INFLATEDXOFFSET
        frontier.pose.position.z = 0.0

        frontier.scale.x = self.OCCUPANCYSIZE
        frontier.scale.y = self.OCCUPANCYSIZE
        frontier.scale.z = .3

        frontier.color.r = 1.0
        frontier.color.g = 0.65
        frontier.color.b = 0.0
        frontier.color.a = 1.0

        frontier.lifetime.sec = 0

        self.frontierPublisher.publish(frontier)


def main(args = None):
    rclpy.init(args = args)

    newNode = VoxelGridVisualizer()

    rclpy.spin(newNode)

    newNode.destroy_node()
    rclpy.shutdown()