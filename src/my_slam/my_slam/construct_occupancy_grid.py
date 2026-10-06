import rclpy
import numpy as np

from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64MultiArray

class occupancyGrid(Node):
    def __init__(self):
        super().__init__("occupancyGrid")

        self.internalScan = None

        self.stateVector = np.zeros(7)
        self.occupancyDict = {}

        self.scanSubscriber = self.create_subscription (
            LaserScan,
            '/scan',
            self.updateScan(),
            1
        )

        self.stateSubscriber = self.create_subscription (
            Float64MultiArray,
            '/corrected',
            self.updateMap(),
            1
        )

    def updateScan(self, message):
        self.internalScan = message

    def updateMap(self, message):
        self.stateVector = np.array(message.data)

        xRobot = self.stateVector[0]
        yRobot = self.stateVector[1]
        tRobot = self.stateVector[2]

        for i, range in enumerate(message.ranges):
            if range > 20: # max range set to 20 meters
                continue

            angle = message.angle_min + i * message.angle_increment # This is the number, in radians, my lidar says it angle increases by each increment.

            localX = range * np.cos(angle)
            localY = range * np.sin(angle)

            globalAngle = angle + tRobot
            globalX = localX * np.cos(globalAngle) - localY * np.sin(globalAngle) + xRobot
            globalY = localY * np.cos(globalAngle) + localX * np.sin(globalAngle) + yRobot

            finalOccupancyX = globalX // .03 # This is the x of the ending grid square
            finalOccupancyY = globalY // .03 # This is the y of the ending grid square

            initialOccupancyX = xRobot // .03
            initialOccupancyY = yRobot // .03

            

