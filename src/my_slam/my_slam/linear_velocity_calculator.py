import rclpy
import numpy as np
import math

from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

class calculateLinearVelocities(Node):
    def __init__(self):
        super().__init__("linearVelocityCalculation")

        self.derivativesList = []
        self.pointsList = []

        self.MAXACCELERATION = 0.1
        self.MAXVELOCITY = 0.4
        self.MAXVELOCITYANGULAR = 0.3
        self.MAXBREAK = 0.1

        self.smoothed_subscriber = self.create_subscription (
            Float64MultiArray,
            '/smoothedPath',
            self.calculateLinearVelocities,
            1
        )

        self.derivatives_subscriber = self.create_subscription (
            Float64MultiArray,
            '/derivatives',
            self.updateDerivatives,
            1
        )

        self.points_subscriber = self.create_subscription (
            Float64MultiArray,
            '/path',
            self.updatePoints,
            1
        )

        self.velocity_publisher = self.create_publisher (
            Float64MultiArray,
            '/velocity',
            1
        )

    def updatePoints(self, message): # This is the points outputted by A*
        pointList = message.data 

        finalPointList = []
        x = 0
        
        for i, value in enumerate(pointList):
            if i % 2 == 0:
                x = value
            else:
                finalPointList.append([x * 0.03, value * 0.03])

        self.pointsList = finalPointList # Outputs a full list of all the points [p0, p1, p2, p3, ...]

    def updateDerivatives(self, message): # This is the derivatives outputted by the path planning algorithm
        derivatives = message.data
        derivativesList = []

        M1 = 0

        for i, value in enumerate(derivatives):
            if i % 2 == 0:
                M1 = value
            else:
                derivativesList.append([M1, value])

        self.derivativesList = derivativesList

    def constructPointList(self, pointsArray): # This is the list of points outputted along the spline
        organizedPointList = []
        x = 0
        
        for i, value in enumerate(pointsArray):
            if i % 2 == 0:
                x = value
            else:
                organizedPointList.append([x, value])

        return organizedPointList

    def calculateFirstDerivative(self, u, v1, v2):
        i = math.floor(u)

        M1 = self.derivativesList[i][0]
        M2 = self.derivativesList[i][1]

        return ((6 * u ** 2) - 6 * u) * v1 + (( 3 * u ** 2) - 4 * u + 1) * M1 + ((-6 * u ** 2) + 6 * u) * v2 + ((3 * u ** 2) - 2 * u) * M2

    def calculateSecondDerivative(self, u, v1, v2):
        i = math.floor(u)

        M1 = self.derivativesList[i][0]
        M2 = self.derivativesList[i][1]

        return (12 * u - 6) * v1 + (6 * u  - 4) * M1 + (-12 * u + 6) * v2 + (6 * u - 2) * M2

    def calculateLinearVelocities(self, message):
        points = np.array(message.data)
        splinePointList = self.constructPointList(points)
    
        velocityList = [0.2] # 1 is a placeholder here. We do not need a velocity for the starting position, since that is the position underneath the car anyways.
        distanceList = [0]

        startPointIndex = 0

        for i, point in enumerate(splinePointList):
            if i == 0:
                continue

            k = (i - 1) % 100 + 1
            u = k / 100

            previousDistance = distanceList[i - 1]
            prevoiusPoint = splinePointList[i - 1]  

            newDistance = previousDistance + np.sqrt((point[0] - prevoiusPoint[0]) ** 2 + (point[1] - prevoiusPoint[1]) ** 2)

            xPrime1 = self.calculateFirstDerivative(u, self.pointsList[startPointIndex][0], self.pointsList[startPointIndex + 1][0])
            yPrime1 = self.calculateFirstDerivative(u, self.pointsList[startPointIndex][1], self.pointsList[startPointIndex + 1][1])

            xPrime2 = self.calculateSecondDerivative(u, self.pointsList[startPointIndex][0], self.pointsList[startPointIndex + 1][0])
            yPrime2 = self.calculateSecondDerivative(u, self.pointsList[startPointIndex][1], self.pointsList[startPointIndex + 1][1])

            curvature = np.abs(xPrime1 * yPrime2 - yPrime1 * xPrime2) / ((xPrime1 ** 2 + yPrime1 ** 2) ** (3/2))

            if abs(curvature) < 1e-6:
                curveVel = self.MAXVELOCITY
            else:
                curveVel = min(self.MAXVELOCITY, np.sqrt(self.MAXACCELERATION / abs(curvature)))

            distanceList.append(newDistance)
            velocityList.append(min(curveVel, self.MAXVELOCITYANGULAR / curvature))

            startPointIndex = (i - 1) // 100

        for i, velocity in enumerate(velocityList):
            if i == 0:
                continue

            prevVel = velocityList[i - 1]
            deltaDistance = distanceList[i] - distanceList[i - 1]

            velocityList[i] = min(velocity, np.sqrt((prevVel) ** 2 + 2 * self.MAXACCELERATION * deltaDistance))

        velocityList[-1] = 0

        for i in range(len(velocityList) - 2, -1, -1):
            if i == 0:
                continue

            deltaDistance = distanceList[i + 1] - distanceList[i]

            maxVChange = np.sqrt(velocityList[i + 1] ** 2 + 2 * self.MAXBREAK * deltaDistance)

            velocityList[i] = min(velocityList[i], maxVChange)

        print("WE POSTING VEOCITIES")

        velocityMsg = Float64MultiArray()
        velocityMsg.data = velocityList
        self.velocity_publisher.publish(velocityMsg)

def main(args = None):
    print("WE CALCULATING LINEAR VELOCITIES")

    rclpy.init(args = args)

    linearVelocityNode = calculateLinearVelocities()

    rclpy.spin(linearVelocityNode)

    linearVelocityNode.destroy_node()

    rclpy.shutdown()

if __name__ == '__main__':
    main()

