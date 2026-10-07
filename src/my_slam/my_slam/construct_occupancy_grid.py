import rclpy
import numpy as np
import sys

from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Float64

class occupancyGrid(Node):
    def __init__(self):
        super().__init__("occupancyGrid")

        self.internalScan = None

        self.stateVector = np.zeros(7)
        self.occupancyDict = {}
        self.maxRange = 3.5
        self.NDT_Error = 0

        self.prevSkipped = 0
        self.skippedNum = 0

        self.scanSubscriber = self.create_subscription (
            LaserScan,
            '/scan',
            self.updateScan,
            1
        )

        self.stateSubscriber = self.create_subscription (
            Float64MultiArray,
            '/corrected',
            self.updateStateVector,
            1
        )

        self.occupancyPublisher = self.create_publisher (
            Float64MultiArray,
            '/occupancyGrid',
            1
        )

        self.NDT_error_subscriber = self.create_subscription (
            Float64,
            '/NDT_error',
            self.updateError,
            1
        )

    def updateScan(self, message):
        self.internalScan = message

    def updateError(self, message):
        self.NDT_Error = message.data

    def updateStateVector(self, message):
        newStateVector = np.array(message.data)
        self.skippedNum += 1

        stateError = sum(newStateVector - self.stateVector)
        if (self.NDT_Error > 0.3):
            print("NDT ERROR: ", self.NDT_Error)
            self.stateVector = newStateVector
            self.updateMap()
            self.prevSkipped = self.skippedNum
        else:
            if self.NDT_Error > 0.2:
                print("NDT ERROR TOO BIG -------------------------------------------------")

            print("WE SKIPPED NUMBER: ", self.skippedNum)

    def updateMap(self):
        xRobot = self.stateVector[0]
        yRobot = self.stateVector[1]
        tRobot = self.stateVector[2]

        for i, range in enumerate(self.internalScan.ranges):
            if range <= 0 or range > self.maxRange: # max range set to 10 meters
                print("NOT IN RANGE")
                continue 

            if not np.isfinite(range):
                print("NOT FINITE")
                continue

            xCurrent = xRobot
            yCurrent = yRobot

            angle = self.internalScan.angle_min + i * self.internalScan.angle_increment # This is the number, in radians, my lidar says it angle increases by each increment.
            globalAngle = angle + tRobot

            xStep = np.cos(globalAngle)
            yStep = np.sin(globalAngle)

            globalX = xRobot + range * np.cos(globalAngle)
            globalY = yRobot + range * np.sin(globalAngle)

            finalOccupancyX = int(globalX // .03) # This is the x of the ending grid square
            finalOccupancyY = int(globalY // .03) # This is the y of the ending grid square

            if xStep > 0:
                nextOccupancyX = (xCurrent //.03 + 1) * .03
            else:
                nextOccupancyX = (xCurrent //.03) * .03 # This is to ensure that we update in the correct direction

            if yStep > 0:
                nextOccupancyY = (yCurrent //.03 + 1) * .03 # After this, to update next occupancy, just add .03 to whatever we contact with first.
            else:
                nextOccupancyY = (yCurrent //.03) * .03 

            currentOccupancyX = int(xCurrent // .03) # x and y are measured relative to the bottom left of the squares.
            currentOccupancyY = int(yCurrent // .03)

            while True:
                logOdds = self.occupancyDict.get((currentOccupancyX, currentOccupancyY), 0)
                distance = np.sqrt((xCurrent - xRobot) ** 2 + (yCurrent - yRobot) ** 2)

                if distance > self.maxRange:
                    break

                distScore = 1 - distance / self.maxRange

                if currentOccupancyX == finalOccupancyX and currentOccupancyY == finalOccupancyY:
                    newLogOdds = max(-4.6, min(4.6, logOdds + (1.5 * distScore)))
                    self.occupancyDict[(currentOccupancyX, currentOccupancyY)] = newLogOdds
                    #print("WE BROKE OUT!")
                    break
                else:
                    newLogOdds = max(-4.6, min(4.6, logOdds - distScore))

                """print("CURRENT OCCUPANCY X: ", currentOccupancyX)
                print("CURRENT OCCUPANCY Y: ", currentOccupancyY)

                print("FINAL OCCUPANCY X: ", finalOccupancyX)
                print("FINAL OCCUPANCY Y: ", finalOccupancyY)

                print("ANGLE: ", globalAngle)
                print("COUNTER: ", counter)"""

                if currentOccupancyY > 500:
                    sys.exit(0)

                self.occupancyDict[(currentOccupancyX, currentOccupancyY)] = newLogOdds

                if abs(xStep) < 1e-12:
                    timeX = np.inf
                else:
                    timeX = (nextOccupancyX - xCurrent) / xStep

                if abs(yStep) < 1e-12: # This is all to prevent division by zero
                    timeY = np.inf
                else:
                    timeY = (nextOccupancyY - yCurrent) / yStep

                if timeX < timeY:
                    xCurrent = nextOccupancyX
                    yCurrent = yCurrent + timeX * yStep

                    if xStep > 0:
                        currentOccupancyX += 1
                        nextOccupancyX = (currentOccupancyX + 1) * 0.03
                    else: # This is to ensure that the next step is pointed towards where the laser is pointed. I.E. ensure we move in the same direction in the next step.
                        currentOccupancyX -= 1
                        nextOccupancyX = currentOccupancyX * 0.03
                    
                elif timeX > timeY:
                    yCurrent = nextOccupancyY
                    xCurrent = xCurrent + timeY * xStep
                    if yStep > 0:
                        currentOccupancyY += 1
                        nextOccupancyY = (currentOccupancyY + 1) * 0.03
                    else:
                        currentOccupancyY -= 1
                        nextOccupancyY = currentOccupancyY * 0.03

                else:
                    xCurrent = nextOccupancyX
                    yCurrent = nextOccupancyY

                    if xStep > 0:
                        currentOccupancyX += 1
                        nextOccupancyX = (currentOccupancyX + 1) * 0.03
                    else:
                        currentOccupancyX -= 1
                        nextOccupancyX = currentOccupancyX * 0.03

                    if yStep > 0:
                        currentOccupancyY += 1
                        nextOccupancyY = (currentOccupancyY + 1) * 0.03
                    else:
                        currentOccupancyY -= 1
                        nextOccupancyY = currentOccupancyY * 0.03

            print("WE BROKE OUT!!")
        
        flattenedGrid = []
        occupancyMsg = Float64MultiArray()

        dictList = list(self.occupancyDict.items())
        for square in dictList:
            flattenedGrid.append(square[0][0])
            flattenedGrid.append(square[0][1])
            flattenedGrid.append(square[1])

        occupancyMsg.data = flattenedGrid

        print("publishing!!")
        self.occupancyPublisher.publish(occupancyMsg)

def main(args = None):
    rclpy.init(args = args)

    newNode = occupancyGrid()

    rclpy.spin(newNode)

    newNode.destroy_node()

    rclpy.shutdown()

if __name__ == "__main__":
    main()
    print("WE constructing Occupancy")

            

