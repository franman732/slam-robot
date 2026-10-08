import rclpy
import numpy as np
import sys
from collections import deque

from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Point

class findFrontiers(Node):
    def __init__(self):
        super().__init__("findFrontiers")

        self.stateVector = np.zeros(7) 
        self.occupancyDict = {}
        self.inflatedOccupancyDict = {}

        self.directions = [(-1, 1), (0, 1), (1, 1),
                            (-1, 0),         (1, 0),
                            (-1, -1), (0, -1), (1, -1)]
        
        self.stateSubscriber = self.create_subscription (
            Float64MultiArray,
            '/corrected',
            self.updateStateVector,
            1
        )

        self.occupancySubscriber = self.create_subscription (
            Float64MultiArray,
            '/OccupancyGrid',
            self.locateFrontiers,
            1
        )

        self.inflatedOccupancySubscriber = self.create_subscription (
            Float64MultiArray,
            '/inflatedOccupancyGrid',
            self.updateInflatedGrid,
            1
        )

        self.frontierPublisher = self.create_publisher (
            Point,
            '/frontier',
            1
        )

    def updateStateVector(self, message):
        self.stateVector = np.array(message.data)

    def updateInflatedGrid(self, message):
        gridList = message.data

        x = 0
        y = 0

        for i, value in enumerate(gridList):
            identifier = (i + 1) % 3

            if identifier == 1:
                x = value
            elif identifier == 2:
                y = value
            else:
                self.inflatedOccupancyDict[(x, y)] = value

    def findBestSquareInflated(self, square):
        squareQueue = deque()
        seenSet = set()
        squareQueue.append(square)
        seenSet.add(square)

        while squareQueue:
            startSquare = squareQueue.popleft()
            for d in self.directions:
                newSquare = (startSquare[0] + d[0], startSquare[1] + d[1])

                if not newSquare in seenSet:
                    seenSet.add(newSquare)

                    if newSquare in self.inflatedOccupancyDict:
                        if self.inflatedOccupancyDict[newSquare] < 0:
                            msg = Point()
                            msg.x = newSquare[0]
                            msg.y = newSquare[1]
                            self.frontierPublisher.publish(msg)

                            return

                        squareQueue.append(newSquare)

    def findBestFrontier(self, frontierSet): # This returns the square that best represents the largest frontier.
        seenFrontierSet = set()
        frontierQueue = deque()
        bestFrontier = [0, 0, 0] # meanX, meanY, squareCounter
        frontierSquare = [0, 0]

        for frontier in frontierSet:
            if frontier in seenFrontierSet:
                continue

            meanX, meanY = 0, 0
            squareCounter = 0

            frontierQueue.append(frontier)

            frontierList = []

            while frontierQueue:
                print("WE IN QUEUE")
                startSquare = frontierQueue.popleft()

                if startSquare in seenFrontierSet:
                    continue

                for d in self.directions:
                    newSquare = (startSquare[0] + d[0], startSquare[1] + d[1])

                    if newSquare in seenFrontierSet:
                        continue

                    if newSquare in frontierSet:
                        seenFrontierSet.add(newSquare)
                        frontierQueue.append(newSquare)

                squareCounter += 1
                meanX = meanX + (startSquare[0] - meanX) / squareCounter
                meanY = meanY + (startSquare[1] - meanY) / squareCounter
                
                frontierList.append(startSquare)
                seenFrontierSet.add(startSquare)

                print("SQUARE COUNTER: ", squareCounter)
                print("MEANX: ", meanX)
                print("MEAN Y: ", meanY)

            print("WE OUT OF QUEUE")

            if squareCounter > bestFrontier[2]:
                bestFrontier[0] = meanX
                bestFrontier[1] = meanY
                bestFrontier[2] = squareCounter

                bestDist = np.inf

                for square in frontierList:
                    dist = np.sqrt((meanX - square[0]) ** 2 + (meanY - square[1]) ** 2)

                    if dist < bestDist:
                        frontierSquare = square
                        bestDist = dist

        self.findBestSquareInflated(frontierSquare)

    def locateFrontiers(self, message):
        print("WERE LOCATING")
        seenLocateSet = set()
        frontierSet = set()
        locateQueue = deque()

        gridList = message.data
        
        x = 0
        y = 0

        for i, value in enumerate(gridList):
            identifier = (i + 1) % 3

            if identifier == 1:
                x = value
            elif identifier == 2:
                y = value
            else:
                self.occupancyDict[(x, y)] = value

        startingSquare = (self.stateVector[0] // .03, self.stateVector[1] // .03)
        seenLocateSet.add(startingSquare)

        for d in self.directions:
            newSquare = (startingSquare[0] + d[0], startingSquare[1] + d[1])

            seenLocateSet.add(newSquare)
            locateQueue.append(newSquare)

        while locateQueue:
            newSquare = locateQueue.popleft()

            if self.occupancyDict.get(newSquare, -100) == -100: # if it is a frontier, do not continue searching its boundaries, just add it to frontier set.
                frontierSet.add(newSquare) # -5 is also outside of the possible vlaues for occupancyDict, since I clamp it at around -4.
                continue 

            if self.occupancyDict[newSquare] > 0: # if we have seen it before, skip searching; if it is a boarder, skip searching
                continue

            for d in self.directions: # append all the new squares surrounding it if it is not a frontier and is not a boarder
                updatedSquare = (newSquare[0] + d[0], newSquare[1] + d[1])
                
                if not (updatedSquare in seenLocateSet):
                    seenLocateSet.add(updatedSquare)
                    locateQueue.append(updatedSquare)

        self.findBestFrontier(frontierSet)

        print("WE LOCATED FRONTIERS")

def main(args = None):
    rclpy.init(args = args)

    frontierNode = findFrontiers()

    rclpy.spin(frontierNode)

    frontierNode.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    print("WE RUNNING FRONTIER FINDER")
    main()