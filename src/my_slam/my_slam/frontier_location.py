import rclpy
import numpy as np
import sys
from collections import deque

from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Bool
from geometry_msgs.msg import Point

class findFrontiers(Node):
    def __init__(self):
        super().__init__("findFrontiers")

        self.stateVector = np.zeros(7) 
        self.occupancyDict = {}
        self.inflatedOccupancyDict = {}

        self.invalidFrontierSet = set()
        self.invalidFrontier = None

        self.invalidFrontierCounter = 0

        self.directions = [(-1, 1), (0, 1), (1, 1),
                            (-1, 0),         (1, 0),
                            (-1, -1), (0, -1), (1, -1)]
        
        self.state_subscriber = self.create_subscription (
            Float64MultiArray,
            '/corrected',
            self.updateStateVector,
            1
        )

        self.occupancy_subscriber = self.create_subscription (
            Float64MultiArray,
            '/occupancyGrid',
            self.locateFrontiers,
            1
        )

        self.inflated_occupancy_subscriber = self.create_subscription (
            Float64MultiArray,
            '/inflatedOccupancyGrid',
            self.updateInflatedGrid,
            1
        )

        self.invalid_frontier_subscriber = self.create_subscription (
            Point,
            '/invalidFrontier',
            self.addInvalidFrontier,
            1
        )

        self.frontier_publisher = self.create_publisher (
            Point,
            '/frontier',
            1
        )

        self.update_path_publisher = self.create_publisher (
            Bool,
            '/updatePath',
            1
        )

        self.stop_publisher = self.create_publisher (
            Bool,
            '/stopVehicle',
            1
        )

    def addInvalidFrontier(self, message):
        point = (int(message.x), int(message.y))
        self.invalidFrontier = point

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

    def findBestSquareInflated(self, square): # square is the target square, in grid coordinates. This function takes that square, and finds the closest valid square in the inflated occupancy grid.
        print("WE CHECKING FOR VALID FRONTIER AGAIN")
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
                            if self.invalidFrontierCounter > 10: 
                                msg = Bool()
                                msg.data = True
                                self.stop_publisher.publish(msg)

                            if (newSquare[0], newSquare[1]) == self.invalidFrontier:
                                print("INVALID FRONTIER ADDED -----------------------------")
                                self.invalidFrontierSet.add((square[0], square[1]))
                                self.invalidFrontierCounter += 1
                                return

                            if len(self.invalidFrontierSet) > 0: # The only time where invalidFrontierSet will have an item, is if pathPlanning published that the given frontier is invalid. If that happens, we need to tell 
                                updateMessage = Bool()            # pathPlanning to run again, which is what this publish does. 
                                updateMessage.data = True
                                self.update_path_publisher.publish(updateMessage)

                            self.invalidFrontierCounter = 0

                            print("PUBLISHED VALID FRONTIER")
                            msg = Point()
                            msg.x = newSquare[0]
                            msg.y = newSquare[1]
                            self.frontier_publisher.publish(msg)
                            self.invalidFrontierSet = set()
                            self.invalidFrontier = None

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
                #print("WE IN QUEUE")
                startSquare = frontierQueue.popleft()

                if startSquare in seenFrontierSet:
                    continue

                for d in self.directions:
                    newSquare = (startSquare[0] + d[0], startSquare[1] + d[1])

                    if newSquare in seenFrontierSet:
                        continue

                    if newSquare in frontierSet:
                        frontierQueue.append(newSquare)

                squareCounter += 1
                meanX = meanX + (startSquare[0] - meanX) / squareCounter
                meanY = meanY + (startSquare[1] - meanY) / squareCounter
                
                frontierList.append(startSquare)
                seenFrontierSet.add(startSquare)

                #print("SQUARE COUNTER: ", squareCounter)

            #print("WE OUT OF QUEUE")

            if squareCounter > bestFrontier[2]: # This gets the mean x and mean y of the frontier.
                #print("BEST FRONTIER: ", bestFrontier[2])

                bestDist = np.inf

                for square in frontierList: # This gets the actual square closest to the frontier
                    dist = np.sqrt((meanX - square[0]) ** 2 + (meanY - square[1]) ** 2)

                    if dist < bestDist:
                        candidate = square
                        bestDist = dist

                if (candidate[0], candidate[1]) in self.invalidFrontierSet:
                    print("INVALID FRONTIER SQUARE. SKIPPING")
                    continue

                frontierSquare = candidate
                bestFrontier[0] = meanX
                bestFrontier[1] = meanY
                bestFrontier[2] = squareCounter

        #print("MEANX: ", bestFrontier[0])
        #print("MEAN Y: ", bestFrontier[1])

        if bestFrontier[2] >= 6:
            self.findBestSquareInflated(frontierSquare)
        else:
            print("NUMSQUARES: ", bestFrontier[2])
            print("WE SEARCHED EVERYTHING ALREADY WE DONE WOOOOO ------------------------")

    def locateFrontiers(self, message):
        #print("WERE LOCATING")
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

        #print("WE LOCATED FRONTIERS")

def main(args = None):
    rclpy.init(args = args)

    frontierNode = findFrontiers()

    rclpy.spin(frontierNode)

    frontierNode.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    print("WE RUNNING FRONTIER FINDER")
    main()