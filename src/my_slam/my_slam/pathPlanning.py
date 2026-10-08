import rclpy
import numpy as np
from collections import deque
import heapq

from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Point

class pathPlanning(Node):
    def __init__(self):
        super().__init__("findFrontiers")

        self.stateVector = np.zeros(7)
        self.inflatedOccupancyDict = {}
        self.shortestDict = {}

        self.directions = [(-1, 1, .42426), (0, 1, 1), (1, 1, .42426),
                            (-1, 0, 1),                 (1, 0, 1),
                            (-1, -1, .42426), (0, -1, 1), (1, -1, .42426)]

        self.stateSubscriber = self.create_subscription (
            Float64MultiArray,
            '/corrected',
            self.updateState,
            1
        )

        self.inflatedOccupancySubscriber = self.create_subscription (
            Float64MultiArray,
            '/inflatedOccupancyGrid',
            self.updateOccupancyGrid,
            1
        )

        self.frontierSubscriber = self.create_subscription (
            Point,
            '/frontier',
            self.createPath,
            1
        )

        self.pathPublisher = self.create_publisher (
            Float64MultiArray,
            '/path',
            1
        )


    def updateState(self, message):
        print("UPDATING STATE")
        self.stateVector = np.array(message.data)

    def updateOccupancyGrid(self, message):
        print("UPDATING OCCUPANCY GRID")
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

    def calculateDistance(self, start, target):
        dx = abs(target[0] - start[2]) # The square has score stored in position 0.
        dy = abs(target[1] - start[3])

        return (dx + dy) + (np.sqrt(2) - 2) * min(dx, dy)

    def determineMoveOrder(self, targetSquare):
        print("WE DETERMINING MOVE ORDER")

        currentSquare = targetSquare
        finalMovesList = []

        print("LENGTH OF DICT: ", len(self.shortestDict.items()))
        iterationCount = 0
        
        while True:
            print("ITERATION COUNT: ", iterationCount)
            iterationCount += 1
            print("SQUARE: ", currentSquare)
            
            finalMovesList.append((currentSquare[0], currentSquare[1]))
            previousSquare = self.shortestDict[currentSquare] # ShortestDict takes coordinate pairs as keys, so I have to exclude score when updating currentSquare.

            if previousSquare == None:
                finalMovesList.reverse()
                self.prepareMoveList(finalMovesList)
                break

            currentSquare = (previousSquare[0], previousSquare[1])

        print("WE DETERMINED MOVE ORDER NOW")

    def prepareMoveList(self, moveList):
        print("WE PREPARING FINAL MOVE LIST")
        finalList = []

        if len(moveList) > 3:
            previousChange = None
            previousMove = moveList[0]

            for move in moveList[1:]:
                currentChange = (move[0] - previousMove[0], move[1] - previousMove[1])
                if currentChange != previousChange:
                    finalList.append(previousMove)

                previousMove = move
                previousChange = currentChange

            finalList.append(moveList[-1])
        else:
            finalList = moveList
        
        msg = Float64MultiArray()
        msg.data = np.array(finalList).flatten().tolist()
        self.pathPublisher.publish(msg)
    
    def createPath(self, message):
        print("WE CREATING PATH")
        self.shortestDict.clear()
        robotSquare = [0, 0, self.stateVector[0] // .03, self.stateVector[1] // .03, None] # squares in the queue are saved as [score, distanceTraveled, x, y, parentNode]
        endPoint = message
        targetSquare = (endPoint.x, endPoint.y) # This is in the form of a list
        
        robotSquare[0] = self.calculateDistance(robotSquare, targetSquare) # Initializes the score of robotSquare.
        self.shortestDict[(robotSquare[2], robotSquare[3])] = None

        squareHeap = []
        heapq.heapify(squareHeap)
        heapq.heappush(squareHeap, robotSquare)

        while squareHeap:
            startSquare = heapq.heappop(squareHeap)

            for d in self.directions:
                # [dx, dy, dTraveled] = d

                newX = startSquare[2] + d[0]
                newY = startSquare[3] + d[1]

                if self.inflatedOccupancyDict.get((newX, newY), 0) > 0:
                    continue

                newT = startSquare[1] + d[2]
                distance = self.calculateDistance((0, 0, newX, newY), targetSquare)

                newScore = newT + distance * 1.2

                heapq.heappush(squareHeap, [newScore, newT, newX, newY, (startSquare[2], startSquare[3])])
                if (newX, newY) != (robotSquare[2], robotSquare[3]):
                    if self.shortestDict.get((newX, newY), (0, 0, np.inf))[2] > newScore:
                        self.shortestDict[(newX, newY)] = (startSquare[2], startSquare[3], newScore)

                    if (newX, newY) == (targetSquare[0], targetSquare[1]):
                        self.determineMoveOrder(targetSquare)
                        return

        print("WE EXITED LOOP")

def main(args = None):
    print("WE PATH PLANNING")

    rclpy.init(args=args)

    pathNode = pathPlanning()

    rclpy.spin(pathNode)

    pathNode.destroy_node()

    rclpy.shutdown()

if __name__ == '__main__':
    main()