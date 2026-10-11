import rclpy
import numpy as np
import heapq

from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Point

class pathPlanning(Node):
    def __init__(self):
        super().__init__("pathPlanning")

        self.stateVector = np.zeros(7)
        self.inflatedOccupancyDict = {}
        self.shortestDict = {}
        self.haveState = False
        self.planned = False # Only one path gets planned for now, since there is no replan condition yet. Set this back to False whenever you want A* to run again.

        self.startSquare = (0, 0)
        self.STARTCLEARANCE = 4 # Squares around the robot that count as free, since the robot can start inside the inflation. Keep this the same as GRACESQUARES in the path checker.
        self.CLEARANCEWEIGHT = 1.0 # Extra cost for every blocked square touching a square. This pushes the path away from the inflated edge.
        self.MAXEXPANSIONS = 200000

        self.directions = [(-1, 1, 1.41421), (0, 1, 1), (1, 1, 1.41421),
                            (-1, 0, 1),                 (1, 0, 1),
                            (-1, -1, 1.41421), (0, -1, 1), (1, -1, 1.41421)]

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

        self.unfilteredPublisher = self.create_publisher (
            Float64MultiArray,
            '/unfilteredPath',
            1
        )


    def updateState(self, message):
        print("UPDATING STATE")
        self.stateVector = np.array(message.data)
        self.haveState = True

    def updateOccupancyGrid(self, message):
        print("UPDATING OCCUPANCY GRID")
        grid = np.array(message.data).reshape(-1, 3) # Every row is [x, y, logOdds]

        squares = zip(grid[:, 0].astype(int).tolist(), grid[:, 1].astype(int).tolist())
        self.inflatedOccupancyDict = dict(zip(squares, grid[:, 2].tolist())) # A brand new dictionary every time, so squares that are no longer inflated disappear.

    def isBlocked(self, x, y):
        if max(abs(x - self.startSquare[0]), abs(y - self.startSquare[1])) <= self.STARTCLEARANCE:
            return False # The robot is sitting here, so these squares can't be treated as blocked.

        return self.inflatedOccupancyDict.get((x, y), 1) > 0 # Default value is 1 so that any square not inside the occupancy dict counts as blocked.

    def nearBlocked(self, x, y): # Counts how many of the 8 squares around (x, y) are blocked.
        count = 0

        for d in self.directions:
            if self.isBlocked(x + d[0], y + d[1]):
                count += 1

        return count

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

                message = Float64MultiArray()
                message.data = np.array(finalMovesList, dtype=float).flatten().tolist()
                self.unfilteredPublisher.publish(message) # This is published BEFORE /path so the path checker already has every square by the time the first smoothed path comes back.

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
        msg.data = np.array(finalList, dtype=float).flatten().tolist()
        self.pathPublisher.publish(msg)

        #print("MOVELIST: ", moveList)
    
    def createPath(self, message):
        if not self.haveState:
            print("NO STATE YET, SKIPPING PATH")
            return

        if self.planned:
            return

        print("WE CREATING PATH")
        self.shortestDict.clear()

        startX = int(np.floor(self.stateVector[0] / .03))
        startY = int(np.floor(self.stateVector[1] / .03))
        self.startSquare = (startX, startY)

        targetSquare = (int(round(message.x)), int(round(message.y))) # This is in the form of a tuple of whole squares.

        if targetSquare == self.startSquare:
            print("WE ARE ALREADY AT THE TARGET")
            return

        if self.isBlocked(targetSquare[0], targetSquare[1]):
            print("TARGET SQUARE IS BLOCKED")
            return

        robotSquare = [0, 0.0, startX, startY] # squares in the queue are saved as [score, distanceTraveled, x, y]. The parent of every square lives in shortestDict.
        robotSquare[0] = self.calculateDistance(robotSquare, targetSquare) * 1.2 # Initializes the score of robotSquare.

        gScore = {self.startSquare: 0.0} # Cheapest known distanceTraveled for every square we have reached.
        self.shortestDict[self.startSquare] = None

        squareHeap = []
        heapq.heapify(squareHeap)
        heapq.heappush(squareHeap, robotSquare)

        expandedCount = 0

        while squareHeap:
            startSquare = heapq.heappop(squareHeap)
            currentSquare = (startSquare[2], startSquare[3])

            if startSquare[1] > gScore[currentSquare]: # A cheaper way to this square was found after this entry was pushed, so this entry is outdated.
                continue

            if currentSquare == targetSquare: # The goal is only checked when it is popped, which is what makes the path the cheapest one.
                self.determineMoveOrder(targetSquare)
                self.planned = True
                return

            expandedCount += 1

            if expandedCount > self.MAXEXPANSIONS:
                print("TOO MANY EXPANSIONS, GIVING UP")
                return

            for d in self.directions:
                # [dx, dy, dTraveled] = d

                newX = startSquare[2] + d[0]
                newY = startSquare[3] + d[1]

                if self.isBlocked(newX, newY):
                    continue

                if d[0] != 0 and d[1] != 0: # A diagonal move can't squeeze between two blocked squares, so both squares beside it have to be free.
                    if self.isBlocked(startSquare[2] + d[0], startSquare[3]) or self.isBlocked(startSquare[2], startSquare[3] + d[1]):
                        continue

                newT = startSquare[1] + d[2]

                if newT >= gScore.get((newX, newY), np.inf): # Cheap check first, because adding the clearance cost can only make newT bigger.
                    continue

                newT += self.CLEARANCEWEIGHT * self.nearBlocked(newX, newY)

                if newT >= gScore.get((newX, newY), np.inf):
                    continue

                distance = self.calculateDistance((0, 0, newX, newY), targetSquare)

                newScore = newT + distance * 1.2

                gScore[(newX, newY)] = newT
                self.shortestDict[(newX, newY)] = (startSquare[2], startSquare[3], newT)
                heapq.heappush(squareHeap, [newScore, newT, newX, newY])

        print("WE EXITED LOOP, NO PATH FOUND")

def main(args = None):
    print("WE PATH PLANNING")

    rclpy.init(args=args)

    pathNode = pathPlanning()

    rclpy.spin(pathNode)

    pathNode.destroy_node()

    rclpy.shutdown()

if __name__ == '__main__':
    main()