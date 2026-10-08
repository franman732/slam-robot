import rclpy
import numpy as np

from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64MultiArray
from slam_messages.msg import VoxelGrid
from slam_messages.msg import Voxel
from std_msgs.msg import Float64

class createVoxelGrid(Node):
    def __init__(self):
        super().__init__("createVoxelGrid")

        self.stateVector = np.array([0, # x position
                                     0, # y position
                                     0, # rotation
                                     0, # linear velocity
                                     0, # angular velocity
                                     0, # linear bias
                                     0]) # angular bias

        self.lidarMessage = None

        self.finalVoxelDict = {} # hash (x, y) --> x, y, mean vector, covarianceMatrix
        self.infoDict = {} # hash (x, y) --> total # of elements, mean vector, product matrix 
        self.lidarOffset = np.array([-0.032, 0.0])

        self.buildCounter = 0

        self.prevSkipped = 0
        self.skippedNum = 0

        self.NDT_Error = 0

        self.lidar_subscriber = self.create_subscription (
            LaserScan,
            '/scan',
            self.updateLidarScan,
            1
        )

        self.EKF_subscriber = self.create_subscription (
            Float64MultiArray,
            '/corrected',
            self.updateMap,
            1
        )

        self.NDT_error_subscriber = self.create_subscription (
            Float64,
            '/NDT_error',
            self.updateError,
            1
        )

        self.voxelPublisher = self.create_publisher (
            VoxelGrid,
            'NDTVoxelMap',
            1
        )

    def updateError(self, message):
        self.NDT_Error = message.data

    def updateMap(self, message):
        if self.buildCounter != 0:
            newStateVector = np.array(message.data)
            self.skippedNum += 1

            stateError = sum(newStateVector - self.stateVector)
            if (self.NDT_Error > 0.15):
                #print("NDT ERROR: ", self.NDT_Error)
                self.stateVector = newStateVector
                self.constructGrid(self.lidarMessage)
                self.prevSkipped = self.skippedNum
            else:
                if self.NDT_Error > 0.2:
                    print("NDT ERROR TOO BIG -------------------------------------------------")

                """if (self.skippedNum - self.prevSkipped > 12):
                    self.stateVector = newStateVector
                    self.constructGrid(self.lidarMessage)
                    self.prevSkipped = self.skippedNum"""

                print("WE SKIPPED NUMBER: ", self.skippedNum)
        else:
            self.buildCounter += 1


    def updateLidarScan(self, message):
        if self.lidarMessage == None:
            self.lidarMessage = message
            self.constructGrid(self.lidarMessage)
        
        self.lidarMessage = message

    def constructGrid(self, message):
        for i, range in enumerate(message.ranges):
            if range > 20: # max range set to 20 meters
                continue

            angle = message.angle_min + i * message.angle_increment # This is the number, in radians, my lidar says it angle increases by each increment.

            relativeX = range * np.cos(angle)
            relativeY = range * np.sin(angle)

            baseX = relativeX + self.lidarOffset[0]
            baseY = relativeY + self.lidarOffset[1]

            theta = self.stateVector[2]

            mapX = baseX * np.cos(theta) - baseY * np.sin(theta) + self.stateVector[0]
            mapY = baseX * np.sin(theta) + baseY * np.cos(theta) + self.stateVector[1]

            voxelX = mapX // .20 # X input for dictionary
            voxelY = mapY // .20 # Y input for dictionary

            # [0] --> number of entries, [1] --> mean vector [meanX, meanY], [2] --> previous product matrix
            previousValues = self.infoDict.get((voxelX, voxelY), [0, np.array([0, 0]), np.array([[0 , 0], [0, 0]])])
            updatedN = previousValues[0] + 1
            
            meanDiff = np.array([mapX, mapY]) - previousValues[1] # All of this is done using vector element by element subtraction, multiplication, and division.

            updatedMean = previousValues[1] + meanDiff / updatedN

            # the np.reshape is necessary to make the horizontal vector vertical; it is equivalent to transposing a 1d vector.
            updatedProductMatrix = previousValues[2] + previousValues[0] / updatedN * meanDiff * np.reshape(meanDiff, (2, 1)) 

            if updatedN > 1:
                covMatrix = updatedProductMatrix / previousValues[0]
            else:
                covMatrix = np.zeros((2, 2))

            self.finalVoxelDict[(voxelX, voxelY)] = [voxelX, voxelY, updatedMean, covMatrix]
            self.infoDict[(voxelX, voxelY)] = [updatedN, updatedMean, updatedProductMatrix]

        msg = VoxelGrid()
        for element in self.finalVoxelDict.values():
            voxelMessage = Voxel()

            voxelMessage.voxel_x = element[0] # voxel x
            voxelMessage.voxel_y = element[1] # voxel y
            
            voxelMessage.x_mean = element[2][0] # mean x 
            voxelMessage.y_mean = element[2][1] # mean y

            voxelMessage.x_var = element[3][0][0] # var x
            voxelMessage.xy_cov = element[3][0][1] # cov xy    both off-diagonal values will be the same, since they both represent covariance between x and y.
            voxelMessage.y_var = element[3][1][1] # var y

            msg.voxels.append(voxelMessage)

        self.voxelPublisher.publish(msg)

    def hash(self, x, y):
        if x >= y:
            return x * x + x + y
        else:
            return y * y + x


def main(args = None):
    rclpy.init(args=args)

    newNode = createVoxelGrid()

    rclpy.spin(newNode)

    newNode.destroy_node()

    rclpy.shutdown()

if __name__ == "__main__":
    main()
    print("WE running voxel grid creation")