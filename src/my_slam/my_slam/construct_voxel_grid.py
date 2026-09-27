import rclpy
import numpy as np

from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64MultiArray
from slam_messages.msg import VoxelGrid
from slam_messages.msg import Voxel

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

        self.finalVoxelDict = {} # hash (x, y) --> x, y, mean vector, covarianceMatrix
        self.infoDict = {} # hash (x, y) --> total # of elements, mean vector, product matrix 

        self.lidar_subscriber = self.create_subscription (
            LaserScan,
            '/scan',
            self.constructGrid,
            1
        )

        self.EKF_subscriber = self.create_subscription (
            Float64MultiArray,
            '/prediction',
            self.updateStateVector,
            1
        )

        self.voxelPublisher = self.create_publisher (
            VoxelGrid,
            'NDTVoxelMap',
            1
        )

    def updateStateVector(self, message):
        self.stateVector = np.array(message.data)

    def constructGrid(self, message):
        for i, range in enumerate(message.ranges):
            if range > 20: # max range set to 20 meters
                continue

            angle = i * 0.01749303564429283 # This is the number, in radians, my lidar says it angle increases by each increment.

            relativeX = range * np.cos(angle)
            relativeY = range * np.sin(angle)

            mapX = relativeX + self.stateVector[0]
            mapY = relativeY + self.stateVector[1]

            voxelX = mapX // .20 # X input for dictionary
            voxelY = mapY // .20 # Y input for dictionary

            hashedVal = self.hash(voxelX, voxelY)

            # [0] --> number of entries, [1] --> mean vector [meanX, meanY], [2] --> previous product matrix
            previousValues = self.infoDict.get(hashedVal, [0, np.array([0, 0]), np.array([[0 , 0], [0, 0]])])
            updatedN = previousValues[0] + 1
            
            meanDiff = np.array([mapX, mapY]) - previousValues[1] # All of this is done using vector element by element subtraction, multiplication, and division.

            updatedMean = previousValues[1] + meanDiff / updatedN

            # the np.reshape is necessary to make the horizontal vector vertical; it is equivalent to transposing a 1d vector.
            updatedProductMatrix = previousValues[2] + previousValues[0] / updatedN * meanDiff * np.reshape(meanDiff, (2, 1)) 

            if updatedN > 1:
                covMatrix = updatedProductMatrix / previousValues[0]
            else:
                covMatrix = np.zeros((2, 2))

            self.finalVoxelDict[hashedVal] = [voxelX, voxelY, updatedMean, covMatrix]
            self.infoDict[hashedVal] = [updatedN, updatedMean, updatedProductMatrix]

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