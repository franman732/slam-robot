import rclpy
import time
import numpy as np

from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64MultiArray

stateVector = np.array([0, # x pos 0
                        0, # y pos 1
                        0, # theta 2
                        0, # linear velocity 3
                        0, # angular velocity 4
                        0.1221926719, # linear bias 5; This is preset to start, and is just the value i got from a 2 minute standstill test
                        -1.28 * 10 ** - 7]) # angular bias 6       These values are updated later using the covariance and noise matrixes when we do correction phase.

class EKF_Prediction(Node):
    def __init__(self):
        super().__init__("EKF_Prediction")

        self.newOdom = 0
        self.odomVel = 0
        self.odomAngVel = 0

        self.startTime = time.perf_counter() # Represents the time from the start of the program, in ms, to the time that we last updated our stateVector. This is used to determine delta T

        self.IMU_subscriber = self.create_subscription(
            Imu,
            '/imu',
            self.updateIMU,
            1
        )

        self.odomSubscriber = self.create_subscription(
            Odometry,
            '/odom',
            self.updateOdom,
            1
        )

        self.publisher = self.create_publisher(
            Float64MultiArray,
            '/prediction',
            10
        )
        

    def updateIMU(self, msg):
        linearAccelX = msg.linear_acceleration.x
        angularVelocity = msg.angular_velocity.z

        self.updateStateVector(time.perf_counter() - self.startTime, angularVelocity, linearAccelX)

    def updateOdom(self, msg):        
        odomVelX = msg.twist.twist.linear.x
        odomVelY = msg.twist.twist.linear.y

        self.odomAngVel = msg.twist.twist.angular.z
        self.newOdom = 1
        self.odomVel = np.sqrt(odomVelX ** 2 + odomVelY ** 2)

    def updateStateVector(self, deltaT, angularVel, linearAccel): # accelerations come from IMU, and angular velocity comes from IMU
        xPos = stateVector[0]
        yPos = stateVector[1]
        theta = stateVector[2]
        stateLinearVel = stateVector[3]
        linearBias = stateVector[5]
        angularBias = stateVector[6]

        # Makes linearVel either the stored state velocity, or the velocity outputted by the odometry if there is a new velocity available.
        linearVel = (stateLinearVel * abs(self.newOdom - 1) + self.newOdom * self.odomVel)
        
        if self.odomVel > 0.001:
            angularVel = angularVel + 0.2 * (self.odomAngVel - angularVel) * self.newOdom - angularBias
        else:
            angularVel = angularVel - angularBias

        stateVector[0] = xPos + linearVel * np.cos(theta) * deltaT + (1/2) * (linearAccel - linearBias) * np.cos(theta) * (deltaT ** 2)
        stateVector[1] = yPos + linearVel * np.sin(theta) * deltaT + (1/2) * (linearAccel - linearBias) * np.sin(theta) * (deltaT ** 2)
        stateVector[2] = theta + angularVel * deltaT
        stateVector[3] = linearVel + (linearAccel - linearBias) * deltaT
        stateVector[4] = angularVel
        
        # stateVector values 5 and 6 are not updated in this prediction phase. They are updated during the propagation of error in correction phase.

        msg = Float64MultiArray()
        msg.data = stateVector.flatten().tolist()
        print("WE SHOUDL HAVE PUBLISHED")
        print("XPOS TO SEE CHANGING: ", xPos)

        self.publisher.publish(msg)
        self.startTime = time.perf_counter()
        self.newOdom = 0

def main(args = None):
    rclpy.init(args=args)

    newNode = EKF_Prediction()

    rclpy.spin(newNode)

    newNode.destroy_node()

    rclpy.shutdown()

if __name__ == "__main__":
    main()
    print("WE running EKF")