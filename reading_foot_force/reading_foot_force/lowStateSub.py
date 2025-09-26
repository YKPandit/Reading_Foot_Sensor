import rclpy
from rclpy.node import Node
from unitree_go.msg import LowState
from  std_msgs.msg import String

# Constructor that inherits from the main Node class
class MinSub(Node):

    def __init__(self):
        super().__init__('min_sub')
        self.subscription_ = self.create_subscription(LowState, 'rt/lowstate', self.listener_callback, 10)
        self.subscription_

    def listener_callback(self, msg):
        self.get_logger().info(f"Recieved: {msg.foot_force}")


def main():
    rclpy.init()
    min_sub = MinSub()
    rclpy.spin(min_sub)
    min_sub.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()