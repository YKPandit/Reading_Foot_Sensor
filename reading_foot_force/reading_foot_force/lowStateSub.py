import rclpy
from rclpy.node import Node
from unitree_go.msg import LowState
from std_msgs.msg import String
from time import perf_counter_ns
from time import time

# Constructor that inherits from the main Node class
class MinSub(Node):
    def __init__(self):
        super().__init__('min_sub')
        self.subscription_ = self.create_subscription(LowState, '/lowstate', self.listener_callback, 10)
        self.subscription_

        self.log_hz = 30
        self.t_delta_ns = 1_000_000_000 / self.log_hz
        self.t_start = perf_counter_ns()

    def listener_callback(self, msg):
        t_end = perf_counter_ns()
        if t_end - self.t_start > self.t_delta_ns:
            self.t_start = t_end
            # if msg.foot_force is a string:
            log = ','.join(msg.foot_force.split(' '))
            # if msg.foot_force is an array:
            # log = ','.join(map(str, msg.foot_force))

            # timestamp in ms
            timestamp_ms = int(time() * 1000)
            log = str(timestamp_ms) + ',' + log

            # check that this works; the below logger prints additional stuff that isn't needed,
            # so does a regular print work?
            print(log)
            # self.get_logger().info(f"Recieved: {msg.foot_force}")

def main():
    rclpy.init()
    min_sub = MinSub()
    rclpy.spin(min_sub)
    min_sub.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
