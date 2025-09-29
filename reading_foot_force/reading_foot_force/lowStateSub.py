import rclpy
from rclpy.node import Node
from unitree_go.msg import LowState
from std_msgs.msg import String
from time import perf_counter_ns, time

class MinSub(Node):
    def __init__(self):
        super().__init__('min_sub')
        self.subscription_ = self.create_subscription(LowState, '/lowstate', self.listener_callback, 10)
        self.subscription_

        self.log_hz = 30
        self.t_delta_ns = 1_000_000_000 / self.log_hz
        self.t_start = perf_counter_ns()
        self.data_buffer = []
        self.interval_sec = 3
        self.interval_start = None
        self.file_count = 0
        self.collecting = False

    def listener_callback(self, msg):
        t_end = perf_counter_ns()
        if not self.collecting:
            user_input = input("Start next 3-second collection? (press Enter to start, or type 'q' to quit): ")
            if user_input.strip().lower() == 'q':
                print("Exiting collection.")
                rclpy.shutdown()
                return
            self.collecting = True
            self.interval_start = time()
            self.data_buffer = []
            print("Collecting data for 3 seconds...")

        if t_end - self.t_start > self.t_delta_ns:
            self.t_start = t_end

            timestamp_ms = int(time() * 1000)
            log = str(timestamp_ms) + ',' + ','.join(map(str, msg.foot_force))
            self.data_buffer.append(log)

            if time() - self.interval_start >= self.interval_sec:
                filename = f"foot_force_{self.file_count}_{timestamp_ms}.csv"
                with open(filename, "w") as f:
                    f.write('\n'.join(self.data_buffer))
                print(f"Saved {len(self.data_buffer)} samples to {filename}")
                self.file_count += 1
                self.collecting = False  # Wait for user to start next cycle

def main():
    rclpy.init()
    min_sub = MinSub()
    rclpy.spin(min_sub)
    min_sub.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()