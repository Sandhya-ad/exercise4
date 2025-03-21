#!/usr/bin/env python3

import rospy
import time
from std_msgs.msg import Float32
from auto_nav_func import Navigator
from led_service import LEDBlinker
from detect_duckiebot import DuckiebotDetector
from duckietown.dtros import DTROS, NodeType
import os

class DuckiebotAvoidanceBehaviour(DTROS):
    def __init__(self, node_name):
        super(DuckiebotAvoidanceBehaviour, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)

        self._vehicle_name = os.environ.get('VEHICLE_NAME', 'default_duckiebot')
        self.navigator = Navigator()  # Controls movement
        self.led_blinker = LEDBlinker()  # Controls LED
        self.detector = DuckiebotDetector(debug=False)  # Detects Duckiebots
        self.distance = None  # Estimated distance to the detected Duckiebot
        self.avoidance_executed = False  # Flag to avoid repeating execution

        rospy.Subscriber(f"/{self._vehicle_name}/duckiebot_detection/distance", Float32, self.distance_callback)

        rospy.loginfo("Duckiebot Avoidance Behaviour Initialized")

        # Start the behavior
        self.execute()

    def distance_callback(self, msg):
        """Updates the estimated distance from the detection node but stops updating after first detection."""
        if self.distance is None:  # Only update distance once
            self.distance = msg.data/100


    def execute(self):
        """Runs the full avoidance maneuver after detecting a Duckiebot."""
        rospy.loginfo("Waiting to detect a Duckiebot...")
        timeout = 60  # seconds
        start_time = rospy.get_time()
        rospy.sleep(0.2)
        while not rospy.is_shutdown() and (self.distance is None or self.distance == float('inf')):
            if rospy.get_time() - start_time > timeout:
                rospy.logwarn("Timeout waiting for Duckiebot. Exiting behavior.")
                return

            # Move forward a small amount and try detection again
            rospy.logwarn("Duckiebot not detected yet. Moving forward 0.05 meters to try again...")
            rospy.sleep(0.2)

        rospy.loginfo(f"Duckiebot detected at {self.distance:.2f} meters")

        # Step 1: Move to 15 cm before the bot
        if self.distance > 0.15:
            approach_distance = self.distance - 0.18
            rospy.loginfo(f"Approaching to {approach_distance:.2f} meters from bot...")
            self.navigator.move_straight(approach_distance)
        else:
            rospy.logwarn("Already within 15 cm. No forward motion needed.")

        self.navigator.stop(3)

        # Step 2: Avoidance sequence
        rospy.loginfo("Executing avoidance sequence: left → forward → right → left → drive")

        self.navigator.sharp_left()     # 90° left
        self.navigator.move_straight(0.08)   # Move 10 cm
        self.navigator.sharp_right()    # 90° right
        self.navigator.move_straight(0.3)   # Move 45 cm
        self.navigator.sharp_right()     # 90° left again
        self.navigator.move_straight(0.1)   # Move 10 cm
        self.navigator.sharp_left()     # 90° left again

        self.navigator.move_straight(0.3)    # Resume driving

        rospy.loginfo("Duckiebot avoidance maneuver complete.")
        self.led_blinker.set_led_color("green")
        self.avoidance_executed = True
        rospy.signal_shutdown("Duckiebot avoidance behavior executed.")

if __name__ == '__main__':
    node = DuckiebotAvoidanceBehaviour(node_name='duckiebot_avoidance_node')

    rate = rospy.Rate(10)
    while not rospy.is_shutdown() and not node.avoidance_executed:
        rospy.sleep(0.1)

    rospy.loginfo("Behavior completed. Exiting program.")
