#!/usr/bin/env python3

import rospy
import time
from std_msgs.msg import Bool
from auto_nav_func import Navigator  # Controls movement
from led_service import LEDBlinker  # LED control
from crosswalk_detection import CrosswalkDetection  # Crosswalk detection
from detect_peDuck import PeDuckstrianDetector  # PeDuckstrian detection
from duckietown.dtros import DTROS, NodeType
import os

class CrosswalkBehaviour(DTROS):
    def __init__(self, node_name):
        super(CrosswalkBehaviour, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)

        self._vehicle_name = os.environ.get('VEHICLE_NAME', 'default_duckiebot')
        self.navigator = Navigator()  # Controls movement
        self.led_blinker = LEDBlinker()  # Controls LED
        self.crosswalk_detector = CrosswalkDetection(node_name="crosswalk_detection_node")  # Detects blue crosswalks
        self.peduck_detector = PeDuckstrianDetector()  # Detects PeDuckstrians

        self.stop_time = 1  # Stop time at the first blue line
        self.crosswalk_distance = None  # Distance to the first blue line
        self.waiting_for_peduck = False  # Track PeDuckstrian waiting state
        self.peduck_detected = False  # Track if a PeDuckstrian is present

        # Subscriber to listen for PeDuckstrian clearance
        rospy.Subscriber(f"/{self._vehicle_name}/peduckstrian_detection/clear", Bool, self.peduck_callback)

        rospy.loginfo("Crosswalk Behaviour Initialized")

    def execute(self):
        """Main behavior: Detect crosswalk, stop, check for PeDuckstrians, and proceed."""

        # Reset flags
        self.crosswalk_distance = None
        self.waiting_for_peduck = False
        self.peduck_detected = False

        # Step 1: Detect Crosswalk Distance with a timeout
        timeout = 200  # seconds
        start_time = rospy.get_time()
        rospy.loginfo("Detecting crosswalk distance before moving...")
        
        while self.crosswalk_distance is None and not rospy.is_shutdown():
            if rospy.get_time() - start_time > timeout:
                rospy.logerr("Timeout waiting for crosswalk. Exiting.")
                rospy.signal_shutdown("Crosswalk detection timeout")
                return
                
            self.crosswalk_distance = self.crosswalk_detector.get_crosswalk_distance()
            if self.crosswalk_distance is None:
                rospy.logwarn("No crosswalk detected. Retrying...")
                rospy.sleep(0.5)
                
        # If we're here, we've detected the crosswalk
        rospy.loginfo(f"Crosswalk detected at {self.crosswalk_distance} meters.")

        # Step 2: Move to the first blue line
        rospy.loginfo(f"Moving to crosswalk at {self.crosswalk_distance} meters...")
        self.navigator.move_straight(self.crosswalk_distance-0.03)  # Move until first blue line, safee distance

        # Step 3: Stop at first blue line
        rospy.loginfo("Stopping at first blue line for 1 second...")
        self.navigator.stop(1)  # Stop for 1 second

        # Step 4: Check for PeDuckstrians with multiple samples for reliability
        samples = 3
        detections = 0
        
        for _ in range(samples):
            if self.peduck_detector.detect_peduckstrians():
                detections += 1
            rospy.sleep(0.2)
            
        # If majority of samples detected peducks
        self.peduck_detected = (detections > samples/2)
        
        # Step 5: Handle PeDuckstrians based on detection
        if self.peduck_detected:
            rospy.loginfo("PeDuckstrians detected at blue line. Waiting until clear...")
            self.wait_for_peduck_clearance()
        else:
            rospy.loginfo("No PeDuckstrians detected. Waiting 1 second, then moving.")
            rospy.sleep(1)
            self.navigator.move_straight(0.5)  # Move past the second blue line

        # Step 6: Stop program after one execution
        rospy.loginfo("Behavior completed. Exiting program.")
        rospy.signal_shutdown("Crosswalk behavior executed successfully.")

    def peduck_callback(self, msg):
        """Handles PeDuckstrian detection callback from the topic."""
        self.peduck_detected = msg.data  # True means PeDuckstrian detected, False means clear
    def wait_for_peduck_clearance(self):
        """Wait for PeDuckstrian clearance before proceeding, with balanced reliability."""
        timeout = 120  # Extended maximum wait time for a PeDuckstrian response
        start_time = rospy.get_time()
        check_interval = 0.5  # Check interval
        consecutive_clear_detections = 0
        required_clear_detections = 4  # Increased number of consecutive clear detections required
        
        # Turn on LED to indicate waiting
        self.led_blinker.set_led_color("red")
        
        while (rospy.get_time() - start_time) < timeout and not rospy.is_shutdown():
            # Actively check for peduckstrians
            current_detection = self.peduck_detector.detect_peduckstrians()
            
            # Update our internal state
            self.peduck_detected = current_detection
            
            if not self.peduck_detected:
                consecutive_clear_detections += 1
                rospy.loginfo(f"Clear detection {consecutive_clear_detections}/{required_clear_detections}")
                
                if consecutive_clear_detections >= required_clear_detections:
                    rospy.loginfo("Crosswalk confirmed clear. Proceeding past the second blue line.")
                    self.led_blinker.set_led_color("green")  # Change LED to green
                    rospy.sleep(0.5)  # Brief pause for safety
                    self.navigator.move_straight(0.3)
                    return
            else:
                # Reset counter if a pedestrian is detected
                consecutive_clear_detections = 0
                rospy.logwarn("PeDuckstrian detected! Waiting...")
            
            rospy.sleep(check_interval)
        
        # If timeout reached
        rospy.logwarn("Timeout reached. Proceeding with caution.")
        self.led_blinker.blink_led("yellow")  # Change LED to yellow for caution
        rospy.sleep(0.5)  # Brief pause
        self.navigator.move_straight(0.3)

if __name__ == '__main__':
    node = CrosswalkBehaviour(node_name='crosswalk_behaviour_node')
    
    rate = rospy.Rate(10)  # 10 Hz
    node.execute()
    while not rospy.is_shutdown():
        rospy.sleep(0.1)

    rospy.loginfo("Behavior completed. Exiting program.")
