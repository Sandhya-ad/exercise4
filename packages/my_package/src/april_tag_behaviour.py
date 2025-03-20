#!/usr/bin/env python3

import os
import rospy
import time
import numpy as np
import cv2
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from preprocess_image import CameraReaderNode
from duckietown_msgs.msg import AprilTagDetectionArray
from led_service import LEDBlinker  # Import LED control service
from auto_nav_func import Navigator  # Import navigation functions
from red_detection import RedTapeDetectionNode  # Import red tape detection

class AprilTagRedTapeBehaviour(DTROS):

    def __init__(self, node_name):
        super(AprilTagRedTapeBehaviour, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)
        
        # Get vehicle name
        self._vehicle_name = os.environ.get('VEHICLE_NAME', 'default_duckiebot')

        # Define Topics
        self.preprocess = CameraReaderNode(node_name="preprocessed_image")
        self._detection_topic = f"/{self._vehicle_name}/camera_apriltag/detections"
        self._camera_topic = f"/{self._vehicle_name}/camera_undistorted/image/compressed"
        self._camera_info_topic = f"/{self._vehicle_name}/camera_node/camera_info"
        self.red_tape_detected = False  # Ensures we only detect red tape once


        # Initialize LED control service
        self.led_blinker = LEDBlinker()

        # Initialize Navigator for movement
        self.navigator = Navigator()

        # Initialize Red Tape Detection
        self.red_tape_detector = RedTapeDetectionNode(node_name="red_tape_detector")

        # Subscribe to AprilTag detections
        self.sub_apriltag = rospy.Subscriber(self._detection_topic, AprilTagDetectionArray, self.apriltag_callback, queue_size=5)

        self.stop_time = 0.5  # Default stop time if no AprilTag is detected
        self.stop_detected = False  # Flag to enable red tape detection
        self.red_tape_distance = None  # Store the detected red tape distance
        self.behavior_executed = False  # Flag to stop script after one full execution

        rospy.loginfo(f"[{node_name}] Initialized with AprilTag and Red Tape detection.")

        # **Step 1: Detect the Red Tape Distance Before Moving**
        self.detect_red_tape_distance()

        # **Step 3: Move Toward the Red Tape**
        if self.red_tape_distance:
            rospy.loginfo(f"Moving toward red tape at {self.red_tape_distance} meters...")
            self.navigator.move_straight(self.red_tape_distance-0.05) 
        self.stop_robot()  # Stop the robot after reaching red tape

    def detect_red_tape_distance(self):
        """Detects how far the red tape is BEFORE starting movement."""

        while self.red_tape_distance is None and not rospy.is_shutdown():
            self.red_tape_distance = self.red_tape_detector.detect_red_tape()
            rospy.sleep(0.5)

        rospy.loginfo(f" Red tape detected at {self.red_tape_distance} meters.")


    def apriltag_callback(self, msg):
        """Handles AprilTag detection, changes LED immediately, and keeps moving until red tape is reached."""
        
        if self.behavior_executed:
            return  # Ignore detections after one behavior execution

        # ** WAIT UNTIL AN IMAGE IS AVAILABLE BEFORE DETECTING**
        wait_time = 5  # Max wait time for detection
        elapsed_time = 0
        while (not msg.detections or len(msg.detections) == 0) and elapsed_time < wait_time:
            rospy.logwarn("Waiting for an AprilTag detection...")
            rospy.sleep(0.5)
            elapsed_time += 0.5
        
        if not msg.detections or len(msg.detections) == 0:
            rospy.logwarn("No AprilTag detected within wait time. Proceeding with default behavior.")
            return  # Skip further processing if no tag was found

        # **Define tag IDs for each category**
        STOP_SIGN_TAGS = [22, 162, 21]  # Stop Sign (3 sec stop)
        T_INTERSECTION_TAGS = [133]  # T-Intersection (2 sec stop)
        UOFA_TAGS = [93, 94, 201, 56]  # UofA Tag (1 sec stop)

        # **Process only the latest detected tag (last item in detections list)**
        latest_detection = msg.detections[-1]  # Get the most recent detection
        tag_id = latest_detection.tag_id

        # **Determine LED color & stop time based on the detected tag**
        if tag_id in STOP_SIGN_TAGS:
            color, self.stop_time = "red", 3
        elif tag_id in T_INTERSECTION_TAGS:
            color, self.stop_time = "blue", 2
        elif tag_id in UOFA_TAGS:
            color, self.stop_time = "green", 1
        else:
            color, self.stop_time = "white", 0.5  # Default state if unknown tag

        # **Change LED immediately**
        self.led_blinker.set_led_color(color)

        # **Enable red tape detection**
        self.stop_detected = True
        self.april_tag_seen = True  # Mark that a tag was detected



    def stop_robot(self):
        """Stops the robot at red tape for the time based on AprilTag detection."""
        rospy.loginfo(f"Stopping robot for {self.stop_time} seconds...")
        self.navigator.stop(self.stop_time)  # STOP MOVEMENT
        rospy.loginfo("Resuming movement.")

        # Move forward 30 cm after stopping
        rospy.loginfo("Moving forward 30 cm after stopping...")
        self.navigator.move_straight(0.30)

        # **Stop execution after one behavior completes**
        self.behavior_executed = True
        self.led_blinker.set_led_color("white")
        rospy.loginfo("Behavior executed. Stopping further detections.")

if __name__ == '__main__':
    node = AprilTagRedTapeBehaviour(node_name='aprilTagRedTapeBehaviour')
    
    # Continuously check if red tape has been reached
    rate = rospy.Rate(10)  # 10 Hz
    while not rospy.is_shutdown() and not node.behavior_executed:
        rospy.sleep(0.1)

    rospy.loginfo("Behavior completed. Exiting program.")