#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import os

class PeDuckstrianDetector:
    def __init__(self):
        self._vehicle_name = os.environ.get('VEHICLE_NAME', 'default_duckiebot')
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        self._clear_topic = f"/{self._vehicle_name}/peduckstrian_detection/clear"
        
        self.bridge = CvBridge()
        self.latest_image = None
        self.last_detection_state = None
        
        # Detection parameters - taking inspiration from the provided code
        self.detection_threshold = 2  # Require multiple detections for stability
        self.recent_detections = [False] * self.detection_threshold  # Track recent detections
        
        # Set debug mode (set to True to see detection visualization)
        self.debug = rospy.get_param('~debug', False)
        
        rospy.Subscriber(self._camera_topic, CompressedImage, self.image_callback, queue_size=1)
        self.pub_clear = rospy.Publisher(self._clear_topic, Bool, queue_size=1)
        self.yellow_lower = np.array([20, 100, 150])
        self.yellow_upper = np.array([30, 255, 255])

        self.red_lower1 = np.array([0, 120, 100])
        self.red_upper1 = np.array([10, 255, 255])  # Red at lower end of HSV spectrum

        self.red_lower2 = np.array([170, 120, 100])
        self.red_upper2 = np.array([180, 255, 255])  # Red at higher end


        
        rospy.loginfo("PeDuckstrian Detector Initialized with balanced parameters")
    
    def image_callback(self, msg):
        """Processes the image and detects PeDuckstrians."""
        try:
            self.latest_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.detect_peduckstrians()
        except Exception as e:
            rospy.logerr(f"Error processing image for PeDuckstrian detection: {e}")
    
    def detect_peduckstrians(self):
        """Detects PeDuckstrians based on yellow color filtering + red beak detection."""
        if self.latest_image is None:
            return False

        # Create a larger region of interest - bottom 60% of image instead of 40%
        height, width = self.latest_image.shape[:2]
        roi_y = int(height * 0.4)  # Changed from 0.6 to look at more of the image
        roi = self.latest_image[roi_y:height, 0:width]

        # Convert to HSV
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Broader yellow range
        yellow_mask = cv2.inRange(hsv, np.array([15, 80, 120]), np.array([35, 255, 255]))
        
        # Broader red range
        red_mask1 = cv2.inRange(hsv, np.array([0, 100, 80]), np.array([15, 255, 255]))
        red_mask2 = cv2.inRange(hsv, np.array([165, 100, 80]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        # Filter out noise
        kernel = np.ones((3, 3), np.uint8)  # Smaller kernel for less aggressive filtering
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

        # Find yellow contours
        contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ducks_detected = 0

        for contour in contours:
            # Lower threshold for smaller ducks
            if cv2.contourArea(contour) < 100:  # Reduced from 500
                continue
            
            # Create bounding box around the yellow region
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check if red is present in the same region
            red_roi = red_mask[y:y+h, x:x+w]
            red_pixels = np.sum(red_roi > 0)

            if red_pixels > 2:  # Reduced from 4 to be more sensitive
                ducks_detected += 1
                if self.debug:
                    # Draw bounding box around detected duck
                    cv2.rectangle(roi, (x, y), (x+w, y+h), (0, 255, 0), 2)


        # Update detection state
        current_detection = ducks_detected > 0
        self.recent_detections.pop(0)
        self.recent_detections.append(current_detection)
        stable_detection = sum(self.recent_detections) > (self.detection_threshold // 2)

        if stable_detection != self.last_detection_state:
            rospy.loginfo(f"PeDuckstrian detection state changed to: {stable_detection}")
            self.pub_clear.publish(Bool(not stable_detection))  # Not detected means clear
            self.last_detection_state = stable_detection

        return stable_detection

if __name__ == "__main__":
    rospy.init_node("peduckstrian_detector")
    node = PeDuckstrianDetector()
    rospy.spin()