#!/usr/bin/env python3

import os
import rospy
import numpy as np
import cv2
from sensor_msgs.msg import CompressedImage, CameraInfo
from std_msgs.msg import Float32
from cv_bridge import CvBridge
from auto_nav_func import Navigator  # Import movement control
from detect_peDuck import PeDuckstrianDetector  # PeDuckstrian detection

class CrosswalkDetection:

    def __init__(self, node_name):
        self._vehicle_name = os.environ.get('VEHICLE_NAME', 'default_duckiebot')

        # Camera topics
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        self._camera_info_topic = f"/{self._vehicle_name}/camera_node/camera_info"
        self.crosswalk_found = False
        self.crosswalk_distance = None

        # PeDuckstrian detection
        self.peduck_detected = False  
        self.peduck_detector = PeDuckstrianDetector()

        self.bridge = CvBridge()
        
        # Subscribers
        self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.image_callback, queue_size=1, buff_size=2**24)
        self.sub_info = rospy.Subscriber(self._camera_info_topic, CameraInfo, self.camera_info_callback)

        # Publisher for crosswalk distance
        self.pub_distance = rospy.Publisher(f"/{self._vehicle_name}/crosswalk_distance", Float32, queue_size=1)

        # Camera parameters
        self.focal_length_px = None  
        self.camera_matrix = None
        self.dist_coeffs = None
        self.latest_image = None  

        # Improved Blue Line HSV range
        self.blue_lower = np.array([85, 50, 50])  
        self.blue_upper = np.array([130, 255, 255])

        # Movement
        self.navigator = Navigator()
        self.moving_forward = False  

        rospy.loginfo(f"[{node_name}] Initialized and looking for blue lanes.")

    def camera_info_callback(self, msg):
        """Receives camera parameters."""
        self.camera_matrix = np.array(msg.K).reshape((3, 3))
        self.dist_coeffs = np.array(msg.D)
        self.focal_length_px = self.camera_matrix[0, 0]  

    def image_callback(self, msg):
        """Processes images to detect the crosswalk."""
        # Skip processing if we've already found the crosswalk
        if self.crosswalk_found:
            return
            
        try:
            raw_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.latest_image = self.undistort_image(raw_image)

            # Detect crosswalk
            detected, distance = self.detect_crosswalk()

            if detected:
                self.crosswalk_found = True  # Set the flag
                self.crosswalk_distance = distance  # Store the distance
                rospy.loginfo(f"Blue line detected at {distance} meters. Stopping detection.")
                self.pub_distance.publish(Float32(distance))
                
                # Check for PeDuckstrians immediately
                self.peduck_detected = self.peduck_detector.detect_peduckstrians()
                
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")

    def undistort_image(self, image):
        """Undistorts image using camera calibration."""
        if self.camera_matrix is None or self.dist_coeffs is None:
            rospy.logwarn("Camera parameters not received yet! Returning original image.")
            return image

        h, w = image.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (w, h), alpha=0.2, newImgSize=(w, h)
        )

        undistorted = cv2.undistort(image, self.camera_matrix, self.dist_coeffs, None, new_camera_matrix)
        x, y, w, h = roi
        return undistorted[y:y+h, x:x+w] if w > 0 and h > 0 else undistorted

    def detect_crosswalk(self):
        """Detects blue lines and estimates distance."""
        if self.latest_image is None:
            return False, None

        hsv = cv2.cvtColor(self.latest_image, cv2.COLOR_BGR2HSV)

        # Improve detection with contrast enhancement
        lab = cv2.cvtColor(self.latest_image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        enhanced_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        hsv = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2HSV)

        # Apply blue mask
        mask = cv2.inRange(hsv, self.blue_lower, self.blue_upper)

        # Noise reduction
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            rospy.logwarn("No blue line detected.")
            return False, None
            
        # Filter contours by aspect ratio to find horizontal lines
        valid_contours = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Look for wide, thin rectangles (horizontal lines)
            aspect_ratio = float(w) / h if h > 0 else 0
            area = cv2.contourArea(contour)
            
            # Filter based on aspect ratio and minimum area
            if aspect_ratio > 3.0 and area > 100:  # Line should be wider than tall
                valid_contours.append(contour)
        
        if not valid_contours:
            return False, None
        
        # Sort by area
        valid_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)
        largest_line = valid_contours[0]
        
        # Get the bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_line)
        
        # Calculate distance using the width of the line
        # Apply a correction factor based on testing
        correction_factor = 0.7  # Adjust this value based on calibration tests
        distance = self.estimate_distance(w) * correction_factor
        
        return True, distance

    def estimate_distance(self, object_pixel_width):
        """Estimates the distance using the detected blue line's width."""
        if self.focal_length_px is None:
            rospy.logwarn("Focal length not set! Cannot estimate distance.")
            return None

        # Use the actual width of the crosswalk line
        real_width = 0.60  # 60 cm
        
        # Calculate raw distance and apply sanity checks
        raw_distance = (real_width * self.focal_length_px) / object_pixel_width
        
        # Limit distance to reasonable range (0.1 to 3.0 meters)
        distance = max(0.1, min(raw_distance, 3.0))
        
        return round(distance, 2)

    def get_crosswalk_distance(self):
        """Returns the detected distance to the blue line or None if not found."""
        if self.crosswalk_found:
            return self.crosswalk_distance
        
        # Try to detect if not previously found
        detected, distance = self.detect_crosswalk()
        if detected:
            self.crosswalk_found = True
            self.crosswalk_distance = distance
            rospy.loginfo(f"Closest blue line detected at {distance} meters.")
            return distance
        else:
            return None


if __name__ == '__main__':
    rospy.init_node('crosswalk_detection_node', anonymous=False)
    node = CrosswalkDetection(node_name='crosswalk_detection_node')
    rospy.spin()
