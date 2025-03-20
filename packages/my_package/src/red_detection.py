#!/usr/bin/env python3

import os
import rospy
import numpy as np
import cv2
from sensor_msgs.msg import CompressedImage, CameraInfo
from cv_bridge import CvBridge

class RedTapeDetectionNode:
    def __init__(self, node_name):
        # Get vehicle name
        self._vehicle_name = os.environ.get("VEHICLE_NAME", "default_duckiebot")

        # Define camera topics
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        self._camera_info_topic = f"/{self._vehicle_name}/camera_node/camera_info"
        self.latest_image = None  # Store the most recent image for processing

        # OpenCV Bridge
        self._bridge = CvBridge()

        # Subscribe to camera topic
        self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.image_callback, queue_size=1, buff_size=2**24)
        self.sub_info = rospy.Subscriber(self._camera_info_topic, CameraInfo, self.camera_info_callback)

        self.focal_length_px = None  # Will be set from camera_info
        self.camera_matrix = None
        self.dist_coeffs = None

        rospy.loginfo(f"RedTapeDetectionNode initialized. Subscribing to {self._camera_topic}")

        # Define HSV Color Range for Red
        self.red_lower1 = np.array([0, 100, 100])
        self.red_upper1 = np.array([10, 255, 255])
        self.red_lower2 = np.array([170, 100, 100])
        self.red_upper2 = np.array([180, 255, 255])

    def image_callback(self, msg):
        """Receives and stores the latest image from the camera."""
        try:
            self.latest_image = self._bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logerr(f"Error converting image: {e}")
            self.latest_image = None

    def camera_info_callback(self, msg):
        """Receives camera intrinsic parameters and stores them."""
        self.camera_matrix = np.array(msg.K).reshape((3, 3))
        self.dist_coeffs = np.array(msg.D)

        # Get the new optimal camera matrix after undistortion
        h, w = msg.height, msg.width
        new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (w, h), alpha=0.2, newImgSize=(w, h)
        )

        # Use the new focal length from the corrected matrix
        self.focal_length_px = new_camera_matrix[0, 0]  # Use the new fx value

    def undistort_image(self, image):
        """Applies undistortion using stored camera parameters."""
        if self.camera_matrix is None or self.dist_coeffs is None:
            rospy.logwarn("Camera parameters not received yet! Returning original image.")
            return image

        h, w = image.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (w, h), alpha=0.2, newImgSize=(w, h)
        )

        # Apply undistortion
        undistorted = cv2.undistort(image, self.camera_matrix, self.dist_coeffs, None, new_camera_matrix)

        # Crop out black borders (use ROI)
        x, y, w, h = roi
        if w > 0 and h > 0:
            undistorted = undistorted[y:y+h, x:x+w]

        return undistorted

    def detect_red_tape(self):
        """Waits for an image, detects red tape in the undistorted image, and returns distance."""
        # **Wait until an image is available**
        while self.latest_image is None and not rospy.is_shutdown():
            rospy.logwarn("Waiting for an image from the camera...")
            rospy.sleep(0.5)

        # **Ensure latest_image is valid before processing**
        if self.latest_image is None:
            rospy.logwarn("No valid image received. Cannot detect red tape.")
            return None

        # **Undistort and convert to HSV**
        image = self.undistort_image(self.latest_image)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # **Create masks for red color**
        red_mask1 = cv2.inRange(hsv, self.red_lower1, self.red_upper1)
        red_mask2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        # **Apply morphological operations to remove noise**
        kernel = np.ones((5, 5), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

        # **Find contours of red tape**
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            _, _, w, _ = cv2.boundingRect(largest_contour)  # Get width

            if w > 7:  # Ensure object is large enough
                return self.estimate_distance(w)  # Return estimated distance

        return None  # No red tape detected

    def estimate_distance(self, object_pixel_width):
        """Estimates the distance to the red tape using the pinhole camera model."""
        if object_pixel_width == 0 or self.focal_length_px is None:
            rospy.logwarn("Invalid object width or focal length missing.")
            return None

        known_object_width_m = 0.27  # 25 cm (corrected value)

        # Ensure the focal length is properly calibrated
        if self.focal_length_px < 100:  # If it's unreasonably low
            rospy.logwarn("Focal length seems too low! Using default 500px.")
            self.focal_length_px = 500  # Fallback default

        # Compute distance
        distance_m = (known_object_width_m * self.focal_length_px) / object_pixel_width
        return round(distance_m, 2)


if __name__ == "__main__":
    rospy.init_node("red_tape_detection_node")
    node = RedTapeDetectionNode(node_name="red_tape_detection_node")
    distance = node.detect_red_tape()

    if distance:
        rospy.loginfo(f" Red tape detected at {distance} meters.")
    else:
        rospy.logwarn("⚠️ No red tape detected.")
