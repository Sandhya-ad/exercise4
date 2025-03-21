#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import os
from sensor_msgs.msg import CompressedImage, CameraInfo
from std_msgs.msg import Bool, Float32
from cv_bridge import CvBridge

class DuckiebotDistanceEstimator:
    def __init__(self):
        self.focal_length = None
        self.real_width = 4.0  # cm
        self.history = []  # Store last N distances
        self.history_size = 5  # Use last 5 values for smoothing

    def set_camera_info(self, camera_matrix):
        self.focal_length = camera_matrix[0, 0]  # fx

    def estimate_distance(self, pixel_width):
        if self.focal_length is None or pixel_width <= 5:  # Avoid division by small widths
            return float('inf')

        distance = (self.focal_length * self.real_width) / pixel_width

        # Add to history and smooth
        self.history.append(distance)
        if len(self.history) > self.history_size:
            self.history.pop(0)  # Keep only last N values

        return np.mean(self.history)  # Return smoothed value


class DuckiebotDetector:
    def __init__(self, debug=False):
        self._vehicle_name = os.environ.get('VEHICLE_NAME', 'default_duckiebot')
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        self._camera_info_topic = f"/{self._vehicle_name}/camera_node/camera_info"
        self._detection_topic = f"/{self._vehicle_name}/duckiebot_detection/clear"
        self._distance_topic = f"/{self._vehicle_name}/duckiebot_detection/distance"

        self.bridge = CvBridge()
        self.latest_image = None
        self.last_detection_state = None
        self.debug = debug

        self.camera_matrix = None
        self.dist_coeffs = None

        self.detection_threshold = 2
        self.recent_detections = [False] * self.detection_threshold

        #  **New Publisher for the Processed Image**

        if debug:
            self._debug_topic = f"/{self._vehicle_name}/camera_ducks/image/compressed"
            self.pub_debug = rospy.Publisher(self._debug_topic, CompressedImage, queue_size=5)

        rospy.Subscriber(self._camera_topic, CompressedImage, self.image_callback, queue_size=1)
        rospy.Subscriber(self._camera_info_topic, CameraInfo, self.camera_info_callback)
        self.pub_clear = rospy.Publisher(self._detection_topic, Bool, queue_size=1)
        self.pub_distance = rospy.Publisher(self._distance_topic, Float32, queue_size=1)

        self.distance_estimator = DuckiebotDistanceEstimator()

        # Blob Detector
        params = cv2.SimpleBlobDetector_Params()
        params.filterByColor = False
        params.filterByArea = True
        params.minArea = 3
        params.minDistBetweenBlobs = 3
        params.maxArea = 80
        params.filterByCircularity = True
        params.minCircularity = 0.85
        params.filterByConvexity = True
        params.minConvexity = 0.8
        params.filterByInertia = True
        params.minInertiaRatio = 0.5
        self.detector = cv2.SimpleBlobDetector_create(params)

        rospy.loginfo("Duckiebot Detector Initialized")
    def camera_info_callback(self, msg):
        self.camera_matrix = np.array(msg.K).reshape((3, 3))
        self.dist_coeffs = np.array(msg.D)
        self.distance_estimator.set_camera_info(self.camera_matrix)


    def image_callback(self, msg):
        try:
            image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
            if self.debug:
                image = self.undistort_image(image)
            self.latest_image = image
            self.detect_duckiebot()
        except Exception as e:
            rospy.logerr(f"Error processing image for Duckiebot detection: {e}")

    def undistort_image(self, image):
        if self.camera_matrix is None or self.dist_coeffs is None:
            rospy.logwarn_throttle(5, "Camera parameters not received yet. Skipping undistortion.")
            return image

        h, w = image.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (w, h), alpha=0.2, newImgSize=(w, h)
        )
        undistorted = cv2.undistort(image, self.camera_matrix, self.dist_coeffs, None, new_camera_matrix)

        x, y, w, h = roi
        if w > 0 and h > 0:
            undistorted = undistorted[y:y+h, x:x+w]
        else:
            rospy.logwarn("Invalid ROI for undistortion. Using full image.")

        # Optional: crop bottom
        crop_bottom = int(0.15 * h)
        undistorted = undistorted[:h - crop_bottom, :]

        return undistorted

    def preprocess_image(self, image):
        """Preprocess image by cropping, contrast enhancement, and noise reduction."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        crop_top = int(0.3 * h)      # Remove top 20%
        crop_bottom = int(0.4 * h)   # Remove bottom 20% (symmetrical)
        crop_side = int(0.4 * w)     # Remove 20% from left and right sides

        cropped_gray = gray[crop_top:h - crop_bottom, crop_side:w - crop_side]


        # === Step 2: Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) ===
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(cropped_gray)

        # === Step 3: Adaptive Thresholding for Binary Image ===
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 10
        )

        # === Step 4: Morphological Processing (More Gentle) ===
        kernel = np.ones((2, 2), np.uint8)  # Reduce kernel size to avoid removing small dots
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)  # Reduce iterations
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        # === Step 5: Keep only circular blobs (filter everything else) ===
        mask = np.zeros_like(binary)  # Start with empty image

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 5 or area > 200:
                continue  # Skip too small or too large

            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if circularity > 0.7:
                cv2.drawContours(mask, [cnt], -1, 255, -1)  # Keep this blob

        binary = mask  # Only valid circular blobs remain


        return binary




    def detect_duckiebot(self):
        if self.latest_image is None:
            return False

        processed = self.preprocess_image(self.latest_image)
        
        # **Publish the Processed Image**

        keypoints = self.detector.detect(processed)
        detected, pattern_pixel_width = self.validate_pattern(keypoints)

        if self.debug:
            debug_image = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)  # Convert binary to BGR for drawing
            for kp in keypoints:
                x, y = int(kp.pt[0]), int(kp.pt[1])
                cv2.circle(debug_image, (x, y), 5, (0, 255, 0), -1)
            debug_msg = self.bridge.cv2_to_compressed_imgmsg(debug_image)
            self.pub_debug.publish(debug_msg)


        if detected and pattern_pixel_width > 0:
            distance = self.distance_estimator.estimate_distance(pattern_pixel_width)
            rospy.loginfo(f"Estimated distance to Duckiebot: {distance:.2f} cm")
            self.pub_distance.publish(Float32(distance))

        self.recent_detections.pop(0)
        self.recent_detections.append(detected)
        stable_detection = sum(self.recent_detections) > (self.detection_threshold // 2)

        if stable_detection != self.last_detection_state:
            rospy.loginfo(f"Duckiebot detection state changed to: {stable_detection}")
            self.pub_clear.publish(Bool(not stable_detection))
            self.last_detection_state = stable_detection

        return stable_detection


    def validate_pattern(self, keypoints):
        if len(keypoints) < 7:
            return False, 0
        points = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints])
        min_x = np.min(points[:, 0])
        max_x = np.max(points[:, 0])
        return True, max_x - min_x


if __name__ == "__main__":
    rospy.init_node("duckiebot_detector")
    node = DuckiebotDetector(debug=True)
    rospy.spin()
