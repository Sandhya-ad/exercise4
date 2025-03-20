import os
import rospy
import numpy as np
import cv2
import threading
import dt_apriltags
from sensor_msgs.msg import CompressedImage, CameraInfo
from cv_bridge import CvBridge
from duckietown_msgs.msg import AprilTagDetectionArray, AprilTagDetection
from geometry_msgs.msg import Transform, Vector3, Quaternion


class CameraReaderNode:

    def __init__(self, node_name):
        # Static parameters
        self._vehicle_name = os.environ.get('VEHICLE_NAME', 'default_duckiebot')
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        self._processed_image_topic = f"/{self._vehicle_name}/camera_preprocessed/image/compressed"
        self._augmented_image_topic = f"/{self._vehicle_name}/camera_apriltag/image/compressed"
        self._detection_topic = f"/{self._vehicle_name}/camera_apriltag/detections"
        self._camera_info_topic = f"/{self._vehicle_name}/camera_node/camera_info"

        # AprilTag Detector
        self.detector = dt_apriltags.Detector(families='tag36h11')

        # Bridge for OpenCV <-> ROS
        self._bridge = CvBridge()

        # Threading locks & latest message storage
        self.latest_msg = None
        self.lock = threading.Lock()

        # Subscribers
        self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.image_callback, queue_size=5, buff_size=2**24)
        self.sub_info = rospy.Subscriber(self._camera_info_topic, CameraInfo, self.camera_info_callback)

        # Publishers
        self.pub = rospy.Publisher(self._processed_image_topic, CompressedImage, queue_size=5)
        self.pub_augmented = rospy.Publisher(self._augmented_image_topic, CompressedImage, queue_size=5)
        self.pub_detections = rospy.Publisher(self._detection_topic, AprilTagDetectionArray, queue_size=5)  # New topic for tag messages

        # Start processing thread
        self.processing_thread = threading.Thread(target=self.process_image, daemon=True)
        self.processing_thread.start()

        rospy.loginfo(f"[{node_name}] Initialized and subscribing to {self._camera_topic}")

    def camera_info_callback(self, msg):
        """ Receives camera intrinsic parameters. """
        self.camera_matrix = np.array(msg.K).reshape((3, 3))
        self.dist_coeffs = np.array(msg.D)

    def image_callback(self, msg):
        """ Stores the latest received image for processing. """
        with self.lock:
            self.latest_msg = msg  # Always store the latest frame

    def process_image(self):
        """ Continuously processes the latest available image. """
        rate = rospy.Rate(10)  # Run at 10Hz

        while not rospy.is_shutdown():
            with self.lock:
                if self.latest_msg is None:
                    rate.sleep()
                    continue

                # Get the latest image and reset storage
                msg = self.latest_msg
                self.latest_msg = None  

            try:
                # Convert ROS image to OpenCV format
                image = self._bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")

                # Undistort and preprocess the image
                undistorted_image = self.undistort_image(image)
                preprocessed_image = self.preprocess_image(undistorted_image)

                # Detect and annotate AprilTags + Publish detections
                annotated_image = self.detect_apriltags(preprocessed_image, msg.header)

                # Encode back to ROS CompressedImage
                _, img_encoded = cv2.imencode('.jpg', preprocessed_image)
                processed_msg = CompressedImage()
                processed_msg.header = msg.header
                processed_msg.format = "jpeg"
                processed_msg.data = np.array(img_encoded).tobytes()

                # Encode and publish annotated image
                _, img_annotated_encoded = cv2.imencode('.jpg', annotated_image)
                annotated_msg = CompressedImage()
                annotated_msg.header = msg.header
                annotated_msg.format = "jpeg"
                annotated_msg.data = np.array(img_annotated_encoded).tobytes()
                self.pub_augmented.publish(annotated_msg)

                # Publish the processed image
                self.pub.publish(processed_msg)
                rospy.loginfo("Processed and annotated images published.")

            except Exception as e:
                rospy.logerr(f"Error processing image: {e}")

            rate.sleep()  # Control processing speed

    def undistort_image(self, image):
        """ Applies undistortion using camera parameters. """
        if self.camera_matrix is None or self.dist_coeffs is None:
            rospy.logwarn("Camera parameters not received yet!")
            return image

        h, w = image.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (w, h), alpha=0.2, newImgSize=(w, h)
        )

        undistorted = cv2.undistort(image, self.camera_matrix, self.dist_coeffs, None, new_camera_matrix)

        # Crop out black borders (use ROI)
        x, y, w, h = roi
        if w > 0 and h > 0:
            undistorted = undistorted[y:y+h, x:x+w]

        return undistorted
    
    def preprocess_image(self, image):
        """ Preprocesses the image to improve AprilTag detection. """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Get image dimensions
        h, w = gray.shape

        # Define cropping percentages
        crop_top = int(0.10 * h)  
        crop_bottom = int(0.25 * h)  
        crop_side = int(0.15 * w)  

        # Apply cropping
        cropped_gray = gray[crop_top:h - crop_bottom, crop_side:w - crop_side]

        # Apply Gaussian Blur
        blurred = cv2.GaussianBlur(cropped_gray, (3, 3), 0)

        return blurred

    def detect_apriltags(self, image, header):
        """ Detects AprilTags, draws bounding boxes, and publishes detections. """
        tags = self.detector.detect(image)

        # Convert grayscale to BGR for color annotations
        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Prepare message for detections
        tags_msg = AprilTagDetectionArray()
        tags_msg.header = header

        for tag in tags:
            # Convert detection to message
            detection = AprilTagDetection(
                tag_id=tag.tag_id,
                tag_family=str(tag.tag_family),
                hamming=tag.hamming,
                decision_margin=tag.decision_margin,
                center=tag.center.tolist(),
                corners=tag.corners.flatten().tolist()
            )
            tags_msg.detections.append(detection)

            # Draw bounding box
            for i in range(4):
                pt1 = tuple(map(int, tag.corners[i]))
                pt2 = tuple(map(int, tag.corners[(i + 1) % 4]))
                cv2.line(image_bgr, pt1, pt2, (0, 255, 0), 2)

            # Draw tag ID
            center = tuple(map(int, tag.center))
            cv2.circle(image_bgr, center, 5, (255, 0, 0), -1)
            cv2.putText(image_bgr, str(tag.tag_id), (center[0] - 10, center[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Publish detections
        self.pub_detections.publish(tags_msg)

        return image_bgr

if __name__ == '__main__':
    rospy.init_node('camera_reader_node', anonymous=False)
    node = CameraReaderNode(node_name='camera_reader_node')
    rospy.spin()
