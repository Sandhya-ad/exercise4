#!/usr/bin/env python3

import os
import rospy
import numpy as np
import cv2
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage, CameraInfo
from cv_bridge import CvBridge
from preprocess_image import CameraReaderNode
from crosswalk_detection import CrosswalkDetection  # Crosswalk detection
from detect_peDuck import PeDuckstrianDetector  # 🚶‍♂️ PeDuckstrian detection

class preprocessedImageNode(DTROS):

    def __init__(self, node_name):
        super(preprocessedImageNode, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)
        
        # Get vehicle name
        self._vehicle_name = os.environ.get('VEHICLE_NAME', 'default_duckiebot')

        # Define Topics
        self.preprocess = CameraReaderNode(node_name="preprocessed_image")
        self._camera_topic = f"/{self._vehicle_name}/camera_preprocessed/image/compressed"
        self._camera_info_topic = f"/{self._vehicle_name}/camera_node/camera_info"

        # OpenCV Bridge
        self._bridge = CvBridge()
        self.latest_image = None  # Store the latest received image
        self.peduck_detector = PeDuckstrianDetector()  # 🚶‍♂️ Detects PeDuckstrians


        # Camera calibration parameters
        self.camera_matrix = None
        self.dist_coeffs = None
        self.focal_length_px = None  # Will be set in camera_info_callback


if __name__ == '__main__':
    node = preprocessedImageNode(node_name='preprocessedImageNode')
    rospy.spin()
