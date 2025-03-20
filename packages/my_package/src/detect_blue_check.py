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

        self.crosswalk_detector = LaneDetectionNode(node_name="crosswalk_detection_node")
        self.peduck_detector = PeDuckstrianDetector()  # 🚶‍♂️ Detects PeDuckstrians



if __name__ == '__main__':
    node = preprocessedImageNode(node_name='preprocessedImageNode')
    rospy.spin()
