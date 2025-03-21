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
from detect_duckiebot import DuckiebotDetector #  Duckiebot distance estimation
from auto_nav_func import Navigator

class preprocessedImageNode(DTROS):

    def __init__(self, node_name):
        super(preprocessedImageNode, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)
        
        # Get vehicle name
        self._vehicle_name = os.environ.get('VEHICLE_NAME', 'default_duckiebot')

        #self.crosswalk_detector = CrosswalkDetection(node_name="crosswalk_detection_node")
        #self.peduck_detector = PeDuckstrianDetector(True)  # 🚶‍♂️ Detects PeDuckstrians
        self.duckiebot_detector = DuckiebotDetector(True)
        #navigator = Navigator()  # Controls movement
        #navigator.sharp_left()
        #navigator.sharp_right()


if __name__ == '__main__':
    node = preprocessedImageNode(node_name='preprocessedImageNode')
    rospy.spin()
