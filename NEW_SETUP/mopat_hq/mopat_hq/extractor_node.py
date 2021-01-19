#MoPAT Design Lab - Guining Pertin
#Extractor node(ROS2) - 19-01-21

'''
This node runs the camera, gets the raw image and the segmented image
Published topics:
    /mopat_hq/vision/raw_image          - sensor_msgs/Image (BGR)
    /mopat_hq/vision/segmented_image    - sensor_msgs/Image (Bool)
Work:
    Uses OpenCV to turn on camera and publish raw image.
    Uses OpenCV to segment out free space.
'''

#ROS
import rclpy
from rclpy.node import Node
from cv_bridge  import CvBridge
#ROS messages
from sensor_msgs.msg import Image
#Others
import numpy as np
import cv2

class extractor_node(Node):
    def __init__(self):
        super().__init__("extractor_node")
        self.get_logger().info("INIT")
        self.pub_raw = self.create_publisher(Image, "/mopat_hq/vision/raw_image", 1)
        self.pub_seg = self.create_publisher(Image, "/mopat_hq/vision/seg_image", 1)
        #Class variables
        self.bridge = CvBridge()        #CV-ROS bridge package

    def run(self, cam_index=0):
        '''
        Function to get raw image
        '''
        self.cap = cv2.VideoCapture(cam_index)
        while 1:
            #Read frame and publish
            _, raw_img = self.cap.read()
            raw_img_bridge = self.bridge.cv2_to_imgmsg(raw_img.astype(np.uint8),
                                                       encoding="passthrough")
            self.pub_raw.publish(raw_img_bridge)
            #Get segmented image and publish
            seg_img = self.segment_image(raw_img)
            seg_img_bridge = self.bridge.cv2_to_imgmsg(seg_img.astype(np.uint8),
                                                       encoding="passthrough")
            self.pub_seg.publish(seg_img_bridge)
            #For closing session
            if cv2.waitKey(1) == ord("q"): break
        #Close cam
        self.cap.release()
        cv2.destroyAllWindows()

    def segment_image(self, frame):
        '''
        Performs segmentation on the given image
        '''
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        _, seg_frame = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((10,10), np.uint8)
        seg_frame = cv2.morphologyEx(seg_frame, cv2.MORPH_OPEN, kernel)
        #Convert to bool
        return seg_frame.astype(bool)

def main(args=None):
    rclpy.init()
    #Create and run
    create_node = extractor_node()
    try:
        rclpy.spin(create_node)
    except KeyboardInterrupt:
        create_node.get_logger().info("EXIT")
        #Close node on exit
        create_node.cap.release()
        cv2.destroyAllWindows()
        create_node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
