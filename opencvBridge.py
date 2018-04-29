#!/usr/bin/env python
import roslib
import sys
import rospy
import cv2
import numpy as np
import std_msgs.msg
from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Point32
from cv_bridge import CvBridge, CvBridgeError     
      
def publishImage(filename):
	# Bridge function
	bridge = CvBridge()
	
	# Declaring Pointcloud. Will be used to publish to ROS
	myPointCloud = PointCloud()
    #filling pointcloud header
	header = std_msgs.msg.Header()
	header.stamp = rospy.Time.now()
	header.frame_id = 'map'
	myPointCloud.header = header
	
	# Read image file
	image_file = cv2.imread(filename)
	# Convert openCV image to publishable ROS image using openCV bridge
	image_message = bridge.cv2_to_imgmsg(image_file, encoding="bgr8")
	
	# Publisher init
	image_pub = rospy.Publisher("myImageTopic/image_raw", Image, queue_size = 1)
	point_pub = rospy.Publisher("/myPointCloud", PointCloud, queue_size = 1)
	rate=rospy.Rate(3)
	
	# Define the color threshold for houghLine detection
	lower_red = np.array([0,0,200])
	upper_red = np.array([10,40,255])
    	
	# Find red houghLines from image
	mask = cv2.inRange(image_file, lower_red, upper_red)
	
	# Extract location data of houghLine pixels
	coords = cv2.findNonZero(mask)
	
	# Now use openCV to translate Screen coordinates to World Coordinates
	
	# 2D image points
	screen_points = np.array([
							(585, 443), # Straight ahead
							(683, 442), # Top right
							(1160, 586),# Bottom Right
							(3, 603),	# Bottom Left
							(466, 442)	# Top Left
							], dtype="double")
							
#	cv2.imshow("Image window", image_file)
#	cv2.waitKey(0)
							
	# 3D model points (coordinates in meters)
	world_points = np.array([
							(0.0, 9.0, 0.0),
							(5.0, 9.0, 0.0),
							(5.0, 1.0, 0.0),
							(-5.0, 1.0, 0.0),
							(-5.0, 9.0, 0.0)
							])
	
	# Create a camera matrix
	size = image_file.shape
	focal_length = size[1]
	center = (size[1]/2, size[0]/2)
	cameraMatrix = np.array(
						[[focal_length, 0, center[0]],
						[0, focal_length, center[1]],
						[0, 0, 1]], dtype = "double")
							
	# Assume no lens distortion (even though there is)
	distCoeffs = np.zeros((4,1)) 
	
	# Use PnP to solve for vectors
	retVal, rotationVecs, translationVecs = cv2.solvePnP(world_points, screen_points, cameraMatrix, distCoeffs)
	
	# Convert rotation vector to matrix for calculations
	rotationMat, jacobian = cv2.Rodrigues(rotationVecs)
	
	# Using intrinsics and the mask, convert 2D points to 3D
	
	# Inverse matrices
	cameraMatrixInv = np.linalg.inv(cameraMatrix)
	rotationMatInv = np.linalg.inv(rotationMat)
	
	# Using s(u,v) = M(R(X,Y,Z) + T)
	for i in range(len(coords)):
		uvPoint = np.matrix([[coords[i][0][0]], [coords[i][0][1]], [1.0]])
		tempMat = np.matmul(rotationMatInv, np.matmul(cameraMatrixInv, uvPoint))
		tempMat2 = np.matmul(rotationMatInv, translationVecs)
		s = tempMat2[2,0]
		s /= tempMat[2,0]
		worldPoint = rotationMatInv * (s * (cameraMatrixInv * uvPoint) - translationVecs)
		myPointCloud.points.append(Point32(worldPoint[0], worldPoint[1], 0))
		#print worldPoint
    
	while not rospy.is_shutdown():
		# Publish image
		image_pub.publish(image_message)
		# Publish point cloud
		point_pub.publish(myPointCloud)
		rate.sleep()
		
if __name__ == '__main__':
    try:
    	rospy.init_node('OpenCV_and_ROS')
        publishImage(sys.argv[1])
    except rospy.ROSInterruptException:
        pass
