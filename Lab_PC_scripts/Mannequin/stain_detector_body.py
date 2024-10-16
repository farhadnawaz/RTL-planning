#!/usr/bin/python3
import rospy
from geometry_msgs.msg import PoseStamped
import numpy as np
import tf
from tf.transformations import *
import pyrealsense2 as rs
import cv2
import time
from numpy.linalg import inv
from std_msgs.msg import Float64MultiArray, Bool

BODY_BOTTOM = 0.55
BODY_UP = 0.3
BODY_LEFT = 0.27
BODY_RIGHT = 0.57
BODY_LOW = 0.2
BODY_HIGH = 0.3

def pose2tf(msg):
    x = msg.pose.position.x
    y = msg.pose.position.y
    z = msg.pose.position.z
    xo = msg.pose.orientation.x
    yo = msg.pose.orientation.y
    zo = msg.pose.orientation.z
    wo = msg.pose.orientation.w

    ret = quaternion_matrix([xo, yo, zo, wo])
    ret[0, 3] = x
    ret[1, 3] = y
    ret[2, 3] = z
    
    return ret


def rs_init():
    pipeline = rs.pipeline()
    config = rs.config()
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))
    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())

    return pipeline, color_image


def stain_det(image0):

    # gray = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)
    # blur = cv2.GaussianBlur(gray, (7, 7), 0)

    # _, thresh = cv2.threshold(blur, THRESH, 255, cv2.THRESH_BINARY_INV)
    # analysis = cv2.connectedComponentsWithStats(thresh, 4, cv2.CV_32S) 
    # output = np.zeros(gray.shape, dtype="uint8") 
    
    hsv = cv2.cvtColor(image0, cv2.COLOR_BGR2HSV)
    blur = cv2.GaussianBlur(hsv, (5, 5), 0)

    # blue
    lower_blue = np.array([80, 50, 50])
    upper_blue = np.array([140, 255, 255])

    # lower_blue = np.array([10, 30, 50])
    # upper_blue = np.array([40, 255, 180])
    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # Apply morphological operations (optional)
    # kernel = np.ones((5,5),np.uint8)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # Find connected components
    analysis = cv2.connectedComponentsWithStats(mask, connectivity=8)
    output = np.zeros(mask.shape, dtype="uint8") 

    (totalLabels, label_ids, values, centroid) = analysis 
    print(">>>>>", totalLabels)
    
    center_ret = []
    top_left_ret = []
    bottom_right_ret = []

    for i in range(1, totalLabels): 
        area = values[i, cv2.CC_STAT_AREA] 
        # define the area space
        if area > 50 and area < 10000:
            x1 = values[i, cv2.CC_STAT_LEFT] 
            y1 = values[i, cv2.CC_STAT_TOP] 
            w = values[i, cv2.CC_STAT_WIDTH] 
            h = values[i, cv2.CC_STAT_HEIGHT] 
            pt1 = (x1, y1) 
            pt2 = (x1+ w, y1+ h) 
            (X, Y) = centroid[i] 
            center_ret.append(centroid[i])
            top_left_ret.append(pt1)
            bottom_right_ret.append(pt2)
            cv2.rectangle(image0,pt1,pt2, (0, 255, 0), 1) 
            cv2.circle(image0, (int(X), int(Y)), 4, (0, 0, 255), -1)
            componentMask = (label_ids == i).astype("uint8") * 255
            output = cv2.bitwise_or(output, componentMask) 
  
    # cv2.imshow("Image", img) 
    cv2.imshow("Filtered Components", output) 
    # cv2.waitKey(0)

    return image0, center_ret, top_left_ret, bottom_right_ret


def get_coords(centers):
    align_to = rs.stream.color
    align = rs.align(align_to)

    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    # cv2.imshow("depth", depth_frame)
    if not depth_frame or not color_frame:
        return
    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()

    # spatial = rs.spatial_filter()
    # temporal = rs.temporal_filter()
    # hole_filling = rs.hole_filling_filter(1) 
    # depth_to_disparity = rs.disparity_transform(True)
    # disparity_to_depth = rs.disparity_transform(False)
    # filtered_frame = depth_to_disparity.process(aligned_depth_frame)
    # filtered_frame = spatial.process(filtered_frame)
    # filtered_frame = temporal.process(filtered_frame)
    # filtered_frame = disparity_to_depth.process(filtered_frame)
    # filtered_frame = hole_filling.process(filtered_frame)
    # filtered_depth_frame = filtered_frame.as_depth_frame()
    # depth_data = (np.asanyarray(filtered_depth_frame.as_frame().get_data()).astype(np.uint16))
    # print(depth_data.min())
    depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics

    result = []
    for center in centers:
        x = center[0]
        y = center[1]
        # depth = filtered_depth_frame.get_distance(min(int(x), 639), min(int(y), 479))
        depth = aligned_depth_frame.get_distance(min(int(x), 639), min(int(y), 479))
        # depth = depth_data[y, x]/1000
        # depth = aligned_depth_frame.get_distance(int(x), int(y))
        # print("Depths: ", depth)
        if depth == 0.0:
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            return None
        # print("x y: ", x, y)
        result.append(rs.rs2_deproject_pixel_to_point(depth_intrin, (x, y), depth))        

    return result


def check_human_in(wrist_msg):
    human_in_msg = Bool()
    human_in_msg.data = False
    # print(wrist_msg.pose.position.x)
    if wrist_msg.pose.position.x < -1:
        # print("SDffffffffffff")
        human_in_msg.data = True
    pub_human_in.publish(human_in_msg)
    

rospy.init_node('pose_subscriber')
rate = rospy.Rate(60)
pub = rospy.Publisher("/stain_position", Float64MultiArray, queue_size=10)
# world2fr3_msg = rospy.wait_for_message("/natnet_ros/fr3/pose", PoseStamped)
world2camtag_msg = rospy.wait_for_message("/natnet_ros/camera_maker/pose", PoseStamped)
pub_human_in = rospy.Publisher("/human_in", Bool, queue_size=10)
wrist_msg = rospy.Subscriber("/natnet_ros/wrist/pose", PoseStamped, check_human_in)

# world2fr3 = pose2tf(world2fr3_msg)
# world2fr3[2, 3] -= 0.015
world2camtag = pose2tf(world2camtag_msg)
# print(world2fr3, '\n', world2camtag)
# exit()
world2fr3 = np.array([[ 9.99865179e-01, -1.44146020e-03,  1.63568172e-02,-2.51966143e+00],
                      [ 1.39646890e-03, 9.99995211e-01,  2.76170315e-03, 5.24470866e-01],
                      [-1.63607198e-02, -2.73848903e-03,  9.99862404e-01,  8.90832946e-01],
                      [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
world2fr3[2, 3] -= 0.015
camtag2cam = np.array([[0.77769256, 0.50455517, -0.37499115, -0.03939482],
                       [0.62845438, -0.60931219, 0.4835119, -0.02450811],
                       [0.01547175, -0.61168844, -0.79094746, -0.0314443],
                       [0, 0, 0, 1]])
# fr32realrobot = np.array([[]])
pipeline, color_image = rs_init()
time.sleep(1)

while not rospy.is_shutdown():

    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:
        print(">>> No depth or color!")
        exit()

    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    # cv2.imwrite('color1.jpg', color_image)
    # break
    color_image, center_ret, top_left_ret, bottom_right_ret = stain_det(color_image)

    # centers = np.array(get_coords(np.array(center_ret)))
    top_left_ret_ = get_coords(np.array(top_left_ret))
    bottom_right_ret_ = get_coords(np.array(bottom_right_ret))
    if top_left_ret_ is None or bottom_right_ret_ is None:
        continue
    corner1 = np.array(top_left_ret_)
    corner2 = np.array(bottom_right_ret_)

    if corner1.shape[0] == 0:
        my_msg = Float64MultiArray()
        pub.publish(my_msg)
        continue


    stain2cam_1 = np.hstack((corner1, np.ones((corner1.shape[0], 1)))).T
    stain2cam_2 = np.hstack((corner2, np.ones((corner2.shape[0], 1)))).T

    world2stain_1 = (world2camtag @ camtag2cam @ stain2cam_1).T
    world2stain_2 = (world2camtag @ camtag2cam @ stain2cam_2).T
    new_corner1 = np.array([world2stain_1[:, 0] - world2fr3[0, 3], 
                            world2stain_1[:, 1] - world2fr3[1, 3], 
                            world2stain_1[:, 2] - world2fr3[2, 3]]).T
    new_corner2 = np.array([world2stain_2[:, 0] - world2fr3[0, 3], 
                            world2stain_2[:, 1] - world2fr3[1, 3], 
                            world2stain_2[:, 2] - world2fr3[2, 3]]).T

    white_board_mask = np.where(
        (new_corner1[:, 0] >= BODY_UP) & (new_corner2[:, 0] <= BODY_BOTTOM) &
        (new_corner1[:, 1] >= BODY_LEFT) & (new_corner2[:, 1] <= BODY_RIGHT) &
        (new_corner1[:, 2] <= BODY_HIGH) & (new_corner2[:, 2] <= BODY_HIGH) &
        (new_corner1[:, 2] >= BODY_LOW) & (new_corner2[:, 2] >= BODY_LOW)
    )
    # new_points = new_points[white_board_mask[0]]
    new_corner1 = new_corner1[white_board_mask[0]]
    new_corner2 = new_corner2[white_board_mask[0]]
    new_stain = np.hstack((new_corner1, new_corner2))
    print("Stains:", new_stain)

    my_msg = Float64MultiArray()
    my_msg.data = new_stain.flatten().tolist()
    pub.publish(my_msg)

    # world2stain_R = np.eye(4)
    # world2stain_R[:3, :3] = world2stain[:3, :3]
    # br.sendTransform(world2stain[:3, 3],
    #                  tf.transformations.quaternion_from_matrix(world2stain_R),
    #                  rospy.Time.now(),
    #                  "stain",
    #                  "world")

    cv2.imshow('Color Image', color_image)
        
    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break

    rate.sleep()

rospy.spin()