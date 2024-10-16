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
import torch

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


def pot_det(image0):

    gray_image = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)
    # depth_image = depth_image * 0.001
    results = model(image0)

    pot_in_msg = Bool()
    pot_in_msg.data = False

    for result in results.xyxy[0]:
        x1, y1, x2, y2, confidence, class_id = result
        if model.names[int(class_id)] in ['bowl', 'cup']:
            pot_in_msg.data = True
            object_depth = np.median(depth_image[int(y1):int(y2), int(x1):int(x2)])
            label = f"{object_depth:.2f}m"

            cv2.rectangle(image0, (int(x1), int(y1)), (int(x2), int(y2)), (252, 119, 30), 2)
            cv2.putText(image0, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (252, 119, 30), 2)
            print(f"{model.names[int(class_id)]}: {object_depth:.2f}m")
    cv2.imshow("Color Image", image0)
    pub_pot_in.publish(pot_in_msg)


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
            print("Zero depth")
            return None
        # print("x y: ", x, y)
        result.append(rs.rs2_deproject_pixel_to_point(depth_intrin, (x, y), depth))        

    return result


def check_human_in(wrist_msg):
    human_in_msg = Bool()
    human_in_msg.data = False
    # print(wrist_msg.pose.position.x)
    if wrist_msg.pose.position.x < -1.6:
        # print("SDffffffffffff")
        human_in_msg.data = True
    pub_human_in.publish(human_in_msg)
    

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.08  # You can adjust this threshold value as needed

rospy.init_node('pose_subscriber')
rate = rospy.Rate(60)
pub = rospy.Publisher("/stain_position", Float64MultiArray, queue_size=10)
# world2fr3_msg = rospy.wait_for_message("/natnet_ros/fr3/pose", PoseStamped)
world2camtag_msg = rospy.wait_for_message("/natnet_ros/camera_maker/pose", PoseStamped)
pub_human_in = rospy.Publisher("/human_in", Bool, queue_size=10)
pub_pot_in = rospy.Publisher("/pot_in", Bool, queue_size=10)
wrist_msg = rospy.Subscriber("/natnet_ros/left_wrist/pose", PoseStamped, check_human_in)

# world2fr3 = pose2tf(world2fr3_msg)
# world2fr3[2, 3] -= 0.015
world2camtag = pose2tf(world2camtag_msg)
# print(world2camtag)
# exit()
# world2camtag = np.array([[    0.30451 ,   -0.94738 ,  -0.098741   ,  -1.5661],
#  [    0.94021 ,    0.31556   , -0.12817   ,  0.96857],
#  [    0.15258 ,  -0.053809  ,   0.98682     , 1.6491],
#  [          0    ,       0       ,    0      ,     1]])

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
    pot_det(color_image)

    # cv2.imshow('Color Image', color_image)
        
    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break

    rate.sleep()

rospy.spin()