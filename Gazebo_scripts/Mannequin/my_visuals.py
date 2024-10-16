#!/usr/bin/python3
import rospy, numpy as np
import rospkg

from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

from geometry_msgs.msg import Vector3
from std_msgs.msg import Int32

import time

import jax.numpy as jnp

# ROS node reference Tianyu: https://github.com/tonylitianyu/golfbot/blob/master/nodes/moving#L63

class my_visualisation:
    def __init__(self):
        self.freq = 200
        self.grip_pos = jnp.zeros((3,1))
        self.dirt_pos = jnp.zeros((3,1))
        self.control_t = 0

        ## Subscribe to dirt and grip position, control_t

        rospy.Subscriber('/my_aut_plan/grip_pos', Vector3, self.grip_pos_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('/my_aut_plan/dirt_pos', Vector3, self.dirt_pos_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('/my_aut_plan/control_t', Int32, self.control_t_callback, queue_size=1, tcp_nodelay=True)

        ## Publish reference path
            
        self.publish_ref_path = rospy.Publisher('/ref_array', MarkerArray, queue_size=1)
        self.publish_data_path = rospy.Publisher('/data_array', MarkerArray, queue_size=1)
        self.publish_wiping_path = rospy.Publisher('/wiping_array', MarkerArray, queue_size=1)
        self.marker_pub = rospy.Publisher("/marker", Marker, queue_size = 2)

    def grip_pos_callback(self, data):
        self.grip_pos = jnp.array([data.x, data.y, data.z]).reshape((-1,1))

    def dirt_pos_callback(self, data):
        self.dirt_pos = jnp.array([data.x, data.y, data.z]).reshape((-1,1))

    def control_t_callback(self, data):
        self.control_t = int(data.data)


def create_marker_arr(traj, color_rgb):
    markerArray = MarkerArray()
    for i in range(traj.shape[0]):

        marker_i = Marker()
        marker_i.header.frame_id = "panda_link0"
        marker_i.id = i
        marker_i.type = marker_i.SPHERE
        marker_i.action = marker_i.ADD
        marker_i.scale.x = 0.01
        marker_i.scale.y = 0.01
        marker_i.scale.z = 0.01
        marker_i.color.a = 1
        marker_i.color.r = color_rgb[0]
        marker_i.color.g = color_rgb[1]
        marker_i.color.b = color_rgb[2]
        marker_i.pose.orientation.w = 1.0
        marker_i.pose.position.x = traj[i, 0]
        marker_i.pose.position.y = traj[i, 1]
        marker_i.pose.position.z = traj[i, 2]

        markerArray.markers.append(marker_i)
    
    return markerArray

def main():

    ## Initialize object

    visualize_obj = my_visualisation()

    rospack = rospkg.RosPack()
    curr_path = rospack.get_path('franka_interactive_controllers')

    ## Loading learned DS motion plan + trajectory

    data_files_orig = ['trajectory_data_legs_orig.npy',
                       'trajectory_data_hands_orig.npy',
                       'trajectory_data_back_orig.npy']

    n_DS = len(data_files_orig)
    
    curr_data_path = curr_path + '/config/Trajectory_data_eff/'
    traj_load_all = []
    xref_all = []

    train_indx = 0
    split_indx = 0

    for i in range(n_DS):

        data_file_orig = curr_data_path + data_files_orig[i]
        with open(data_file_orig, 'rb') as f:
            traj_all_combine_process = jnp.load(f) # nD x ntrajs (split) x nsamples x 2*dim
            vel_stavel_all_combine_process = jnp.load(f)

        traj_load = traj_all_combine_process[train_indx, split_indx, :, :3]
        traj_load_all.append(traj_load) # ignore quaternions

        ref_file = curr_path + '/config/Trajectory_data_eff/' + 'trajectory_ref_' + str(i) + '.npy'
        with open(ref_file, 'rb') as f:
            xref = jnp.load(f)

        xref_all.append(xref)
    
    ## Initialize node

    rospy.init_node('my_visuals', anonymous=True)
    rate = rospy.Rate(visualize_obj.freq) # Hz

    ## Marker dirt/grip

    marker = Marker()

    marker.header.frame_id = "panda_link0"
    marker.header.stamp = rospy.Time.now()

    # set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
    marker.type = 2
    marker.id = 0

    # Set the scale of the marker
    marker.scale.x = 0.05
    marker.scale.y = 0.05
    marker.scale.z = 0.05

    # Set the color
    marker.color.r = 1.0
    marker.color.g = 0.0
    marker.color.b = 0.0
    marker.color.a = 1.0

    # Set the pose of the marker
    marker.pose.position.x = 0
    marker.pose.position.y = 0
    marker.pose.position.z = 0
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0

    i = 0

    markerArray_data = []
    markerArray_ref = []

    for i in range(n_DS):
        traj_load = traj_load_all[i]
        traj_ref = xref_all[i]
        markerArray_data.append(create_marker_arr(traj_load, [0.0, 0.0, 1.0]))
        markerArray_ref.append(create_marker_arr(traj_ref, [1.0, 0.0, 1.0]))

    while not rospy.is_shutdown():


        if visualize_obj.control_t == 4:
            marker.pose.position.x = visualize_obj.grip_pos[0]
            marker.pose.position.y = visualize_obj.grip_pos[1]
            marker.pose.position.z = visualize_obj.grip_pos[2]

            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0

            marker.color.a=1.0
            visualize_obj.marker_pub.publish(marker)

        elif visualize_obj.control_t == 0:
            marker.color.a = 0.0
            visualize_obj.publish_data_path.publish(markerArray_data[0])
            visualize_obj.publish_ref_path.publish(markerArray_ref[0])

        elif visualize_obj.control_t == 1:
            marker.color.a = 0.0
            visualize_obj.publish_data_path.publish(markerArray_data[1])
            visualize_obj.publish_ref_path.publish(markerArray_ref[1])

        elif visualize_obj.control_t == 2:
            marker.color.a = 0.0
            visualize_obj.publish_data_path.publish(markerArray_data[2])
            visualize_obj.publish_ref_path.publish(markerArray_ref[2])



        elif visualize_obj.control_t == 3:
            marker.pose.position.x = visualize_obj.dirt_pos[0]
            marker.pose.position.y = visualize_obj.dirt_pos[1]
            marker.pose.position.z = visualize_obj.dirt_pos[2]

            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0

            marker.color.a = 1.0

            visualize_obj.marker_pub.publish(marker)
            visualize_obj.publish_ref_path.publish(markerArray_ref[2])
        rate.sleep()


if __name__ == '__main__':
    try:
        main()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass