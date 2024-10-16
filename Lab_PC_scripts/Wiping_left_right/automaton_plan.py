#!/usr/bin/python3
import rospy, numpy as np
import rospkg
from catkin.find_in_workspaces import find_in_workspaces
from IPython import embed
from franka_msgs.msg import FrankaState
from sensor_msgs.msg import JointState 
from geometry_msgs.msg import Vector3
from std_msgs.msg import Int32, Int32MultiArray, Float64MultiArray
from functools import partial
import pickle
import copy
from visualization_msgs.msg import Marker

import shlex
from psutil import Popen

import time

import jax
from jax import jit
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
from jaxopt import OSQP

jax.config.update("jax_enable_x64", True)

dip_pos_global = jnp.array([0.78, -0.01, 0.06]).reshape((-1,1))

def close_gripper():
	node_process = Popen(shlex.split('rosrun franka_interactive_controllers franka_gripper_run_node 0'))
	# messagebox.showinfo("Close Gripper", "Gripper Closed")
	time.sleep(1) # Sleep for 1 seconds   
	node_process.terminate()

def open_gripper():
	node_process = Popen(shlex.split('rosrun franka_interactive_controllers franka_gripper_run_node 1'))
	# messagebox.showinfo("Close Gripper", "Gripper Closed")
	time.sleep(1) # Sleep for 1 seconds   
	node_process.terminate()

## Automaton Graph

class Node:

    def __init__(self, vertexNumber, is_acc):
        self.vertexNumber = vertexNumber
        self.is_acc = is_acc
        self.out = []

class Edge:
    
    def __init__(self, EdgeNumber, src, dst, cond):
        self.EdgeNumber = EdgeNumber
        self.src = src
        self.dst = dst
        self.cond = cond # cond_list in graph_construct class. See graph_construct class.
        
        cond_control_APs=[]        
        for i in range(len(cond)):
            cond_control = cond[i][0]
            cond_control_APs_i= []
            j = 0
            while(j<len(cond_control)):
                if cond_control[j]=='(':
                    cond_control_APs_i.append(0)
                    j += 12 # indicates number of characters to bypass to get to the next AP
                else:
                    cond_control_APs_i.append(1)
                    j += 4
            cond_control_APs.append(cond_control_APs_i)
        self.cond_control_APs = cond_control_APs # list = [control_AP_1, control_AP_2, ..] where, for e.g.,
        # control_AP_1 = [0, 0, 1, 0], control_AP_2 = [0, 0, 0, 1] and 1 indicates the active control_prop

class Graph:
    
    def __init__(self, nodes, edges, APs, n_cont_AP, n_uncont_AP):
        self.nodes = nodes
        self.edges = edges
        self.APs = APs
        self.n_cont_AP = n_cont_AP
        self.n_uncont_AP = n_uncont_AP
    
    
    def trans(self, node_index, label):
        control_APs = label[0]
        uncontrol_APs = label[1]
        ## symbols for APs -> eval() method depends on this
        a = control_APs[0]
        b = control_APs[1]
        c = control_APs[2]        
        d = control_APs[3]
        e = control_APs[4]
        
        w = uncontrol_APs[0]
        x = uncontrol_APs[1]
        y = uncontrol_APs[2]
        z = uncontrol_APs[3]
        
        for i in self.nodes[node_index].out:
            for j in i.cond:
                if eval(j[0]) & eval(j[1]):
                    return i.dst
        return 'invalid'
    
    def get_edge(self, start_node, dest_node):
        for i in start_node.out:
            if i.dst == dest_node.vertexNumber:
                return i
        return None

def mod_graph(aut_graph, uncontrol_AP):
    ## symbols for uncontrollable propositions
    w = uncontrol_AP[0]
    x = uncontrol_AP[1]
    y = uncontrol_AP[2]
    z = uncontrol_AP[3]
    
    mod_aut_graph = copy.deepcopy(aut_graph)
    remove_count = 0
    for i in range(len(aut_graph.edges)):
        j = 0
        remove_flag = 1
        # Number of subformula combined with 'or' in the label of the transition
        while j<(len(aut_graph.edges[i].cond)):
            if eval(aut_graph.edges[i].cond[j][1]):
                j = len(aut_graph.edges[i].cond)
                remove_flag = 0
            else:
                j += 1
        if remove_flag:
            remove_ind = i - remove_count # index in the list of edges
            edge_to_be_removed = mod_aut_graph.edges[remove_ind]
            source_node_index = edge_to_be_removed.src
            mod_aut_graph.nodes[source_node_index].out.remove(edge_to_be_removed) # remove edge from source node
            mod_aut_graph.edges.remove(edge_to_be_removed) # remove edge from graph
            remove_count += 1
    return mod_aut_graph

def shortest_path(aut_graph, start_node, goal_nodes):
    explored = []

    # Queue for traversing the
    # graph in the BFS
    queue = [[start_node]]

    # If the desired node is
    # reached
    if start_node.is_acc & (start_node.vertexNumber in [i.dst for i in start_node.out]):
        path = [start_node, start_node]
        path_edges = [aut_graph.get_edge(start_node, start_node)]
        return path, path_edges

    # Loop to traverse the graph
    # with the help of the queue
    while queue:
        path = queue.pop(0)
        node = path[-1]

        # Condition to check if the
        # current node is not visited
        if node not in explored:
            neighbours = [aut_graph.nodes[i.dst] for i in node.out]

            # Loop to iterate over the
            # neighbours of the node
            for neighbour in neighbours:
                new_path = list(path)
                new_path.append(neighbour)
                queue.append(new_path)

                # Condition to check if the
                # neighbour node is the goal
                if neighbour in goal_nodes:
                    path_edges = []
                    # Reading out the edges of the path
                    for i in range(len(new_path) - 1):
                        path_edges.append(aut_graph.get_edge(new_path[i], new_path[i + 1]))
                    return new_path, path_edges
            explored.append(node)

    # Condition when the nodes
    # are not connected
    return None

def get_curr_control_prop(curr_edge, uncontrol_AP):
    w = uncontrol_AP[0]
    x = uncontrol_AP[1]
    y = uncontrol_AP[2]
    z = uncontrol_AP[3]

    for i in range(len(curr_edge.cond)):
        if eval(curr_edge.cond[i][1]):
            return int(np.nonzero(curr_edge.cond_control_APs[i])[0]) # index of the active control_prop [a, b, c, ...] -> [0, 1, 2, ...]
    
    return None

def generate_target_array_jit(dirt_pos, dirt_size): # dirt_pos is R^2, dirt_size is also R^2
    
    d = 0.04

    spacing_y = 0.01
    dirt_size_x = jnp.maximum(dirt_size[0], 0.1)
    dirt_size_y = jnp.maximum(dirt_size[1], 0.15)
    dirt_size = jnp.array([dirt_size_x, dirt_size_y])
    nsamples_y = int(dirt_size[1] / spacing_y)
    spacing_x = 0.01
    nsamples_x = int(d / spacing_x)
    N = int(jnp.ceil(dirt_size[0] / (2 * d)))
    indices = jnp.arange(N).reshape((-1, 1))

    traj_y_atom = jnp.linspace(dirt_pos[1], dirt_pos[1] + dirt_size[1], num=nsamples_y).reshape((-1, 1))
    traj_x_atom = jnp.linspace(dirt_pos[0], dirt_pos[0] + d, num=nsamples_x).reshape((-1, 1))

    z = jnp.array([[0.005]])
    x_start = dirt_pos[0].reshape((-1, 1))
    y_start = dirt_pos[1].reshape((-1, 1))
    y_end = traj_y_atom[-1].reshape((-1, 1))

    target_init_x = jnp.vstack((jnp.repeat(dirt_pos[0], nsamples_y, axis=0).reshape((-1, 1)), traj_x_atom,
                                jnp.repeat(traj_x_atom[-1], nsamples_y, axis=0).reshape((-1, 1)),
                                traj_x_atom + d))
    target_init_y = jnp.vstack((traj_y_atom,  jnp.repeat(traj_y_atom[-1], nsamples_x, axis=0).reshape((-1, 1)),
                                jnp.flip(traj_y_atom), jnp.repeat(traj_y_atom[0], nsamples_x, axis=0).reshape((-1, 1))))

    # rospy.loginfo(N)

    target_x = jnp.ravel(jnp.tile(target_init_x, reps=N), order='F').reshape((-1,1)) + 2*d*jnp.repeat(indices, repeats=target_init_x.shape[0]).reshape((-1,1))
    target_y = jnp.ravel(jnp.tile(target_init_y, reps=N), order='F').reshape((-1,1))

    ts = jnp.linspace(0, 1, num=100)
    ts_orig = jnp.linspace(0, 1, num=target_x.shape[0])
    target_x_final = jnp.interp(ts, ts_orig, target_x[:, 0]).reshape((-1,1))
    target_y_final = jnp.interp(ts, ts_orig, target_y[:, 0]).reshape((-1,1))

    target_array = jnp.hstack((target_x_final, target_y_final, jnp.repeat(z, target_x_final.shape[0], axis=0)))
    vel_array = jnp.gradient(target_array, axis=0)

    return target_array, vel_array

class automaton_decision:

    def __init__(self):

        global dip_pos_global

        # Publishers

        self.pub_dirt_pos = rospy.Publisher('/my_aut_plan/dirt_pos', Vector3, queue_size=10)

        self.pub_dip_pos = rospy.Publisher('/my_aut_plan/dip_pos', Vector3, queue_size=10)

        self.pub_control_t = rospy.Publisher('/my_aut_plan/control_t', Int32, queue_size=10)

        ## Subscribers

        rospy.Subscriber('/franka_state_controller/franka_states', FrankaState, 
                                          self.franka_callback, queue_size=1, tcp_nodelay=True)
        
        rospy.Subscriber("/stain_position", Float64MultiArray, self.stain_callback, queue_size=10)

        rospy.Subscriber("/joint_states", JointState, self.gripper_callback, queue_size=10)

        self.xt = jnp.zeros((3,1)) # franka cartesian state

        self.control_t = 0

        self.dirt_pos = jnp.zeros((3,1))

        self.dirt_pos_end = jnp.zeros((3,1))

        self.dirt_size = jnp.zeros((3,1))

        self.dip_pos =  dip_pos_global # eraser position

        self.stain_pos = 100*np.ones(3)
        self.h_w = 100*np.ones(3)
        self.gripper = 0.

    def franka_callback(self, state_msg):
        O_T_EE = jnp.array(state_msg.O_T_EE).reshape(4, 4).T
        # self.xt = self.xt.at[0, 1].set(O_T_EE[0, 3])
        # self.xt = self.xt.at[1, 1].set(O_T_EE[1, 3])
        # self.xt = self.xt.at[2, 1].set(O_T_EE[2, 3])
        self.xt = jnp.array(O_T_EE[:3,3]).reshape((-1,1))

    def gripper_callback(self, joint_msg):
        self.gripper = joint_msg.position[-1] # right gripper?

    def stain_callback(self, pos_msg):
        # print(pos_msg)
        pos_msg = np.array(pos_msg.data)
        if pos_msg.shape[0] == 0:
            self.stain_pos = 100*np.ones(3)
            self.h_w = 100*np.ones(3)
        else:
            self.stain_pos = np.array(pos_msg[:3])
            self.stain_pos[2] = 0.01
            self.h_w = np.array(pos_msg[3:6] - pos_msg[:3])
            self.h_w[2] = 0
        rospy.loginfo("pos_msg: %s", self.stain_pos)

    def pub_dirt_pos_fn(self, dirt_pos_x, dirt_pos_y, dirt_pos_z):
        dirt_pos_vector = Vector3()
        
        dirt_pos_vector.x = dirt_pos_x
        dirt_pos_vector.y = dirt_pos_y
        dirt_pos_vector.z = dirt_pos_z

        self.pub_dirt_pos.publish(dirt_pos_vector)

    def pub_dip_pos_fn(self, dip_pos_x, dip_pos_y, dip_pos_z):
        dip_pos_vector = Vector3()
        
        dip_pos_vector.x = dip_pos_x
        dip_pos_vector.y = dip_pos_y
        dip_pos_vector.z = dip_pos_z

        self.pub_dip_pos.publish(dip_pos_vector)


buttons_data = jnp.zeros((2,1)) # number of DS choices

def buttons_callback(buttons_msg):
    global buttons_data
    buttons_data = jnp.array(buttons_msg.data).reshape((-1,1)) # DS1, DS2

## Subscriber for buttons

rospy.Subscriber('/buttons', Int32MultiArray, 
                                buttons_callback, queue_size=1, tcp_nodelay=True)

def main():
    global  buttons_data, dip_pos_global

    rospack = rospkg.RosPack()
    curr_path = rospack.get_path('franka_interactive_controllers')

    ## Load automaton graph

    aut_file = curr_path + '/config/Automaton_graph/aut_graph_two_DS.dictionary'
    with open(aut_file, 'rb') as f:
        a1_graph = pickle.load(f)
    
    ## Initialize node
    
    freq = 100
    rospy.init_node('automaton_plan', anonymous=True)
    rate = rospy.Rate(freq) # Hz

    ## Initialization of automaton planning

    # modulate graph

    uncontrol_AP = [1, 0, 1, 0] # uncontrollable propositions: w (DS1: right), x (DS2: left), y (gripper yes), z (stain) 
    uncontrol_AP_prevs = uncontrol_AP.copy()
    mod_aut_graph = mod_graph(a1_graph, uncontrol_AP)

    # Should get the goal nodes again for the new modified graph
    goal_nodes = [i for i in mod_aut_graph.nodes if i.is_acc]

    # Controllable propositions

    control_AP = [1, 0, 0, 0, 0] # a (DS1: right), b (DS2: left), c (go to dirt), d (wipe dirt), e (go to gripper)
    control_prop_t = int(np.nonzero(control_AP)[0])

    # Initial node and transition

    curr_node_index = 0
    curr_node = mod_aut_graph.nodes[curr_node_index]
    label = [control_AP, uncontrol_AP]
    curr_node_index = mod_aut_graph.trans(curr_node.vertexNumber, label) # curr_node.vertexNumber = curr_node_index
    curr_node = mod_aut_graph.nodes[curr_node_index]

    # Initial prefix and suffix path

    path, path_edges = shortest_path(mod_aut_graph, curr_node, goal_nodes)
    prefix = {'path_states': path, 'path_edges': path_edges}
    path_s, path_edges_s = shortest_path(mod_aut_graph, path[-1], path[-1])
    suffix = {'path_states': path_s, 'path_edges': path_edges_s}

    # Planner parameters    
    epsilon_B = 0.02  # STL -> CBF
    epsilon_margin = 0.05

    # Initialization of time varying planning parameters

    prefix_c = 0
    suffix_c = 0
    curr_edge = path_edges[prefix_c]
    
    ## Initialize object

    aut_dec_obj = automaton_decision()
    time.sleep(1)        
    stain_start_time = time.time()
    stain_finish_time = time.time()
    stain_count = 0
    gripper_threshold = 0.03
    first_dip = 0
    gripper_open = 0
    
    while not rospy.is_shutdown():
        now = time.time()

        # print("Buttons: ", buttons_data)

        ## Environmental observations

        active_DS = [idx for idx, val in enumerate(uncontrol_AP) if val != 0][0]
        uncontrol_AP_prevs = uncontrol_AP.copy()

        if int(aut_dec_obj.stain_pos[0]) != 100:
            stain_flag = 1
        else:
            stain_flag = 0

        # logic here: Stain has been **stable** for n seconds
        if not stain_flag: # no stain yet
            stain_start_time = time.time()
            stain_pos_history = []
            stain_hw_history = []
            stain_count = 0
        else:
            stain_finish_time = time.time()
            stain_count += 1

        gripper_open = aut_dec_obj.gripper > gripper_threshold

        # 1. Gripper dropped 
        if gripper_open:
            # time.sleep(1)
            uncontrol_AP[2] = 0
            control_prop_t = 4 # manually changing it
        
        # 2. Stain/dirt detected
        if stain_count == 200 and control_prop_t<=1:
            print(">>>>>>>>>>>>>> Stain detected")
            uncontrol_AP[3] = 1
            enlarge_factor_x = 0.015
            enlarge_factor_y = 0.02
            dirt_pos = aut_dec_obj.stain_pos.reshape((3, 1))
            h_w = aut_dec_obj.h_w.reshape((3, 1)) 
            print(">>>", dirt_pos.shape, h_w.shape)
            target_array, vel_array = generate_target_array_jit(dirt_pos[:2, 0] + jnp.array([-enlarge_factor_x, -enlarge_factor_y]), 
            h_w[:2, 0] + jnp.array([2*enlarge_factor_x, 2*enlarge_factor_y]))
            hand_dirt_file = curr_path + '/config/Trajectory_data_eff/trajectory_hand_dirt.npy'
            with open(hand_dirt_file, 'wb') as f:
                jnp.save(f, target_array)
                jnp.save(f, vel_array)
            aut_dec_obj.dirt_pos_end = target_array[-1].reshape((-1,1))
            decrease_z = 0.015 # for real robot, 0 for gazebo (too small a velocity near dirt_pos in real robot)
            enlarge_factor_pos = 0.02
            aut_dec_obj.dirt_pos = jnp.copy(dirt_pos) + jnp.array([-enlarge_factor_pos, -enlarge_factor_pos, decrease_z]).reshape((3,1))
            aut_dec_obj.dirt_size = jnp.copy(h_w)
            aut_dec_obj.pub_dirt_pos_fn(aut_dec_obj.dirt_pos[0], aut_dec_obj.dirt_pos[1], 
                                        aut_dec_obj.dirt_pos[2])
            aut_dec_obj.pub_dirt_pos_fn(aut_dec_obj.dirt_pos[0], aut_dec_obj.dirt_pos[1], 
                                        aut_dec_obj.dirt_pos[2])
            
        # 3. Choose DS1: right
        if buttons_data[0] == 1:
            uncontrol_AP[0] = 1
            uncontrol_AP[1] = 0
        
        # 4. Choose DS2: left
        if buttons_data[1] == 1:
            uncontrol_AP[1] = 1
            uncontrol_AP[0] = 0

        # if robot reached near gripper position       
        if control_prop_t == 4:
            if jnp.linalg.norm(aut_dec_obj.xt - aut_dec_obj.dip_pos)<(epsilon_B + 0.06):
                if first_dip == 0:
                    first_dip = 1
                    aut_dec_obj.dip_pos = jnp.copy(dip_pos_global) + jnp.array([0.0, 0.0, -0.04]).reshape((-1,1))
                else:
                    uncontrol_AP[2] = 1
                    control_AP = [0] * len(control_AP)
                    control_AP[active_DS] = 1
                    control_prop_t = active_DS # something other than 4 (DS1)
                    curr_node_index = 0
                    time.sleep(1) 
                    close_gripper() # close gripper
                    time.sleep(1)                  
                    first_dip = 0

            rospy.loginfo("Closeness to goal: %f", jnp.linalg.norm(aut_dec_obj.xt - aut_dec_obj.dip_pos))

        # if robot reached near stain/dirt position  
        elif control_prop_t == 2:  # go to stain/dirt position
            if jnp.linalg.norm(aut_dec_obj.xt - aut_dec_obj.dirt_pos)<(epsilon_B + 0.05):
                control_AP = [0] * len(control_AP)
                control_AP[2] = 1
            # rospy.loginfo("Closeness to goal: %f", jnp.linalg.norm(xt - aut_dec_obj.dirt_pos))

        # if robot finished wiping dirt
        elif control_prop_t == 3:  # wiping stain/dirt
            control_AP = [0] * len(control_AP)
            control_AP[3] = 1
            if jnp.linalg.norm(aut_dec_obj.xt - aut_dec_obj.dirt_pos_end)<(epsilon_B + epsilon_margin):
                uncontrol_AP[3] = 0 # stain has gone as observed by the environment.
        
        # doing DS models
        else:
            control_AP = [0] * len(control_AP)
            control_AP[int(control_prop_t)] = 1        


        label = [control_AP, uncontrol_AP]
        prevs_node_index = jnp.copy(curr_node_index)

        # print("Label:" + str(label))
        # print("Curr_node: ", curr_node.vertexNumber)
        
        if uncontrol_AP_prevs == uncontrol_AP and (not control_prop_t == 4):
            curr_node_index = mod_aut_graph.trans(curr_node.vertexNumber, label)
            curr_node = mod_aut_graph.nodes[curr_node_index]

            if not curr_node_index == prevs_node_index:
                prefix_c += 1
                if prefix_c < len(prefix['path_edges']):
                    curr_edge = path_edges[prefix_c]
                else:
                    if suffix_c < len(suffix['path_edges']):
                        curr_edge = path_edges_s[suffix_c]
                        suffix_c += 1
                    else:
                        suffix_c = 0
                        curr_edge = path_edges_s[suffix_c]
        elif not control_prop_t==4:
            mod_aut_graph = mod_graph(a1_graph, uncontrol_AP)
            curr_node_index = mod_aut_graph.trans(curr_node.vertexNumber, label)
            # rospy.loginfo("Next node index" + str(next_node_index))
            curr_node = mod_aut_graph.nodes[curr_node_index]
            goal_nodes = [i for i in mod_aut_graph.nodes if i.is_acc]
            path, path_edges = shortest_path(mod_aut_graph, curr_node, goal_nodes)
            prefix = {'path_states': path, 'path_edges': path_edges}
            path_s, path_edges_s = shortest_path(mod_aut_graph, path[-1], path[-1])
            suffix = {'path_states': path_s, 'path_edges': path_edges_s}
            curr_edge = path_edges[0]
            prefix_c = 0
            suffix_c = 0

        if not control_prop_t == 4:
            control_prop_t = get_curr_control_prop(curr_edge, uncontrol_AP)

        # Publish

        if control_prop_t == 2 or control_prop_t == 3:
            aut_dec_obj.pub_dirt_pos_fn(aut_dec_obj.dirt_pos[0], aut_dec_obj.dirt_pos[1], 
                                        aut_dec_obj.dirt_pos[2])
            
        elif control_prop_t == 4:
            aut_dec_obj.pub_dip_pos_fn(aut_dec_obj.dip_pos[0], aut_dec_obj.dip_pos[1], 
                                        aut_dec_obj.dip_pos[2])

        aut_dec_obj.pub_control_t.publish(int(control_prop_t)) # publishing type of controller

        # rospy.loginfo("Controller: %f", control_prop_t)

        rate.sleep()


if __name__ == '__main__':
    try:
        main()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass