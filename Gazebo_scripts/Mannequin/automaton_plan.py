#!/usr/bin/python3
import rospy, numpy as np
import rospkg
from catkin.find_in_workspaces import find_in_workspaces
from IPython import embed
from franka_msgs.msg import FrankaState
from geometry_msgs.msg import Vector3
from std_msgs.msg import Int32, Int32MultiArray
from functools import partial
import pickle
import copy

import time

import jax
from jax import jit
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
from jaxopt import OSQP

jax.config.update("jax_enable_x64", True)

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

class automaton_decision:

    def __init__(self):

        self.control_t = 0

        self.dirt_pos = jnp.zeros((3,1))

        self.dirt_pos_end = jnp.zeros((3,1))

        self.gripper_pos = jnp.array([0.5, 0.5, 0.1]).reshape((-1,1))

xt = jnp.zeros((3,1))
buttons_data = jnp.zeros((2,1)) # number of DS choices

def franka_callback(state_msg):
    global xt
    O_T_EE = jnp.array(state_msg.O_T_EE).reshape(4, 4).T
    # self.xt = self.xt.at[0, 1].set(O_T_EE[0, 3])
    # self.xt = self.xt.at[1, 1].set(O_T_EE[1, 3])
    # self.xt = self.xt.at[2, 1].set(O_T_EE[2, 3])
    xt = jnp.array(O_T_EE[:3,3]).reshape((-1,1))

def buttons_callback(buttons_msg):
    global buttons_data
    buttons_data = jnp.array(buttons_msg.data).reshape((-1,1)) # DS1, DS2

## Subscribers

rospy.Subscriber('/franka_state_controller/franka_states', FrankaState, 
                                franka_callback, queue_size=1, tcp_nodelay=True)
rospy.Subscriber('/buttons', Int32MultiArray, 
                                buttons_callback, queue_size=1, tcp_nodelay=True)

## Publishers

pub_dirt_pos = rospy.Publisher('/my_aut_plan/dirt_pos', Vector3, queue_size=10)
pub_grip_pos = rospy.Publisher('/my_aut_plan/grip_pos', Vector3, queue_size=10)
pub_control_t = rospy.Publisher('/my_aut_plan/control_t', Int32, queue_size=10)

def pub_dirt_pos_fn(dirt_pos_x, dirt_pos_y, dirt_pos_z):
    dirt_pos_vector = Vector3()
    
    dirt_pos_vector.x = dirt_pos_x
    dirt_pos_vector.y = dirt_pos_y
    dirt_pos_vector.z = dirt_pos_z

    pub_dirt_pos.publish(dirt_pos_vector)

def pub_grip_pos_fn(dip_pos_x, dip_pos_y, dip_pos_z):
    dip_pos_vector = Vector3()
    
    dip_pos_vector.x = dip_pos_x
    dip_pos_vector.y = dip_pos_y
    dip_pos_vector.z = dip_pos_z

    pub_grip_pos.publish(dip_pos_vector)

def main():
    global xt, buttons_data

    rospack = rospkg.RosPack()
    curr_path = rospack.get_path('franka_interactive_controllers')

    ## Load automaton graph

    aut_file = curr_path + '/config/Automaton_graph/aut_graph_mannequin.dictionary'
    with open(aut_file, 'rb') as f:
        a1_graph = pickle.load(f)
    
    ## Initialize node
    
    freq = 100
    rospy.init_node('automaton_plan', anonymous=True)
    rate = rospy.Rate(freq) # Hz

    ## Initialization of automaton planning

    # modulate graph

    uncontrol_AP = [1, 0, 1, 0] # uncontrollable propositions: w (DS1: legs), x (DS2: hands), y (wet yes), z (stain) 
    uncontrol_AP_prevs = uncontrol_AP.copy()
    mod_aut_graph = mod_graph(a1_graph, uncontrol_AP)

    # Should get the goal nodes again for the new modified graph
    goal_nodes = [i for i in mod_aut_graph.nodes if i.is_acc]

    # Controllable propositions

    control_AP = [1, 0, 0, 0, 0] # a (DS1: legs), b (DS2: hands), c (wipe dirt), d (go to dirt), e (go to wet)
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
    stain_count_init = int(20*freq)
    gripper_count_init = int(30*freq)

    # Initialization of time varying planning parameters

    i = 0 # count of execution time
    prefix_c = 0
    suffix_c = 0
    stain_pos_flag = 1
    grip_flag = 1
    curr_edge = path_edges[prefix_c]
    gripper_count = gripper_count_init
    stain_count = stain_count_init
    
    ## Initialize object

    aut_dec_obj = automaton_decision()
    
    while not rospy.is_shutdown():
        now = time.time()

        # print("Buttons: ", buttons_data)

        ## Environmental observations

        active_DS = [idx for idx, val in enumerate(uncontrol_AP) if val != 0][0]
        uncontrol_AP_prevs = uncontrol_AP.copy()

        # 1. Not wet 
        if gripper_count == 0:
            uncontrol_AP[2] = 0
            control_prop_t = 4 # manually changing it
        
        # 2. Stain/dirt detected
        if stain_count == 0:
            uncontrol_AP[3] = 1
            # dirt_pos = jnp.array([np.random.uniform(0.3, 0.5), np.random.uniform(-0.5, 0.1), 0.035]).reshape((-1,1)) # R^3
            # h_w = jnp.array([np.random.uniform(0.1, 0.2), np.random.uniform(0.2, 0.3), 0]).reshape((-1,1)) # R^3
            # target_array, vel_array = generate_target_array_jit(dirt_pos[:2, 0], h_w[:2, 0])
            hand_dirt_file = curr_path + '/config/Trajectory_data_eff/trajectory_ref_2.npy'
            with open(hand_dirt_file, 'rb') as f:
                target_array = jnp.load(f)
                vel_array = jnp.load(f)
                scaler_t = jnp.load(f)
            dirt_pos = target_array[0].reshape((-1,1))
            aut_dec_obj.dirt_pos_end = target_array[-1].reshape((-1,1))
            pub_dirt_pos_fn(dirt_pos[0], dirt_pos[1], dirt_pos[2])
            aut_dec_obj.dirt_pos = jnp.copy(dirt_pos)
        
        # 3. Choose DS1: legs
        if buttons_data[0] == 1:
            uncontrol_AP[0] = 1
            uncontrol_AP[1] = 0
        
        # 4. Choose DS2: hands
        if buttons_data[1] == 1:
            uncontrol_AP[1] = 1
            uncontrol_AP[0] = 0

        # if robot reached near dipping bowl    
        if control_prop_t == 4: 
            if jnp.linalg.norm(xt - aut_dec_obj.gripper_pos)<(epsilon_B + epsilon_margin):
                uncontrol_AP[2] = 1
                control_AP = [0] * len(control_AP)
                control_AP[active_DS] = 1
                control_prop_t = active_DS # something other than 4 (DS1)
                curr_node_index = 0
                gripper_count = gripper_count_init
                # close()

            # rospy.loginfo("Closeness to goal: %f", jnp.linalg.norm(xt - aut_dec_obj.gripper_pos))

        # if robot reached near stain/dirt position  
        elif control_prop_t == 3:  # go to stain/dirt position
            if jnp.linalg.norm(xt - aut_dec_obj.dirt_pos)<(epsilon_B + 0.5*epsilon_margin):
                control_AP = [0] * len(control_AP)
                control_AP[3] = 1
            # rospy.loginfo("Closeness to goal: %f", jnp.linalg.norm(xt - aut_dec_obj.dirt_pos))

        # if robot finished wiping dirt
        elif control_prop_t == 2:  # wiping stain/dirt
            control_AP = [0] * len(control_AP)
            control_AP[2] = 1
            if jnp.linalg.norm(xt - aut_dec_obj.dirt_pos_end)<(epsilon_B + epsilon_margin):
                uncontrol_AP[3] = 0 # stain has gone as observed by the environment.                
                stain_count = stain_count_init
        
        # doing DS models
        else:
            control_AP = [0] * len(control_AP)
            control_AP[int(control_prop_t)] = 1        


        label = [control_AP, uncontrol_AP]
        prevs_node_index = jnp.copy(curr_node_index)

        print("Label:" + str(label))
        print("Curr_node: ", curr_node.vertexNumber)
        
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
            print("Modifying graph")
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
            pub_dirt_pos_fn(aut_dec_obj.dirt_pos[0], aut_dec_obj.dirt_pos[1], 
                                        aut_dec_obj.dirt_pos[2])
            
        elif control_prop_t == 4:
            pub_grip_pos_fn(aut_dec_obj.gripper_pos[0], aut_dec_obj.gripper_pos[1], 
                                        aut_dec_obj.gripper_pos[2])

        pub_control_t.publish(int(control_prop_t)) # publishing type of controller

        # rospy.loginfo("Controller: %f", control_prop_t)

        gripper_count = gripper_count - 1
        stain_count = stain_count - 1

        rate.sleep()


if __name__ == '__main__':
    try:
        main()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass