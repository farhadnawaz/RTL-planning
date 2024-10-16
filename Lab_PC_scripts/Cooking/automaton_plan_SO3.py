#!/usr/bin/python3
import rospy, numpy as np
import rospkg
from catkin.find_in_workspaces import find_in_workspaces
from IPython import embed
from franka_msgs.msg import FrankaState
from geometry_msgs.msg import Vector3
from std_msgs.msg import Int32, Int32MultiArray, Bool
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

        x = uncontrol_APs[0]
        y = uncontrol_APs[1]
        z = uncontrol_APs[2]
        
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
    x = uncontrol_AP[0]
    y = uncontrol_AP[1]
    z = uncontrol_AP[2]
    
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
    x = uncontrol_AP[0]
    y = uncontrol_AP[1]
    z = uncontrol_AP[2]

    for i in range(len(curr_edge.cond)):
        if eval(curr_edge.cond[i][1]):
            return int(np.nonzero(curr_edge.cond_control_APs[i])[0]) # index of the active control_prop [a, b, c, ...] -> [0, 1, 2, ...]
    
    return None

class automaton_decision:

    def __init__(self):

        self.control_t = 3

xt = jnp.zeros((3,1))
qt = jnp.array([1.0, 0.0, 0.0, 0.0]).reshape((-1,1))
buttons_data = jnp.zeros((2,1)) # place/remove pan 2
human_data = 0
pot_data = 1

def franka_callback(state_msg):
    global xt
    O_T_EE = jnp.array(state_msg.O_T_EE).reshape(4, 4).T
    xt = jnp.array(O_T_EE[:3,3]).reshape((-1,1))

# def buttons_callback(buttons_msg):
#     global buttons_data
#     buttons_data = jnp.array(buttons_msg.data).reshape((-1,1)) # place/remove pan 2

def human_callback(msg):
    global human_data
    human_data = msg.data

def pot_callback(msg):
    global pot_data
    pot_data = msg.data

## Subscribers

rospy.Subscriber('/franka_state_controller/franka_states', FrankaState, 
                                franka_callback, queue_size=1, tcp_nodelay=True)
# rospy.Subscriber('/buttons', Int32MultiArray, buttons_callback, queue_size=1, tcp_nodelay=True)
rospy.Subscriber('/human_in', Bool, human_callback, queue_size=1, tcp_nodelay=True)
rospy.Subscriber('/pot_in', Bool, pot_callback, queue_size=1, tcp_nodelay=True)

## Publishers

pub_control_t = rospy.Publisher('/my_aut_plan/control_t', Int32, queue_size=10)

def main():
    global xt, buttons_data, pot_data, human_data

    rospack = rospkg.RosPack()
    curr_path = rospack.get_path('franka_interactive_controllers')
    
    ## Initialize node
    
    freq = 500
    rospy.init_node('automaton_plan_SO3', anonymous=True)
    rate = rospy.Rate(freq) # Hz

    ## Load scooping trajectory to get end and starting position

    ref_file_scoop = curr_path + '/config/Trajectory_data_eff/trajectory_ref_0.npy'
    with open(ref_file_scoop, 'rb') as f:
        xref_scoop = jnp.load(f)
        xref_vel_scoop = jnp.load(f)

    scoop_start = xref_scoop[0]
    scoop_end = xref_scoop[-1]

    ## Load transit_drop trajectory to get end position

    ref_file_drop = curr_path + '/config/Trajectory_data_eff/trajectory_ref_1.npy'
    with open(ref_file_drop, 'rb') as f:
        xref_drop = jnp.load(f)
        xref_vel_drop = jnp.load(f)

    drop_end = xref_drop[-1]

    ## Load automaton graph

    aut_file = curr_path + '/config/Automaton_graph/aut_graph_cooking.dictionary'
    with open(aut_file, 'rb') as f:
        a1_graph = pickle.load(f)

    ## Initialization of automaton planning

    # modulate graph

    uncontrol_AP = [1, 0, 0] # uncontrollable propositions: x (Pan 2), y (beans), z (dropped) 
    uncontrol_AP_prevs = uncontrol_AP.copy()
    mod_aut_graph = mod_graph(a1_graph, uncontrol_AP)

    # Should get the goal nodes again for the new modified graph
    goal_nodes = [i for i in mod_aut_graph.nodes if i.is_acc]

    # Controllable propositions

    control_AP = [1, 0, 0, 0, 0] # a (scoop), b (transit_drop), c (stir), d (go to scoop), e (go up (from stir))
    control_prop_t = 3

    # Initial node and transition

    curr_node_index = 0
    curr_node = mod_aut_graph.nodes[curr_node_index]
    label = [control_AP, uncontrol_AP]
    # curr_node_index = mod_aut_graph.trans(curr_node.vertexNumber, label) # curr_node.vertexNumber = curr_node_index
    # curr_node = mod_aut_graph.nodes[curr_node_index]

    # Initial prefix and suffix path

    path, path_edges = shortest_path(mod_aut_graph, curr_node, goal_nodes)
    prefix = {'path_states': path, 'path_edges': path_edges}
    path_s, path_edges_s = shortest_path(mod_aut_graph, path[-1], path[-1])
    suffix = {'path_states': path_s, 'path_edges': path_edges_s}

    # Planner parameters    
    epsilon_B = 0.05  # STL -> CBF
    epsilon_margin = 0.05

    # Initialization of time varying planning parameters

    i = 0 # count of execution time
    prefix_c = 0
    suffix_c = 0
    curr_edge = path_edges[prefix_c]
    control_prop_t = get_curr_control_prop(curr_edge, uncontrol_AP)
    reached_start = 0
    pot_human_count = 0
    drop_count = 0
    start_scoop_count = 0
    end_scoop_count = 0
    
    ## Initialize object

    aut_dec_obj = automaton_decision()
    
    while not rospy.is_shutdown():
        now = time.time()

        # print("Buttons: ", buttons_data)

        ## Environmental observations

        uncontrol_AP_prevs = uncontrol_AP.copy()

        # 1. Pan 2
        # pan placed
        # if buttons_data[0] == 1 and buttons_data[1] == 0:
        #     uncontrol_AP[0] = 1

        # # pan removed
        # elif buttons_data[0] == 0 and buttons_data[1] == 1:
        #     uncontrol_AP = [0] * len(uncontrol_AP)

        if human_data == 1:
            pot_human_count += 1
        else:
            pot_human_count = 0

        # 1. Pan 2: with camera and human
        # pan removed
        if control_prop_t == 2 and pot_data == 0 and human_data == 1: # stirring motion, no pot detected and human in
            uncontrol_AP = [0] * len(uncontrol_AP)
        elif pot_human_count == 100:
            uncontrol_AP[0] = 1

        # if robot finished scooping (beans in spoon)    
        if control_prop_t == 0: 
            control_AP = [0] * len(control_AP)
            control_AP[0] = 1
            if jnp.linalg.norm(xt[:, 0] - scoop_end[:3])<(epsilon_B):
                end_scoop_count += 1
                if end_scoop_count == 150:                    
                    uncontrol_AP[1] = 1
                    end_scoop_count = 0
        
        # if robot reached scooping start position
        elif control_prop_t == 3:
            reached_start = 0
            if jnp.linalg.norm(xt[:, 0] - scoop_start[:3])<(epsilon_B):
                start_scoop_count += 1
                if start_scoop_count == 150:
                    control_AP = [0] * len(control_AP)
                    control_AP[3] = 1
                    reached_start = 1
                    start_scoop_count = 0

            # print("start_scoop_count: ", start_scoop_count)
            # print("To Goal: ", jnp.linalg.norm(xt[:, 0] - scoop_start[:3]))

        # if robot finished dropping  
        elif control_prop_t == 1:
            control_AP = [0] * len(control_AP)
            control_AP[1] = 1
            if jnp.linalg.norm(xt[:, 0] - drop_end[:3])<(epsilon_B):
                drop_count += 1 
                if drop_count == 150:
                    uncontrol_AP[2] = 1
                    drop_count = 0


        label = [control_AP, uncontrol_AP]
        prevs_node_index = jnp.copy(curr_node_index)

        print("Label:" + str(label))
        print("Curr_node: ", curr_node.vertexNumber)

        # if not (control_prop_t == 3 or control_prop_t == 0 or control_prop_t == 1):
        #     control_prop_t = 1

        # else:
        
        if uncontrol_AP_prevs == uncontrol_AP and (reached_start):
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
        elif reached_start:
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

        control_prop_t = get_curr_control_prop(curr_edge, uncontrol_AP)
        control_AP = [0] * len(control_AP)
        control_AP[control_prop_t] = 1        


        # Publish
        pub_control_t.publish(int(control_prop_t)) # publishing type of controller

        # rospy.loginfo("Controller: %f", control_prop_t)

        rate.sleep()


if __name__ == '__main__':
    try:
        main()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass