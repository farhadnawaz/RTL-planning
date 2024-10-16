#!/usr/bin/python3
import rospy, numpy as np
import rospkg
from catkin.find_in_workspaces import find_in_workspaces
from IPython import embed
from franka_msgs.msg import FrankaState
from geometry_msgs.msg import Twist, Vector3
from std_msgs.msg import Int32
from tf.transformations import quaternion_from_matrix
from nav_msgs.msg import Path
from functools import partial

import time

import jax
from jax import jit
import jax.numpy as jnp
import jax.random as jrandom
from jaxopt import OSQP
from jax.tree_util import register_pytree_node

# jax.config.update("jax_enable_x64", True)

# ROS node reference Tianyu: https://github.com/tonylitianyu/golfbot/blob/master/nodes/moving#L63

## Planner parameters
epsilon_B = 0.02  # STL -> CBF
epsilon_margin = 0.05

class ustar_m:

    def __init__(self, ref_range_start, ref_range, vopt_max, 
                 v_uniform, xt, fxt, control_t, control_t_p, dirt_pos, 
                 trans_period, dt, change_flag):
        self.ref_range_start = ref_range_start # from where to start looking forward?
        self.ref_range = ref_range # look how much forward, horizon N
        self.vopt_max = vopt_max # maximum virtual control input
        self.v_uniform = v_uniform
        self.xt = xt
        self.fxt =  fxt
        self.control_t = control_t
        self.control_t_p = control_t_p
        self.dirt_pos = dirt_pos
        self.trans_period = trans_period
        self.dt = dt
        self.change_flag = change_flag
        # /passiveDS/desired_twist
        # /cartesian_impedance_controller/desired_twist

    @staticmethod
    def q_from_R(R):
        """ generates quaternion from 3x3 rotation matrix """
        _R = jnp.eye(4)
        _R = _R.at[:3, :3].set(R)
        return jnp.array(quaternion_from_matrix(_R))
    
    # @partial(jit, static_argnums=(0,))
    @jit
    def target_track(self, x1, xref_t, fx1, fxref_t, pos,
                     STL_t, t_star, k, scaler_t):
        
        z_orig = xref_t[2, 0]        
        xref_t = xref_t.at[2, 0].set(z_orig - 0.01)
        vopt_bound = 5*self.vopt_max * (1 / scaler_t)
        alpha_V = 70
        dV_de = 2 * (x1 - xref_t)
        V = jnp.sum(jnp.square(dV_de))/4
        alpha_BB_CLF = alpha_V * V

        Q = jnp.eye(fx1.shape[0])
        G = jnp.vstack((dV_de.T, jnp.eye(fx1.shape[0]), -jnp.eye(fx1.shape[0])))
        h = jnp.hstack((jnp.array([-alpha_BB_CLF  - jnp.sum(dV_de * (fx1 - fxref_t))]),
                        jnp.repeat(vopt_bound, repeats=2*fx1.shape[0], axis=0)))

        qp = OSQP()
        sol = qp.run(params_obj=(Q,jnp.zeros(fx1.shape[0])), params_ineq=(G, h)).params
        
        ustar = sol.primal.reshape((-1,1))

        k_scale = 1

        return jnp.float32(k_scale*scaler_t*(ustar + fx1))
    
    # @partial(jit, static_argnums=(0,))
    @jit
    def go_to_dirt(self, x1, xref_t, fx1, fxref_t, dirt_pos,
                   STL_t, t_star, k ,scaler_t):

        vopt_bound = self.vopt_max
        s=1
        alpha_B = 5
        # gamma_t = (k) * (np.exp(-(dip_t / t_star_dip)))
        gamma_t = jnp.maximum((k/(1-jnp.exp(-s*t_star))) * (jnp.exp(-s*STL_t) - jnp.exp(-s*t_star)), 0)
        dB_dx = -2 * (x1 - dirt_pos)
        B = (epsilon_B + gamma_t)**2 - jnp.sum(jnp.square(dB_dx))/4
        # dB_dt = k * (np.exp(-(dirt_t / t_star))) * (-1 / t_star)
        dB_dt = jnp.where(gamma_t > 0, 2*(epsilon_B + gamma_t)*(-k/(1-jnp.exp(-s*t_star))) * jnp.exp(-s*STL_t), 0)
        alpha_BB = alpha_B * B ** 1

        Q = jnp.eye(x1.shape[0])
        G = jnp.vstack((-dB_dx.T, jnp.eye(x1.shape[0]), -jnp.eye(x1.shape[0])))
        h = jnp.hstack((jnp.array([alpha_BB  + dB_dt]),
                        jnp.repeat(vopt_bound, repeats=2*x1.shape[0], axis=0)))

        qp = OSQP()
        sol = qp.run(params_obj=(Q,jnp.zeros(x1.shape[0])), params_ineq=(G, h)).params
        
        ustar = sol.primal.reshape((-1,1))

        return jnp.float32(ustar)
    
    # @partial(jit, static_argnums=(0,))
    @jit
    def go_to_dip(self, x1, xref_t, fx1, fxref_t, dip_pos,
                  STL_t, t_star, k, scaler_t):

        vopt_bound = self.vopt_max
        s=1
        alpha_B = 5
        # gamma_t = (k) * (np.exp(-(dip_t / t_star_dip)))
        gamma_t = jnp.maximum((k/(1-jnp.exp(-s*t_star))) * (jnp.exp(-s*STL_t) - jnp.exp(-s*t_star)), 0)
        dB_dx = -2 * (x1 - dip_pos)
        B = (epsilon_B + gamma_t) ** 2 - jnp.sum(jnp.square(dB_dx))/4
        # dB_dt = k * (np.exp(-(dip_t / t_star_dip))) * (-1 / t_star_dip)
        dB_dt = jnp.where(gamma_t > 0 , 2*(epsilon_B + gamma_t)*(-k/(1-jnp.exp(-s*t_star))) * jnp.exp(-s*STL_t), 0)
        alpha_BB = alpha_B * B ** 1

        Q = jnp.eye(x1.shape[0])
        G = jnp.vstack((-dB_dx.T, jnp.eye(x1.shape[0]), -jnp.eye(x1.shape[0])))
        h = jnp.hstack((jnp.array([alpha_BB  + dB_dt]),
                        jnp.repeat(vopt_bound, repeats=2*x1.shape[0], axis=0)))

        qp = OSQP()

        # rospy.loginfo(x1.shape)

        sol = qp.run(params_obj=(Q,jnp.zeros(x1.shape[0])), params_ineq=(G, h)).params
        
        ustar = sol.primal.reshape((-1,1))

        return jnp.float32(ustar)
    
    # @partial(jit, static_argnums=(0,))
    @jit
    def trans_controller(self, beta_t, vel_cmd, vel_cmd_prevs, u1):
        u1 = jnp.where(beta_t <= 0.001, vel_cmd_prevs, jnp.copy(u1))
        result_vel_cmd = beta_t * vel_cmd + (1 - beta_t) * u1
        return result_vel_cmd, u1
    
    # @partial(jit, static_argnums=(0,))
    @jit
    def no_trans_controller(self, beta_t, vel_cmd, vel_cmd_prevs, u1):
        return vel_cmd, u1

    # @partial(jit, static_argnums=(0,))
    @jit
    def compute_ustar_m(self, count, state_data, NN_vel, control_t, control_t_p, change_flag, trans_period, beta_t, dt_beta_t, k, t_star, 
                        STL_t, vel_cmd_prevs, dirt_pos, dip_pos, u1, xref_DS, xref_vel_DS, scaler_t):

        # quat_ee = self.q_from_R(O_T_EE[:3, :3])

        dim = 3
        
        change_flag = jnp.where(control_t == control_t_p, 0, 1)

        x_t = state_data
        # STL_t = jnp.copy(self.STL_t)
        # t_star = jnp.copy(self.t_star)
        # k = jnp.copy(self.k)

        ## scale down since demos are fast
        # Mixing task := 0.1, wiping big board := 0.5, Spiral big board := 0.5,
        # Wiping big board 2 half := 0.1
        # Spiral 2 := 0.1
        # Cap := 0.1
        # Massage := 1

        k_fx = 1
        fx = k_fx*NN_vel
        
        # xref = jnp.where(control_t == 3, xref_wiping, xref_DS[:, :dim])
        # xref_wiping = xref

        alpha = 0.0 # weightage for cosine distance
        
        dist_ref = jnp.linalg.norm(xref_DS - x_t.reshape((1,-1)), axis=1)
        dist_ref_u = (1-alpha) * dist_ref / 1
        closest_ind_t = jnp.argmin(dist_ref_u)

        # dist_ref_w = jnp.linalg.norm(xref - x_t.reshape((1,-1)), axis=1)
        # dist_ref_u_w = (1-alpha) * dist_ref_w / 1
        # closest_ind_w = jnp.argmin(dist_ref_u_w)

        # closest_ind = jnp.where(control_t==3, closest_ind_w, closest_ind_t)

        closest_ind = closest_ind_t

        ## looking forward

        # ref_range_start = jnp.where(control_t == 3, self.ref_range_start_wiping, self.ref_range_start) # look ahead horizon for wiping stain and nominal target
        # ref_range_start = jnp.where(control_t == 1, self.ref_range_start + 2, ref_range_start)
        ref_range_start = jnp.where(control_t == 1, self.ref_range_start + 3, self.ref_range_start)
        ref_range_start = jnp.where(control_t == 2, self.ref_range_start - 2, ref_range_start)

        chosen_ind = closest_ind + ref_range_start

        # terminal_time = jnp.where(control_t==3, xref_wiping.shape[0]-1, xref_DS.shape[0]-1)
        terminal_time = xref_DS.shape[0]-1

        is_ind_limit = (chosen_ind < terminal_time)

        loop_ind = jnp.where(control_t == 2, terminal_time-1, 0)

        xref_ind = jnp.where(is_ind_limit, chosen_ind, loop_ind)
            
        # xref_t = jnp.where(control_t==3, xref_wiping[xref_ind, :dim].reshape((-1,1)), xref_DS[xref_ind, :dim].reshape((-1,1)))
        # fxref = jnp.where(control_t == 3, xref_vel_wiping[xref_ind, :dim].reshape((-1,1)), xref_vel_DS[xref_ind, :dim].reshape((-1,1)))
        xref_t = xref_DS[xref_ind].reshape((-1,1))
        fxref = xref_vel_DS[xref_ind, :dim].reshape((-1,1))

        ## if controller changed

        trans_period = jnp.where(change_flag, 1, trans_period)

        pos = jnp.where(control_t == 3, dirt_pos, dip_pos)
        
        STL_t = jnp.where(control_t < 3, STL_t, STL_t + self.dt)

        STL_t = jnp.where(trans_period, 0, STL_t)

        STL_t = jnp.where(change_flag, 0., STL_t)
        
        k = jnp.where(change_flag, jnp.linalg.norm(pos - x_t) - epsilon_B + epsilon_margin,
                      k)
                      
        t_star = jnp.where(change_flag, (jnp.linalg.norm(pos - x_t)) / self.v_uniform,
                           t_star)

        ## Different controllers

        branches = [self.target_track, self.target_track, self.target_track, self.go_to_dirt, self.go_to_dip]
        vel_cmd = jax.lax.switch(control_t, branches, x_t, xref_t, fx, fxref, pos,
                                 STL_t, t_star, k, scaler_t)
        
        ## Transition period
        
        trans_period = jnp.where(beta_t>=1, 0, trans_period)

        beta_t = jnp.where(change_flag, 0., beta_t)

        vel_cmd_final, u1 = jax.lax.cond(trans_period, self.trans_controller, 
                                     self.no_trans_controller, 
                                     beta_t, vel_cmd, jnp.copy(vel_cmd_prevs), u1)
        
        beta_t = jnp.where(trans_period, jnp.clip(beta_t + dt_beta_t, 0, 1), 0)
        
        ## Scaling and final command

        # k_scale = jnp.where(control_t == 1, 1.1, 1)

        vel_cmd_robot = jnp.copy(1*vel_cmd_final)

        vel_cmd_robot_p = jnp.where(count<1, jnp.zeros(vel_cmd_robot.shape), vel_cmd_robot)

        # self.STL_t = jnp.copy(STL_t)
        # self.t_star = jnp.copy(t_star)
        # self.k = jnp.copy(k)

        return vel_cmd_robot_p[0], vel_cmd_robot_p[1], vel_cmd_robot_p[2], trans_period, beta_t, change_flag, k, t_star, STL_t, vel_cmd_robot, u1

    def _tree_flatten(self):
        # You might also want to store self.b in either the first group
        # (if it's not hashable) or the second group (if it's hashable)
        return (self.ref_range_start, self.ref_range, self.vopt_max, self.v_uniform, self.xt, self.fxt, self.control_t, self.control_t_p, self.dirt_pos, self.trans_period, 
                self.dt, self.change_flag,), ()

    @classmethod
    def _tree_unflatten(cls, aux, children):
        return cls(*children)

register_pytree_node(ustar_m, ustar_m._tree_flatten, ustar_m._tree_unflatten)

xt = jnp.zeros((3,1))
fxt =  jnp.zeros((3,1))
fxt_wiping =  jnp.zeros((3,1))
control_t = 0
control_t_p = 0
dirt_pos = jnp.zeros((3,1))
dip_pos = jnp.zeros((3,1))

pub = rospy.Publisher('/passiveDS/desired_twist', Twist, queue_size=10)

def franka_callback(state_msg):
    global xt
    O_T_EE = jnp.array(state_msg.O_T_EE).reshape(4, 4).T
    # self.xt = self.xt.at[0, 1].set(O_T_EE[0, 3])
    # self.xt = self.xt.at[1, 1].set(O_T_EE[1, 3])
    # self.xt = self.xt.at[2, 1].set(O_T_EE[2, 3])
    xt = jnp.array(O_T_EE[:3,3]).reshape((-1,1))

def NN_callback(NN_msg):
    global fxt
    fxt = jnp.array([NN_msg.x, NN_msg.y, NN_msg.z]).reshape((-1,1))

def NN_wiping_callback(NN_msg):
    global fxt_wiping
    fxt_wiping = jnp.array([NN_msg.x, NN_msg.y, NN_msg.z]).reshape((-1,1))

def control_t_callback(control_t_msg):
    global control_t_p, control_t
    control_t_p = jnp.copy(control_t)
    control_t = control_t_msg.data

def dirt_pos_callback(dirt_pos_msg):
    global dirt_pos
    dirt_pos = jnp.array([dirt_pos_msg.x, dirt_pos_msg.y, dirt_pos_msg.z]).reshape((-1,1))

def dip_pos_callback(dip_pos_msg):
    global dip_pos
    dip_pos = jnp.array([dip_pos_msg.x, dip_pos_msg.y, dip_pos_msg.z]).reshape((-1,1))

def pub_desired_vel(linearx, lineary, linearz):
    global pub
    # now = time.time()
    desired_twist = Twist()

    # vel = jnp.array([linearx, lineary, linearz])

    desired_twist.linear.x = linearx
    desired_twist.linear.y = lineary
    desired_twist.linear.z = linearz

    # desired_twist.linear.x = vel[0]
    # desired_twist.linear.y = vel[1]
    # desired_twist.linear.z = vel[2]

    # rospy.loginfo("Time for linear: %f ms", 1000*(time.time() - now))
    # now1 = time.time()
    desired_twist.angular.x = 0
    desired_twist.angular.y = 0
    desired_twist.angular.z = 0

    # rospy.loginfo("Time for angualr: %f ms", 1000*(time.time() - now1))

    pub.publish(desired_twist)

def main():
    global xt, fxt, control_t, control_t_p, dirt_pos, dip_pos
    ## Load data

    rospack = rospkg.RosPack()
    curr_path = rospack.get_path('franka_interactive_controllers')

    ## Loading learned DS motion plan + trajectory

    n_DS = 3
    
    curr_data_path = curr_path + '/config/Trajectory_data_eff/'
    xref_all = []
    xref_vel_all = []
    scaler_t_all = []

    for i in range(n_DS):

        ref_file = curr_data_path + 'trajectory_ref_' + str(i) + '.npy'
        with open(ref_file, 'rb') as f:
            xref = jnp.load(f)
            xref_vel = jnp.load(f)
            scaler_t = jnp.load(f)


        xref_all.append(xref)
        xref_vel_all.append(xref_vel)
        scaler_t_all.append(scaler_t)

    ## Initialization for the class object

    freq = 500
    ref_range_start = 6 # from where to start looking forward?
    ref_range = 5 # look how much forward, horizon N
    vopt_max = 1 # maximum virtual control input
    trans_period = 1
    beta_t = 0.
    dt = 1/freq
    change_flag = 0
    v_uniform = 0.2
    
    ## Initialize node

    rospy.init_node('cmd_vel', anonymous=True)
    rate = rospy.Rate(freq) # Hz


    rospy.Subscriber('/franka_state_controller/franka_states', FrankaState, 
                                        franka_callback, queue_size=1, tcp_nodelay=True)
    rospy.Subscriber('/my_DS_plan/ODE_vel', Vector3, NN_callback,
                                    queue_size=1, tcp_nodelay=True)
    rospy.Subscriber('/my_aut_plan/control_t', Int32, control_t_callback,
                                    queue_size=1, tcp_nodelay=True)
    rospy.Subscriber('/my_aut_plan/dirt_pos', Vector3, dirt_pos_callback,
                                    queue_size=1, tcp_nodelay=True)
    rospy.Subscriber('/my_aut_plan/dip_pos', Vector3, dip_pos_callback,
                                    queue_size=1, tcp_nodelay=True)
    rospy.Subscriber('/my_aut_plan/ODE_vel', Vector3, NN_wiping_callback,
                                    queue_size=1, tcp_nodelay=True)
    # /passiveDS/desired_twist
    # /cartesian_impedance_controller/desired_twist

    k = 0.
    t_star = 0.
    STL_t = 0.
    
    control_t_ind = jnp.where(control_t >= n_DS, 0, control_t)
    xref = xref_all[control_t_ind]
    xref_vel = xref_vel_all[control_t_ind]
    scaler_t = scaler_t_all[control_t_ind]

    u1 = jnp.zeros((3,1))
    vel_cmd_prevs = jnp.zeros((3,1))

    ustar_obj = ustar_m(ref_range_start, ref_range, vopt_max, v_uniform, 
                        xt, fxt, control_t, control_t_p, dirt_pos, trans_period, dt, change_flag)

    count = 0

    while not rospy.is_shutdown():
        # O_T_EE = jnp.array(sub_obj.state_data.O_T_EE).reshape(4, 4).T
        now = time.time()
        # rospy.loginfo("Xref shape: %f", xref.shape[1])
        # NN_data_t = sub_obj.NN_data
        # fxt = jnp.array([[NN_data_t.x], [NN_data_t.y], [NN_data_t.z]])

        count += 1
        count = jnp.where(count<100, count, 2)

        control_t_ind = jnp.where(control_t >= n_DS, 0, control_t)
        xref = xref_all[control_t_ind]
        xref_vel = xref_vel_all[control_t_ind]
        scaler_t = scaler_t_all[control_t_ind]
        
        scale_dt_beta = jnp.where(control_t == 4 or control_t == 3, 1.2, 0.5)
        dt_beta_t = dt*scale_dt_beta

        # count = jnp.where(control_t == 3 & change_flag, 0, count)

        f1x, f1y, f1z, trans_period, beta_t, change_flag, k, t_star, STL_t, vel_cmd_prevs, u1 = ustar_obj.compute_ustar_m(count, xt, fxt, control_t, control_t_p, change_flag, trans_period, beta_t, dt_beta_t, k, t_star, STL_t, vel_cmd_prevs, dirt_pos, dip_pos,
                                                                                                                          u1, xref, xref_vel, scaler_t)
        
        # rospy.loginfo("Time for computation: %f ms", 1000*(time.time() - now))
        # rospy.loginfo(count)
        # rospy.loginfo("Current state[0]: %f ", ustar_obj.xt[0, 1])
        # rospy.loginfo("Time scale: %f /s", scaler_t)
        pub_desired_vel(f1x, f1y, f1z)
        # rospy.loginfo("Current state: %f, %f, %f", x_t[0], x_t[1], x_t[2])
        # rospy.loginfo("Change flag: %.1f, Controller: %.1f, Trans_period: %.1f, Beta_t :%.4f", 
        #               change_flag, ustar_obj.control_t, trans_period, beta_t)
        # rospy.loginfo("vel_cmd_norm :%.4f", vel_cmd)
        # rospy.loginfo("k :%.4f", k)
        # rospy.loginfo("Time for computation: %f ms", 1000*(time.time() - now))
        # trans_period = jnp.copy(trans_period1)
        # beta_t = jnp.copy(beta_t1)
        # change_flag = jnp.copy(change_flag1)
        # k = jnp.copy(k1)
        # t_star = jnp.copy(t_star1)  
        # STL_t = jnp.copy(STL_t1)
        # vel_cmd_prevs = jnp.copy(vel_cmd_final)
        rate.sleep()


if __name__ == '__main__':
    try:
        main()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass