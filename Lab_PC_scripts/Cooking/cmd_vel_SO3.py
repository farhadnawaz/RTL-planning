#!/usr/bin/python3
import rospy, numpy as np
import rospkg
from catkin.find_in_workspaces import find_in_workspaces
from IPython import embed
from franka_msgs.msg import FrankaState
from geometry_msgs.msg import Twist, Vector3, Pose, PoseStamped
from tf.transformations import quaternion_from_matrix
from nav_msgs.msg import Path
from functools import partial
from jax.tree_util import register_pytree_node
from std_msgs.msg import Int32

import time

import jax
from jax import jit
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
from jaxopt import OSQP

jax.config.update("jax_enable_x64", True)

# import cvxpy as cp

# from sklearn.preprocessing import MinMaxScaler
# import plotly.graph_objects as go


# ROS node reference Tianyu: https://github.com/tonylitianyu/golfbot/blob/master/nodes/moving#L63

count_start_point = 0

class ustar_m:

    def __init__(self, xref, xref_vel, ref_range_start, ref_range, vopt_max, dt_beta_t):
        self.xref = xref
        self.xref_vel = xref_vel
        self.ref_range_start = ref_range_start # from where to start looking forward?
        self.ref_range = ref_range # look how much forward, horizon N
        self.vopt_max = vopt_max # maximum virtual control input
        self.dt_beta_t = dt_beta_t
        # /passiveDS/desired_twist
        # /cartesian_impedance_controller/desired_twist

    @jit
    def quat_mult(self, q1, q2):
        q1_s = q1[0]
        q2_s = q2[0]
        q1_v = q1[1:].reshape((-1,1))
        q2_v = q2[1:].reshape((-1,1))
        scalar = q1_s*q2_s - q1_v.T @ q2_v
        skew = jnp.array([[0, -q1[3], q1[2]],
                            [q1[3], 0, -q1[1]],
                            [-q1[2], q1[1], 0]])
        vector = q1_s*q2_v + q2_s*q1_v + skew @ q2_v
        q_result = jnp.concatenate((scalar, vector), axis=0).flatten()
        return q_result
    
    @jit
    def target_track_stir(self, count, p_t, q_t, fp_t, w_t, xref, xref_vel, scaler_t, control_t, point):

        global count_start_point

        dist_ref = jnp.linalg.norm(xref[:, :3] - p_t.reshape((1,-1)), axis=1)
        closest_ind = jnp.argmin(dist_ref)

        # ref_range_start = jnp.where(control_t ==2, self.ref_range_start + 10, self.ref_range_start)

        chosen_ind = closest_ind + self.ref_range_start

        is_ind_limit = chosen_ind < xref.shape[0] - 1

        xref_ind = jnp.where(is_ind_limit, chosen_ind, 30)

        # xref_ind = jnp.where(is_ind_limit, chosen_ind, loop_ind)

        # count_start_point = jnp.where(is_ind_limit, count_start_point+1, 0)
        # xref_ind = jnp.where(count_start_point < 10, loop_ind, chosen_ind)

        # xref_ind = i

        # xref_ind = -1

        xref_t = xref[xref_ind].reshape((-1,1)) # dim x 1

        # xref_t = xref_g

        qref = xref_t[3:]
        # qref = self.quat_mult(q_t[:, 0], jnp.array([jnp.cos(theta/2), jnp.sin(theta/2)]))
        pref = xref_t[:3]
        # qref = jnp.array([-0.77427671,  0.42536951,  0.22131513,  0.41300843]).reshape((-1,1))
        # qref = jnp.array([0.66429895, -0.21115419,  0.67721987,  0.23557176]).reshape((-1,1))

        scale_q = 1

        scale_p = 1

        # print(Q_opt.shape)
        e_q = q_t - qref
        # e_q = e_q/jnp.linalg.norm(e_q)
        e_p = p_t - pref
        V = jnp.sum(jnp.square(2*e_p))/4 + jnp.sum(jnp.square(2*e_q))/4

        g_p_x = fp_t

        w_g_x = w_t
        g_w_x = jnp.vstack((0.0, w_g_x))

        g_xref = xref_vel[xref_ind].reshape((-1,1))
        g_p_xref = g_xref[:3]

        w_g_xref = g_xref[3:].reshape((-1,1))
        g_w_xref = jnp.vstack((0.0, w_g_xref))

        # print(g_w_xref.shape)

        quat_mult_q = self.quat_mult(g_w_x[:, 0], q_t[:, 0])
        quat_mult_q_ref = self.quat_mult(g_w_xref[:, 0], qref[:, 0])

        s_p = jnp.vdot(e_p, g_p_x - g_p_xref)
        s_q = jnp.vdot(e_q/2.0, quat_mult_q.reshape((-1,1)) - quat_mult_q_ref.reshape((-1,1)))

        s = s_p + s_q

        Q = jnp.array([[q_t[0, 0], q_t[1, 0], q_t[2, 0], q_t[3, 0]],
                        [-q_t[1, 0], q_t[0,0], q_t[3, 0], -q_t[2, 0]],
                        [-q_t[2, 0], -q_t[3, 0], q_t[0, 0], q_t[1, 0]],
                        [-q_t[3, 0], q_t[2, 0], -q_t[1, 0], q_t[0, 0]]])
        Q2 = Q[:, 1:] # 4 x 3
        Q2_minus = -Q2
        Q2_1 = jnp.vstack((Q2_minus[0, :], -Q2_minus[1:,:]))

        alpha_V = 40

        G_opt = 2*jnp.hstack((e_p.T, (e_q.T/2.0) @ Q2_1))
        h_opt = jnp.array([-alpha_V*V - 2*s])

        # qp = OSQP()

        # sol = qp.run(params_obj=(Q_opt,jnp.ones(Q_opt.shape[0])), params_ineq=(G_opt, h_opt)).params

        # virtual_u = sol.primal.reshape((-1,1))

        ustar_closed = (-alpha_V*V - 2*s) * G_opt.T / (jnp.sum(jnp.square(G_opt)))

        virtual_u = jnp.where(2*s <= -alpha_V*V, jnp.zeros((6,1)), ustar_closed)

        ## bound on the magnitude of velocity

        virtual_u_final = virtual_u.at[:3].set(
            jnp.where(jnp.linalg.norm(virtual_u[:3]) >= 2*self.vopt_max*(1/scaler_t), 
                      (virtual_u[:3]/jnp.linalg.norm(virtual_u[:3])) * 2*self.vopt_max*(1/scaler_t), virtual_u[:3]))
        
        virtual_u_final_f = virtual_u_final.at[3:].set(
            jnp.where(jnp.linalg.norm(virtual_u_final[3:]) >= 2*jnp.pi*(1/scaler_t), 
                      (virtual_u_final[3:]/jnp.linalg.norm(virtual_u_final[3:])) * 2*jnp.pi*(1/scaler_t), virtual_u_final[3:]))

        vel_cmd_pos = fp_t + virtual_u_final_f[:3]
        vel_cmd_rot = w_t + virtual_u_final_f[3:]

        vel_cmd_pos = fp_t + virtual_u_final[:3]
        vel_cmd_rot = w_t + virtual_u_final_f[3:]

        # k_angular = jnp.where(jnp.linalg.norm(q_t - qref) < 0.001, 0.0, 1.0)
        # k_scale =jnp.where(control_t == 2, 0.3, 0.2)
        k_scale = 0.2
        k_linear = 0.4
        k_angular = 0.6

        vel_cmd_pos_final = vel_cmd_pos * scaler_t * k_linear
        vel_cmd_rot_final = vel_cmd_rot * scaler_t * k_angular

        vel_cmd = jnp.vstack((vel_cmd_pos_final, vel_cmd_rot_final))

        return vel_cmd, qref

    
    @jit
    def target_track_SO3(self, count, p_t, q_t, fp_t, w_t, xref, xref_vel, scaler_t, control_t, point):

        global count_start_point

        dist_ref = jnp.linalg.norm(xref[:, :3] - p_t.reshape((1,-1)), axis=1)
        closest_ind = jnp.argmin(dist_ref)

        # ref_range_start = jnp.where(control_t ==2, self.ref_range_start + 10, self.ref_range_start)

        chosen_ind = closest_ind + self.ref_range_start

        is_ind_limit = chosen_ind < xref.shape[0]

        xref_ind = jnp.where(is_ind_limit, chosen_ind, -1)

        # xref_ind = jnp.where(is_ind_limit, chosen_ind, loop_ind)

        # count_start_point = jnp.where(is_ind_limit, count_start_point+1, 0)
        # xref_ind = jnp.where(count_start_point < 10, loop_ind, chosen_ind)

        # xref_ind = i

        # xref_ind = -1

        xref_t = xref[xref_ind].reshape((-1,1)) # dim x 1

        # xref_t = xref_g

        qref = xref_t[3:]
        # qref = self.quat_mult(q_t[:, 0], jnp.array([jnp.cos(theta/2), jnp.sin(theta/2)]))
        pref = xref_t[:3]
        # qref = jnp.array([-0.77427671,  0.42536951,  0.22131513,  0.41300843]).reshape((-1,1))
        # qref = jnp.array([0.66429895, -0.21115419,  0.67721987,  0.23557176]).reshape((-1,1))

        scale_q = 1

        scale_p = 1

        # print(Q_opt.shape)
        e_q = q_t - qref
        # e_q = e_q/jnp.linalg.norm(e_q)
        e_p = p_t - pref
        V = jnp.sum(jnp.square(2*e_p))/4 + jnp.sum(jnp.square(2*e_q))/4

        g_p_x = fp_t

        w_g_x = w_t
        g_w_x = jnp.vstack((0.0, w_g_x))

        g_xref = xref_vel[xref_ind].reshape((-1,1))
        g_p_xref = g_xref[:3]

        w_g_xref = g_xref[3:].reshape((-1,1))
        g_w_xref = jnp.vstack((0.0, w_g_xref))

        # print(g_w_xref.shape)

        quat_mult_q = self.quat_mult(g_w_x[:, 0], q_t[:, 0])
        quat_mult_q_ref = self.quat_mult(g_w_xref[:, 0], qref[:, 0])

        s_p = jnp.vdot(e_p, g_p_x - g_p_xref)
        s_q = jnp.vdot(e_q/2.0, quat_mult_q.reshape((-1,1)) - quat_mult_q_ref.reshape((-1,1)))

        s = s_p + s_q

        Q = jnp.array([[q_t[0, 0], q_t[1, 0], q_t[2, 0], q_t[3, 0]],
                        [-q_t[1, 0], q_t[0,0], q_t[3, 0], -q_t[2, 0]],
                        [-q_t[2, 0], -q_t[3, 0], q_t[0, 0], q_t[1, 0]],
                        [-q_t[3, 0], q_t[2, 0], -q_t[1, 0], q_t[0, 0]]])
        Q2 = Q[:, 1:] # 4 x 3
        Q2_minus = -Q2
        Q2_1 = jnp.vstack((Q2_minus[0, :], -Q2_minus[1:,:]))

        alpha_V = 50

        G_opt = 2*jnp.hstack((e_p.T, (e_q.T/2.0) @ Q2_1))
        h_opt = jnp.array([-alpha_V*V - 2*s])

        # qp = OSQP()

        # sol = qp.run(params_obj=(Q_opt,jnp.ones(Q_opt.shape[0])), params_ineq=(G_opt, h_opt)).params

        # virtual_u = sol.primal.reshape((-1,1))

        ustar_closed = (-alpha_V*V - 2*s) * G_opt.T / (jnp.sum(jnp.square(G_opt)))

        virtual_u = jnp.where(2*s <= -alpha_V*V, jnp.zeros((6,1)), ustar_closed)

        ## bound on the magnitude of velocity

        virtual_u_final = virtual_u.at[:3].set(
            jnp.where(jnp.linalg.norm(virtual_u[:3]) >= 2*self.vopt_max*(1/scaler_t), 
                      (virtual_u[:3]/jnp.linalg.norm(virtual_u[:3])) * 2*self.vopt_max*(1/scaler_t), virtual_u[:3]))
        
        virtual_u_final_f = virtual_u_final.at[3:].set(
            jnp.where(jnp.linalg.norm(virtual_u_final[3:]) >= 2*jnp.pi*(1/scaler_t), 
                      (virtual_u_final[3:]/jnp.linalg.norm(virtual_u_final[3:])) * 2*jnp.pi*(1/scaler_t), virtual_u_final[3:]))

        vel_cmd_pos = fp_t + virtual_u_final_f[:3]
        vel_cmd_rot = w_t + virtual_u_final_f[3:]

        # k_angular = jnp.where(jnp.linalg.norm(q_t - qref) < 0.001, 0.0, 1.0)
        # k_scale =jnp.where(control_t == 2, 0.3, 0.2)
        k_scale = 0.2
        k_linear = 0.4
        k_angular = 0.6

        vel_cmd_pos_final = vel_cmd_pos * scaler_t * k_linear
        vel_cmd_rot_final = vel_cmd_rot * scaler_t * k_angular

        vel_cmd = jnp.vstack((vel_cmd_pos_final, vel_cmd_rot_final))
    
        return vel_cmd, qref


    @jit
    def go_to_point(self, count, p_t, q_t, fp_t, w_t, xref, xref_vel, scaler_t, control_t, point):

        global count_start_point

        # ref_range_start = jnp.where(control_t ==2, self.ref_range_start + 10, self.ref_range_start)

        # xref_ind = jnp.where(is_ind_limit, chosen_ind, loop_ind)

        # count_start_point = jnp.where(is_ind_limit, count_start_point+1, 0)
        # xref_ind = jnp.where(count_start_point < 10, loop_ind, chosen_ind)

        # xref_ind = i

        # xref_ind = -1

        xref_t = point.reshape((-1,1))

        # xref_t = xref_g

        qref = xref_t[3:]
        # qref = self.quat_mult(q_t[:, 0], jnp.array([jnp.cos(theta/2), jnp.sin(theta/2)]))
        pref = xref_t[:3]
        # qref = jnp.array([-0.77427671,  0.42536951,  0.22131513,  0.41300843]).reshape((-1,1))
        # qref = jnp.array([0.66429895, -0.21115419,  0.67721987,  0.23557176]).reshape((-1,1))

        scale_q = 1

        scale_p = 1

        # print(Q_opt.shape)
        e_q = q_t - qref
        # e_q = e_q/jnp.linalg.norm(e_q)
        e_p = p_t - pref
        V = jnp.sum(jnp.square(2*e_p))/4 + jnp.sum(jnp.square(2*e_q))/4

        g_p_x = 0.0*fp_t

        w_g_x = w_t
        g_w_x = jnp.vstack((0.0, w_g_x))

        g_xref = xref_vel[-1].reshape((-1,1))
        g_p_xref = 0.0*g_xref[:3]

        w_g_xref = g_xref[3:].reshape((-1,1))
        g_w_xref = jnp.vstack((0.0, w_g_xref))

        # print(g_w_xref.shape)

        quat_mult_q = self.quat_mult(g_w_x[:, 0], q_t[:, 0])
        quat_mult_q_ref = self.quat_mult(g_w_xref[:, 0], qref[:, 0])

        s_p = jnp.vdot(e_p, g_p_x - g_p_xref)
        s_q = jnp.vdot(e_q/2.0, quat_mult_q.reshape((-1,1)) - quat_mult_q_ref.reshape((-1,1)))

        s = s_p + s_q

        Q = jnp.array([[q_t[0, 0], q_t[1, 0], q_t[2, 0], q_t[3, 0]],
                        [-q_t[1, 0], q_t[0,0], q_t[3, 0], -q_t[2, 0]],
                        [-q_t[2, 0], -q_t[3, 0], q_t[0, 0], q_t[1, 0]],
                        [-q_t[3, 0], q_t[2, 0], -q_t[1, 0], q_t[0, 0]]])
        Q2 = Q[:, 1:] # 4 x 3
        Q2_minus = -Q2
        Q2_1 = jnp.vstack((Q2_minus[0, :], -Q2_minus[1:,:]))

        alpha_V = 20

        G_opt = 2*jnp.hstack((e_p.T, (e_q.T/2.0) @ Q2_1))
        h_opt = jnp.array([-alpha_V*V - 2*s])

        # qp = OSQP()

        # sol = qp.run(params_obj=(Q_opt,jnp.ones(Q_opt.shape[0])), params_ineq=(G_opt, h_opt)).params

        # virtual_u = sol.primal.reshape((-1,1))

        ustar_closed = (-alpha_V*V - 2*s) * G_opt.T / (jnp.sum(jnp.square(G_opt)))

        virtual_u = jnp.where(2*s <= -alpha_V*V, jnp.zeros((6,1)), ustar_closed)

        ## bound on the magnitude of velocity

        virtual_u_final = virtual_u.at[:3].set(
            jnp.where(jnp.linalg.norm(virtual_u[:3]) >= 2*self.vopt_max*(1/scaler_t), 
                      (virtual_u[:3]/jnp.linalg.norm(virtual_u[:3])) * 2*self.vopt_max*(1/scaler_t), virtual_u[:3]))
        
        virtual_u_final_f = virtual_u_final.at[3:].set(
            jnp.where(jnp.linalg.norm(virtual_u_final[3:]) >= 2*jnp.pi*(1/scaler_t), 
                      (virtual_u_final[3:]/jnp.linalg.norm(virtual_u_final[3:])) * 2*jnp.pi*(1/scaler_t), virtual_u_final[3:]))

        vel_cmd_pos = fp_t + virtual_u_final_f[:3]
        vel_cmd_rot = w_t + virtual_u_final_f[3:]

        # k_angular = jnp.where(jnp.linalg.norm(q_t - qref) < 0.001, 0.0, 1.0)
        # k_scale =jnp.where(control_t == 2, 0.3, 0.2)
        k_scale = 0.2
        k_linear = 0.4
        k_angular = 0.6

        vel_cmd_pos_final = vel_cmd_pos * scaler_t * k_linear
        vel_cmd_rot_final = vel_cmd_rot * scaler_t * k_angular

        vel_cmd = jnp.vstack((vel_cmd_pos_final, vel_cmd_rot_final))
    
        return vel_cmd, qref
    
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


    @jit
    def compute_ustar_m(self, count, p_t, q_t, fp_t, w_t, control_t, control_t_p, change_flag, trans_period, beta_t,
                        vel_cmd_prevs, scoop_start, go_up, u1, xref, xref_vel, scaler_t):
        ## Different controllers

        change_flag = jnp.where(control_t == control_t_p, 0, 1)

        trans_period = jnp.where(change_flag, 1, trans_period)

        point = jnp.where(control_t == 3, scoop_start, go_up)

        point = jnp.where(control_t == 4, go_up, point)

        branches = [self.target_track_SO3, self.target_track_SO3, self.target_track_stir, self.go_to_point, self.go_to_point]

        vel_cmd, qref = jax.lax.switch(control_t, branches, count, p_t, q_t, fp_t, w_t, xref, xref_vel, scaler_t, control_t, point)

        ## Transition period
        
        trans_period = jnp.where(beta_t>=1, 0, trans_period)

        beta_t = jnp.where(change_flag, 0., beta_t)

        vel_cmd_final, u1 = jax.lax.cond(trans_period, self.trans_controller, 
                                     self.no_trans_controller, 
                                     beta_t, vel_cmd, jnp.copy(vel_cmd_prevs), u1)

        vel_cmd_robot = jnp.copy(vel_cmd_final)
        
        beta_t = jnp.where(trans_period, jnp.clip(beta_t + self.dt_beta_t, 0, 1), 0)

        vel_cmd_final_cmd = jnp.where(count<1, jnp.zeros(vel_cmd_robot.shape), vel_cmd_robot)

        return vel_cmd_final_cmd[0], vel_cmd_final_cmd[1], vel_cmd_final_cmd[2], vel_cmd_final_cmd[3], vel_cmd_final_cmd[4], vel_cmd_final_cmd[5], beta_t, change_flag, trans_period, vel_cmd_final_cmd, u1
    
    def _tree_flatten(self):
        # You might also want to store self.b in either the first group
        # (if it's not hashable) or the second group (if it's hashable)
        return (self.xref, self.xref_vel, self.ref_range_start, 
                self.ref_range, self.vopt_max, self.dt_beta_t,), ()

    @classmethod
    def _tree_unflatten(cls, aux, children):
        return cls(*children)

register_pytree_node(ustar_m, ustar_m._tree_flatten, ustar_m._tree_unflatten)

xt = jnp.zeros((3,1))
qt = jnp.array([1.0, 0.0, 0.0, 0.0]).reshape((-1,1))
fxt =  jnp.zeros((3,1))
fxt_rot =  jnp.zeros((3,1))
control_t = 0
control_t_p = 0

pub = rospy.Publisher('/passiveDS/desired_twist', Twist, queue_size=10)

def q_from_R(R):
    """ generates quaternion from 3x3 rotation matrix """
    _R = jnp.eye(4)
    _R = _R.at[:3, :3].set(R)
    return quaternion_from_matrix(_R)

def franka_callback(state_msg):
    global xt, qt
    O_T_EE = jnp.array(state_msg.O_T_EE).reshape(4, 4).T
    xt = jnp.array([O_T_EE[0, 3], O_T_EE[1, 3], O_T_EE[2, 3]]).reshape((-1,1))
    quat_ee = q_from_R(O_T_EE[:3, :3]) # x, y, z, w
    q_w = jnp.where(quat_ee[3] < 0, -quat_ee[3], quat_ee[3]) 
    q_x = jnp.where(quat_ee[3] < 0, -quat_ee[0], quat_ee[0])
    q_y = jnp.where(quat_ee[3] < 0, -quat_ee[1], quat_ee[1])  
    q_z = jnp.where(quat_ee[3] < 0, -quat_ee[2], quat_ee[2]) 
    qt = jnp.array([q_w, q_x, q_y, q_z]).reshape((-1,1))

def NN_callback(NN_msg):
    global fxt
    fxt = jnp.array([NN_msg.x, NN_msg.y, NN_msg.z]).reshape((-1,1))

def NN_callback_rot(NN_msg):
    global fxt_rot
    fxt_rot = jnp.array([NN_msg.x, NN_msg.y, NN_msg.z]).reshape((-1,1))

def control_t_callback(control_t_msg):
    global control_t_p, control_t
    control_t_p = jnp.copy(control_t)
    control_t = control_t_msg.data

def pub_desired_vel_rot(linearx, lineary, linearz, angularx, angulary, angularz):
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
    desired_twist.angular.x = angularx
    desired_twist.angular.y = angulary
    desired_twist.angular.z = angularz

    # rospy.loginfo("Time for angualr: %f ms", 1000*(time.time() - now1))

    pub.publish(desired_twist)

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

    global xt, qt, fxt, fxt_rot, count_start_point, control_t, control_t_p
    ## Load data

    rospack = rospkg.RosPack()
    curr_path = rospack.get_path('franka_interactive_controllers')

    n_DS = 3

    curr_data_path = curr_path + '/config/Trajectory_data_eff/'
    xref_all = []
    xref_vel_all = []
    scaler_t_all = []

    for i in range(n_DS):

        step = 1
        start_ind = 0
        end_ind = -1

        ref_file = curr_data_path + 'trajectory_ref_' + str(i) + '.npy'
        with open(ref_file, 'rb') as f:
            xref = jnp.load(f)[start_ind:end_ind]
            xref_vel = jnp.load(f)[start_ind:end_ind]
            scaler_t = jnp.load(f)

        if i == 0:
            scoop_start = xref[0]
            scoop_end = xref[-1]
        elif i == 1:
            drop_end = xref[-1] 
        elif i == 2:
            go_up = xref[0] # + jnp.array([0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0])
            print("Go up shape: ", go_up.shape)

        xref_all.append(xref)
        xref_vel_all.append(xref_vel)
        scaler_t_all.append(scaler_t)

    ## Initialization for the class object

    freq = 500
    ref_range_start = 5 # from where to start looking forward?
    ref_range = 5 # look how much forward, horizon N
    vopt_max = 1 # maximum virtual control input
    trans_period = 1
    beta_t = 0.
    dt = 1/freq
    scale_dt_beta = 2
    dt_beta_t = dt*scale_dt_beta
    change_flag = 0
    
    ## Initialize node

    rospy.init_node('cmd_vel_SO3', anonymous=True)
    rate = rospy.Rate(freq) # Hz

    rospy.Subscriber('/franka_state_controller/franka_states', FrankaState, 
                                        franka_callback, queue_size=1, tcp_nodelay=True)
    rospy.Subscriber('/my_DS_plan/ODE_vel', Vector3, NN_callback,
                                    queue_size=1, tcp_nodelay=True)
    rospy.Subscriber('/my_DS_plan/ODE_vel_rot', Vector3, NN_callback_rot,
                                    queue_size=1, tcp_nodelay=True)
    rospy.Subscriber('/my_aut_plan/control_t', Int32, control_t_callback,
                                    queue_size=1, tcp_nodelay=True)
    
    control_t_ind = jnp.where(control_t >= n_DS, 0, control_t)
    xref = xref_all[control_t_ind]
    xref_vel = xref_vel_all[control_t_ind]
    scaler_t = scaler_t_all[control_t_ind]

    ustar_obj = ustar_m(xref, xref_vel, ref_range_start, ref_range, vopt_max, dt_beta_t)
    count = 0
    count_switch = 0
    stationary_point = 0

    u1 = jnp.zeros((6,1))
    vel_cmd_prevs = jnp.zeros((6,1))
    
    while not rospy.is_shutdown():
        now = time.time()
        count += 1
        count = jnp.where(count<100, count, 1)

        control_t_ind = jnp.where(control_t >= n_DS, 0, control_t)
                           
        xref = xref_all[control_t_ind]
        xref_vel = xref_vel_all[control_t_ind]
        scaler_t = scaler_t_all[control_t_ind]

        f1x, f1y, f1z, f1x_rot, f1y_rot, f1z_rot, beta_t, trans_period, change_flag, vel_cmd_prevs, u1 = ustar_obj.compute_ustar_m(count, xt, qt, fxt, fxt_rot, 
                                                                                                                     control_t, control_t_p, change_flag, trans_period, beta_t, 
                                                                                                                     vel_cmd_prevs, scoop_start, go_up, u1, xref, xref_vel, scaler_t)
        # print("closeness to qref: ", jnp.linalg.norm(qref - q_t))
        if jnp.linalg.norm(jnp.array([f1x_rot, f1y_rot, f1z_rot])) > jnp.pi:
            print("angular velocity: ", jnp.linalg.norm(jnp.array([f1x_rot, f1y_rot, f1z_rot])))
        if jnp.linalg.norm(jnp.array([f1x, f1y, f1z])) > 2*vopt_max:
            print("Linear velocity: ", jnp.linalg.norm(jnp.array([f1x, f1y, f1z])))

        # if control_t == 3:
        #     print("Closeness to goal: ", jnp.linalg.norm(xt[:, 0] - scoop_start[:3]))                                      
        
        pub_desired_vel_rot(f1x, f1y, f1z, f1x_rot, f1y_rot, f1z_rot)
        rate.sleep()


if __name__ == '__main__':
    try:
        main()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass