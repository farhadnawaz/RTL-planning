#!/usr/bin/python3
import rospy
import rospkg
from catkin.find_in_workspaces import find_in_workspaces
from IPython import embed
from franka_msgs.msg import FrankaState
from geometry_msgs.msg import Vector3, Pose, PoseStamped
from tf.transformations import quaternion_from_matrix
from nav_msgs.msg import Path
from std_msgs.msg import Int32

from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

import time

import diffrax
import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
from jax import jit
from jax.tree_util import register_pytree_node

# ROS node reference Tianyu: https://github.com/tonylitianyu/golfbot/blob/master/nodes/moving#L63

class Func(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, data_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        initializer = jnn.initializers.orthogonal()
        self.mlp = eqx.nn.MLP(
            in_size=data_size,
            out_size=data_size,
            width_size=width_size,
            depth=depth,
            activation=jnn.tanh,
            key=key,
        )
        model_key = key
        key_weights = jrandom.split(model_key, depth+1)

        for i in range(depth+1):
          where = lambda m: m.layers[i].weight
          shape = self.mlp.layers[i].weight.shape
          self.mlp = eqx.tree_at(where, self.mlp, replace=initializer(key_weights[i], shape, dtype = jnp.float32))

    @eqx.filter_jit
    def __call__(self, t, y, args):

        return self.mlp(y)
    
class NeuralODE(eqx.Module):
    func: Func

    def __init__(self, data_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        self.func = Func(data_size, width_size, depth, key=key)

    @eqx.filter_jit
    def __call__(self, ts, yd0):
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func),
            diffrax.Tsit5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=yd0,
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
            saveat=diffrax.SaveAt(ts=ts),
        )
        return solution.ys
    
class Func_rot(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, data_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        initializer = jnn.initializers.orthogonal()
        self.mlp = eqx.nn.MLP(
            in_size=data_size,
            out_size=data_size - 1, # -1 because we only need angular velocity
            width_size=width_size,
            depth=depth,
            activation=jnn.tanh,
            key=key,
        )
        model_key = key
        key_weights = jrandom.split(model_key, depth+1)

        for i in range(depth+1):
          where = lambda m: m.layers[i].weight
          shape = self.mlp.layers[i].weight.shape
          self.mlp = eqx.tree_at(where, self.mlp, replace=initializer(key_weights[i], shape, dtype = jnp.float32))

    @eqx.filter_jit
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

    @eqx.filter_jit
    def __call__(self, t, y, args):

        raw_vect_field = self.mlp(y)

        ang_vel = raw_vect_field[3:]

        quat = y[3:]

        ang_vel_quat = jnp.concatenate((jnp.array([0]), ang_vel))

        quat_vect_field = 0.5*self.quat_mult(ang_vel_quat, quat)

        result_vect_field = jnp.concatenate((raw_vect_field[:3], quat_vect_field))

        return result_vect_field


class NeuralODE_rot(eqx.Module):
    func_rot: Func_rot

    def __init__(self, data_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        self.func_rot = Func_rot(data_size, width_size, depth, key=key)

    @eqx.filter_jit
    def __call__(self, ts, yd0):
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func_rot),
            diffrax.Tsit5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=yd0,
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
            saveat=diffrax.SaveAt(ts=ts),
        )

        pose_traj = solution.ys

        position = pose_traj[:, :3]
        quat = pose_traj[:, 3:]
        quat_norm = jnp.linalg.norm(quat, axis=-1)
        quat_pred_norm = quat/jnp.expand_dims(quat_norm, axis=-1)
        # quat_pred_norm = quat
        pose_traj_result = jnp.concatenate((position, quat_pred_norm), axis=-1)

        return pose_traj


def load_model(traj, model_file):
    _, data_size = traj.shape
    width_size = 64 # 64 for mixing pan split1, wiping_loop_not_touching_board
    depth = 3
    seed = 1000
    model_key = jrandom.PRNGKey(seed)
    if data_size == 3:
        model1 = NeuralODE(data_size, width_size, depth, key=model_key)
    elif data_size == 7:
        model1 = NeuralODE_rot(data_size, width_size, depth, key=model_key)
    model_load = eqx.tree_deserialise_leaves(model_file, model1)

    return model_load

## Messages for visualization

# path = Path()
# path_r = Path()
# path_g = Path()

class NN_output(object):

    def __init__(self, pub, pub_rot, xt):
        self.pub = pub
        self.pub_rot = pub_rot
        self.xt = xt
            
    @eqx.filter_jit
    def get_NN_vel_rot(self, model, xt):

        x = xt[0, 0]
        y = xt[1, 0]
        z = xt[2, 0]
        q_w = jnp.where(xt[3, 0] < 0, -xt[3, 0], xt[3, 0]) 
        q_x = jnp.where(xt[3, 0] < 0, -xt[4, 0], xt[4, 0]) 
        q_y = jnp.where(xt[3, 0] < 0, -xt[5, 0], xt[5, 0]) 
        q_z = jnp.where(xt[3, 0] < 0, -xt[6, 0], xt[6, 0]) 

        # x_p = xt[0, 0] - xt_p[0, 0]
        # y_p = xt[1, 0] - xt_p[1, 0]
        # z_p = xt[2, 0] - xt_p[2, 0]

        # x_d_t = jnp.array([x, y, z, x_p, y_p, z_p])
        x_d_t = jnp.array([x, y, z, q_w, q_x, q_y, q_z])

        fx = model.func_rot.mlp(x_d_t)

        return fx
    
    @eqx.filter_jit
    def get_NN_vel(self, model, xt):

        x = xt[0, 0]
        y = xt[1, 0]
        z = xt[2, 0]

        # x_p = xt[0, 0] - xt_p[0, 0]
        # y_p = xt[1, 0] - xt_p[1, 0]
        # z_p = xt[2, 0] - xt_p[2, 0]

        # x_d_t = jnp.array([x, y, z, x_p, y_p, z_p])
        x_d_t = jnp.array([x, y, z])

        fx = model.func.mlp(x_d_t)

        return fx

    def pub_NN_vel(self, x, y, z):
        NN_vel = Vector3()

        NN_vel.x = x
        NN_vel.y = y
        NN_vel.z = z

        self.pub.publish(NN_vel)

    def pub_NN_vel_rot(self, x, y, z):
        NN_vel_rot = Vector3()

        NN_vel_rot.x = x
        NN_vel_rot.y = y
        NN_vel_rot.z = z

        self.pub_rot.publish(NN_vel_rot)

    def _tree_flatten(self):
        # You might also want to store self.b in either the first group
        # (if it's not hashable) or the second group (if it's hashable)
        return (self.pub, self.pub_rot, self.xt,), ()

    @classmethod
    def _tree_unflatten(cls, aux, children):
        return cls(*children)
    
register_pytree_node(NN_output, NN_output._tree_flatten, NN_output._tree_unflatten)
    
xt = jnp.zeros((3, 1))
qt = jnp.array([1.0, 0.0, 0.0, 0.0]).reshape((-1,1))
control_t = 3

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
    qt = jnp.array([quat_ee[3], quat_ee[0], quat_ee[1], quat_ee[2]]).reshape((-1,1))

def control_t_callback(state_msg):
    global control_t
    control_t = state_msg.data

def main():

    global xt, qt, control_t
    ## Load data

    rospack = rospkg.RosPack()
    curr_path = rospack.get_path('franka_interactive_controllers')

    ## Loading nominal motion plan + trajectory

    data_files = ['trajectory_data_scooping.npy',
                  'trajectory_data_transit_drop.npy',
                  'trajectory_data_stirring.npy']
    data_files_orig = ['trajectory_data_scooping_orig.npy',
                       'trajectory_data_transit_drop_orig.npy',
                       'trajectory_data_stirring_orig.npy']
    model_files = ['/config/STL_scooping.eqx',
                   '/config/STL_transit_drop.eqx',
                   '/config/STL_stirring.eqx']

    curr_data_path = curr_path + '/config/Trajectory_data_eff/'
    traj_load_all = []
    model_load_all = []
    scaler_t_all = []

    train_indx = 0
    split_indx = 0

    n_DS = len(data_files)

    for i in range(n_DS):
        data_file = curr_data_path + data_files[i]
        with open(data_file, 'rb') as f:
            traj_standard = jnp.load(f) # nD x ntrajs (split) x nsamples x 2*dim
            vel_standard = jnp.load(f)
            ts_new = jnp.load(f)
            scaler_all_t_combine = jnp.load(f) # 1/T (end time) nD x ntrajs (split) x 1
            minmax = jnp.load(f) # bounding cube (3 x 2)

        data_file_orig = curr_data_path + data_files_orig[i]
        with open(data_file_orig, 'rb') as f:
            traj_all_combine_process = jnp.load(f) # nD x ntrajs (split) x nsamples x 2*dim
            vel_stavel_all_combine_process = jnp.load(f)
        
        traj_load = traj_all_combine_process[train_indx, split_indx]

        traj_load_all.append(traj_load)# ignore quaternions
        model_file_name = curr_path + model_files[i]
        model_load = load_model(traj_load, model_file_name)
        model_load_all.append(model_load)
        
        end_ind = -1
    
        xref = model_load(jnp.linspace(0, ts_new[end_ind], num=ts_new.shape[0]), traj_load[0, :])
        xref_vel = jax.vmap(model_load.func_rot.mlp, in_axes=0)(xref)

        ref_file = curr_path + '/config/Trajectory_data_eff/' + 'trajectory_ref_' + str(i) + '.npy'
        with open(ref_file, 'wb') as f:
            jnp.save(f, xref)
            jnp.save(f, xref_vel)
            jnp.save(f, scaler_all_t_combine[train_indx, split_indx])

        scaler_t_all.append(scaler_all_t_combine[train_indx, split_indx])

    ## Initialize node
    freq = 500
    rospy.init_node('NODE_model_vel_SO3', anonymous=True)
    rate = rospy.Rate(freq) # Hz

    pub = rospy.Publisher('/my_DS_plan/ODE_vel', Vector3, queue_size=2)
    pub_rot = rospy.Publisher('/my_DS_plan/ODE_vel_rot', Vector3, queue_size=2)
    # pub_wiping_small = rospy.Publisher('/my_aut_plan/ODE_vel', Vector3, queue_size=2)
    rospy.Subscriber('/franka_state_controller/franka_states', FrankaState, franka_callback,
                                          queue_size=1,
                                          tcp_nodelay=True)
    # rospy.Subscriber('/tkinter/control_t', Int32, control_t_callback,
    #                                       queue_size=1,
    #                                       tcp_nodelay=True)
    # rospy.Subscriber('/control_t_ours', Int32, control_t_callback,
    #                                       queue_size=1,
    #                                       tcp_nodelay=True)
    rospy.Subscriber('/my_aut_plan/control_t', Int32, control_t_callback,
                                          queue_size=1,
                                          tcp_nodelay=True)

    ## Initialize object

    NN_obj = NN_output(pub, pub_rot, xt)
    # NN_obj_wiping_small = NN_output_wiping_small(model_load_wiping_small, scaler_all_t_combine[train_indx, split_indx], minmax, pub_wiping_small, xt)
    

    while not rospy.is_shutdown():
        now = time.time()    
        control_t_ind = jnp.where(control_t >= n_DS, 0, control_t)        
        model = model_load_all[control_t_ind]
        NN_vel_value = NN_obj.get_NN_vel_rot(model, xt)
        NN_obj.pub_NN_vel(NN_vel_value[0], NN_vel_value[1], NN_vel_value[2])
        NN_obj.pub_NN_vel_rot(NN_vel_value[3], NN_vel_value[4], NN_vel_value[5])
        # NN_vel_value_wiping_small = NN_obj_wiping_small.get_NN_vel(xt)
        # NN_obj_wiping_small.pub_NN_vel(NN_vel_value_wiping_small[0], NN_vel_value_wiping_small[1], NN_vel_value_wiping_small[2])
        # rospy.loginfo("Time for processing NN node: %f ms", 1000*(time.time() - now))
        rate.sleep()


if __name__ == '__main__':
    try:
        main()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass