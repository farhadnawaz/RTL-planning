# RTL-planning

This repo contains the code base for the below project, which is accepted to appear in [IROS 2024](https://iros2024-abudhabi.org/).

## Paper
Farhad Nawaz, Shaoting Peng, Lars Lindemann, Nadia Figueroa, Nikolai Matni, "Reactive Temporal Logic-based Planning and Control for Interactive Robotic Tasks", _arXiv preprint arXiv:2404.19594_, 2024. (available at [https://arxiv.org/abs/2308.00186](https://arxiv.org/abs/2404.19594)). 

**TL;DR**: We build a discrete task planner (200 Hz) and a continuous motion planner (1 KHz) for reactive time-critical tasks and complex periodic motions. Project webpage: [https://sites.google.com/view/lfd-neural-ode/home](https://sites.google.com/view/rtl-plan-control-hri/home)

## Dataset

The **Data Demos** folder contains the trajectory data for all the experiments. We train on this data to learn the Neural ODE models. 

## Automaton graph and python simulation

The $\texttt{Notebooks}$ folder contains the following two jupyter notebooks.

1. **automaton_graph.ipynb**: to generate the automaton graph given a Reactive Temporal Logic (RTL) specification using the [Spot 2.0](https://spot.lre.epita.fr/) tool. It also has classes to compute the shortest path on the graph and choose controllable propositions, given the uncontrollable propositions (environmental observations).
2. **STL_ My_wiping_nODE.ipynb**: Code to create toy python simulations that illustrate a periodic 2D wiping task with a reaching motion specified by an RTL specification. The reaching motion is executed using a time-varying Control Barrying Function (CBF).

## Implement on the Franka robot arm

Follow the [installation instructions](https://github.com/farhadnawaz/CLF-CBF-NODE?tab=readme-ov-file#installation) from our [prior work](https://github.com/farhadnawaz/CLF-CBF-NODE) to setup and [build](https://github.com/farhadnawaz/CLF-CBF-NODE?tab=readme-ov-file#catkin-make) the ROS workspace for both gazebo simulation and real robot implementation on the Franka robot arm. 

### Experiment setup

* Move the files from the **config** and **launch** folder to ``/franka_interactive_controllers/config/`` and ``/franka_interactive_controllers/launch/``, respectively. The **config** folder containts the Neural ODE models (<model>.eqx), reference trajectories (<traj>.npy) and automaton graphs (<graph.dictionary>) for planning.

* For a specific <experiment>, move all the scripts from ``<path-to-scripts>`` to ``/franka_interactive_controllers/scripts/``, where ``<path-to-scripts>``:=``/Gazebo_scripts/<experiment>`` for gazebo simulation, or ``/Lab_PC_scripts/<experiment>`` for real robot implementation. 

* For gazebo, check the launch file ``/franka_interactive_controllers/launch/simulate_panda_gazebo.launch`` for the initial joint configuration of the task you want to run.

### Running the experiment

* In the first terminal, run 

```
roslaunch franka_interactive_controllers <launch-file> controller:=passiveDS_impedance
```

where, ``launch-file``:=``simulate_panda_gazebo.launch`` for gazebo simulation, or ``franka_interactive_bringup.launch`` for real robot implementation. We can also change the ``controller`` from ``passiveDS_impedance`` to ``cartesian_twist_impedance``, etc.

* In the second terminal, run 

```
roslaunch franka_interactive_controllers NODE_model.launch
```

Change the name of the ROS nodes to ``<node>_SO3.py`` in the launch file for cooking experiment. 

* In the third terminal, run 
```
rosrun franka_interactive_controllers cmd_vel.py
```

Run cmd_vel_SO3.py instead for cooking experiment.
