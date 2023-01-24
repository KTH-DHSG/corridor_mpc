#!/usr/bin/env python
import numpy as np

from corridor_mpc.models.ff_kinematics import FreeFlyerKinematics
from corridor_mpc.controllers.corridor_mpc import CorridorMPC
from corridor_mpc.simulation_trajectory import EmbeddedSimEnvironment

# Sim and MPC Params
SIM_TIME = 15.0
# SIM_TIME = 0.1
Q = np.diag([100, 100, 100, 50, 50, 50])
R = np.diag([50, 50, 50, 30, 30, 30])
P = Q * 100

# Instantiante Model
abee = FreeFlyerKinematics()

# Instantiate controller (to track a velocity)
ctl = CorridorMPC(model=abee,
                  dynamics=abee.model,
                  horizon=.3,
                  solver_type='ipopt',
                  Q=Q, R=R, P=P,
                  ulb=abee.ulb,
                  uub=abee.uub,
                  set_zcbf=True)

# Sinusoidal Trajectory
xr0 = np.zeros((6, 1))
abee.set_trajectory(length=SIM_TIME, start=xr0)
sim_env_full = EmbeddedSimEnvironment(model=abee,
                                      dynamics=abee.model,
                                      ctl_class=ctl,
                                      controller=ctl.mpc_controller,
                                      noise={"pos": 0.1, "att": 0.1},
                                      time=SIM_TIME, collect=True,
                                      animate=False)
sim_env_full.use_trajectory_control(True)
_, _, _, avg_ct = sim_env_full.run([1.32, 0, 0, 0, 0.4, 0])

print("Average computational cost:", avg_ct)
