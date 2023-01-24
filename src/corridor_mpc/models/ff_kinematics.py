from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import casadi as ca
import numpy as np


class FreeFlyerKinematics(object):
    def __init__(self, dt=None):
        """
        Astrobee Robot kinematics class class.

        :param mass: mass of the Astrobee
        :type mass: float
        :param inertia: inertia tensor of the Astrobee
        :type inertia: np.diag
        :param dt: sampling time of the discrete system, defaults to 0.01
        :type dt: float, optional
        """

        # Model
        self.nonlinear_model = self.astrobee_dynamics
        self.model = None
        self.n = 6
        self.m = 6

        # Tracking
        self.total_trajectory_time = None

        # Control bounds
        self.max_v = 0.5
        self.max_w = 0.2
        self.ulb = [-self.max_v, -self.max_v, -self.max_v, -self.max_w, -self.max_w, -self.max_w]
        self.uub = [self.max_v, self.max_v, self.max_v, self.max_w, self.max_w, self.max_w]

        # Trajectory reference
        vx = self.max_v * 0.1
        wz = self.max_w * 0.1
        self.constant_v_tracking = np.array([[vx, 0, 0, 0, 0, wz]]).T

        self.set_casadi_options()
        self.set_system_constants()
        self.set_dynamics()
        self.set_barrier_functions()

    def set_casadi_options(self):
        """
        Helper function to set casadi options.
        """
        self.fun_options = {
            "jit": False,
            "jit_options": {"flags": ["-O2"]}
        }

    def set_dynamics(self):
        """
        Helper function to populate Astrobee's dynamics.
        """

        self.model = self.rk4_integrator(self.astrobee_dynamics)

        return

    def psi(self, euler):
        """
        Body to Inertial Attitude jacoboian matrix

        :param euler: euler vector with (roll, pitch,  yaw)
        :type euler: ca.MX, ca.DM, np.ndarray
        :return: attitude jacobian matrix
        :rtype: ca.MX
        """
        phi = euler[0]
        varphi = euler[1]

        Psi = ca.MX.zeros((3, 3))
        Psi[0, 0] = 1
        Psi[0, 1] = ca.sin(phi) * ca.tan(varphi)
        Psi[0, 2] = ca.cos(phi) * ca.tan(varphi)

        Psi[1, 1] = ca.cos(phi)
        Psi[1, 2] = -ca.sin(phi)

        Psi[2, 1] = ca.sin(phi) / ca.cos(varphi)
        Psi[2, 2] = ca.cos(phi) / ca.cos(varphi)

        return Psi

    def astrobee_dynamics(self, x, u):
        """
        Pendulum nonlinear dynamics.

        :param x: state
        :type x: ca.MX
        :param u: control input
        :type u: ca.MX
        :return: state time derivative
        :rtype: ca.MX
        """

        # State extraction
        t = x[3:]

        # 3D Linear velocity
        v = u[0:3]

        # 3D Angular velocity
        w = u[3:]

        # Model
        pdot = v
        tdot = self.psi(t) @ w

        dxdt = [pdot, tdot]

        return ca.vertcat(*dxdt)

    def rk4_integrator(self, dynamics):
        """
        Runge-Kutta 4th Order discretization.

        :param x: state
        :type x: ca.MX
        :param u: control input
        :type u: ca.MX
        :return: state at next step
        :rtype: ca.MX
        """
        x0 = ca.MX.sym('x0', self.n, 1)
        u = ca.MX.sym('u', self.m, 1)

        x = x0

        k1 = dynamics(x, u)
        k2 = dynamics(x + self.dt / 2 * k1, u)
        k3 = dynamics(x + self.dt / 2 * k2, u)
        k4 = dynamics(x + self.dt * k3, u)
        xdot = x0 + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        # Normalize quaternion: TODO(Pedro-Roque): check best way to propagate
        rk4 = ca.Function('RK4', [x0, u], [xdot], self.fun_options)

        return rk4

    def get_trajectory(self, t0, npoints):
        """
        Generate trajectory to be followed.

        :param x0: starting position
        :type x0: ca.DM
        :param t0: starting time
        :type t0: float
        :param npoints: number of trajectory points
        :type npoints: int
        :return: trajectory with shape (Nx, npoints)
        :rtype: np.array
        """

        if t0 == 0.0:
            print("Creating trajectory...", end="")
            # Trajectory params
            traj_points = int(self.trajectory_time / self.dt) + npoints

            # Generate u_r
            u_r = self.constant_v_tracking
            u_r = np.repeat(u_r, traj_points, 1)

            # Initial trajectory point
            xr0 = self.trajectory_start_point

            # Generate the reference trajectory from the system dynamics
            self.trajectory_set = np.empty((self.m, 0))
            self.trajectory_set = np.append(self.trajectory_set, xr0, axis=1)
            for i in range(traj_points - 1):
                x_ri = self.model(self.trajectory_set[:, -1].reshape((self.m, 1)), u_r[:, i])
                self.trajectory_set = np.append(self.trajectory_set, x_ri, axis=1)
            x_r = self.trajectory_set[:, 0:npoints]
            print(" Done")
        else:
            id_s = int(round(t0 / self.dt))
            id_e = int(round(t0 / self.dt)) + npoints
            x_r = self.trajectory_set[:, id_s:id_e]

        return x_r

    def get_constant_u_sp(self, npoints):
        """
        Generate constant velocity input for the system.

        :param npoints: number of trajectory points
        :type npoints: int
        :return: constant velocity input
        :rtype: np.array
        """
        u_r = np.repeat(self.constant_v_tracking, npoints, axis=1)
        return u_r

    def set_trajectory(self, length, start):
        """
        Set trajectory type to be followed

        """

        self.trajectory_time = length
        self.trajectory_start_point = start

    def set_system_constants(self, l1=None, l2=None):
        """
        Helper method to set the constants for the barriers h1 and h2
        """

        # Barrier properties - position
        self.dt_p = 0.01
        self.eps_p = 1.32
        self.lambda_1 = 1.0075
        self.rah_1 = 0.2397

        # Barrier properties - attitude
        self.dt_t = 0.01
        self.eps_t = 0.4338
        self.lambda_2 = 1.6249
        self.rah_2 = 0.0650

        # Get minimum dt
        self.dt = min(self.dt_p, self.dt_t)

    def set_barrier_functions(self):
        """
        Helper method to set the desired barrier functions.

        :param hp: position barrier, defaults to None
        :type hp: ca.MX, optional
        :param hpdt: time-derivative of hp, defaults to None
        :type hpdt: ca.MX, optional
        :param hq: attitude barrier, defaults to None
        :type hq: ca.MX, optional
        :param hqdt: time-derivative of hq, defaults to None
        :type hqdt: ca.MX, optional
        """

        # Paper Translation barrier
        u = ca.MX.sym("u", self.n, 1)
        u1 = u[0:3]
        u2 = u[3:]

        # Setup position barrier
        p = ca.MX.sym("p", 3, 1)
        pr = ca.MX.sym("pr", 3, 1)

        h1 = self.eps_p**2 - ca.norm_2(p - pr)**2
        h1_ineq = - 2 * (p - pr).T @ u1 + self.lambda_1 * h1 - self.rah_1

        self.h1 = ca.Function('h1', [p, pr], [h1], self.fun_options)
        self.h1_ineq = ca.Function('h1_ineq', [p, pr, u], [h1_ineq], self.fun_options)

        # Setup attitude barrier
        t = ca.MX.sym("t", 3, 1)
        tr = ca.MX.sym("tr", 3, 1)

        h2 = self.eps_t**2 - ca.norm_2(t - tr)**2
        h2_ineq = - 2 * (t - tr).T @ self.psi(t) @ u2 + self.lambda_2 * h2 - self.rah_2

        self.h2 = ca.Function('h2', [t, tr], [h2], self.fun_options)
        self.h2_ineq = ca.Function('h2_ineq', [t, tr, u], [h2_ineq], self.fun_options)

    def get_barrier_value(self, x_t, x_r, u_t):
        """
        Helper function to get the barrier function values.

        :param x_t: system state
        :type x_t: np.array or ca.MX
        :param x_r: reference state
        :type x_r: np.array or ca.MX
        :param u_t: system control input
        :type u_t: np.array or ca.MX
        :return: values for the position and attitude barrier conditions
        :rtype: float
        """

        p = x_t[0:3]
        pr = x_r[0:3]
        t = x_t[3:]
        tr = x_r[3:]
        u = u_t

        # hp_ineq >= 0
        h1_ineq = self.h1_ineq(p, pr, u)

        # hq_ineq >= 0
        h2_ineq = self.h2_ineq(t, tr, u)

        return h1_ineq, h2_ineq

    def get_barrier_error_epsilon(self, x_t, x_r):
        """
        Helper function to get the barrier function values.

        :param x_t: system state
        :type x_t: np.array or ca.MX
        :param x_r: reference state
        :type x_r: np.array or ca.MX
        :return: values for the position and attitude barrier conditions
        :rtype: float
        """

        p = x_t[0:3]
        t = x_t[3:]

        pr = x_r[0:3]
        tr = x_r[3:]

        #    self.hp(p, pr, v, vr) - self.eps_p - self.eps_v
        e1 = self.eps_p**2 - ca.norm_2(p - pr)**2
        #    self.hq(q, qr, w, wr) - self.eps_q - self.eps_w
        e2 = self.eps_t**2 - ca.norm_2(t - tr)**2

        return e1, e2
