from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib import rc
try:
    mpl.use('TkAgg')
except ImportError:
    print("Running with no-GUI backend.")


class EmbeddedSimEnvironment(object):
    def __init__(self, model, dynamics, ctl_class, controller,
                 noise=None, time=100.0, collect=True, animate=False):
        """
        Embedded simulation environment. Simulates the syste given
        dynamics and a control law, plots in matplotlib.

        :param model: model object
        :type model: object
        :param dynamics: system dynamics function (x, u)
        :type dynamics: casadi.DM
        :param controller: controller function (x, r)
        :type controller: casadi.DM
        :param time: total simulation time, defaults to 100 seconds
        :type time: float, optional
        :param noise: maximum noise added to the system
        :type noise: np.array 13x1
        :param collect: collect data for the paper flag
        :type collect: boolean
        :param animate: animate the simulation
        :type animate:boolean
        """
        self.model = model
        self.Nx = model.n
        self.Nu = model.m
        self.dynamics = dynamics
        self.ctl_class = ctl_class
        self.controller = controller
        self.total_sim_time = time  # seconds
        self.dt = self.model.dt
        self.noise = noise
        self.estimation_in_the_loop = False
        self.using_trajectory_ref = False
        self.animate = animate
        self.collect = collect

        # Plotting definitions
        self.plt_window = float("inf")  # running plot window [s]/float("inf")
        self.sim_loop_length = int(self.total_sim_time / self.dt) + 1
        plt.style.use('ggplot')

    def run(self, x0):
        """
        Run simulator with specified system dynamics and control function.
        """
        x_init = np.array([x0]).T
        data = {}
        t = np.array([0])
        y_vec = x_init
        ref_vec = np.empty((x_init.shape[0], 0))
        u_vec = np.array([[0, 0, 0, 0, 0, 0]]).T
        slv_time = np.empty((0, 1))
        h1_h2_vec = np.empty((2, 0))
        h1_h2_e_vec = np.empty((2, 0))

        # Start figure
        if self.animate is True:
            self.prepare_animated_plots()

        for i in range(self.sim_loop_length):
            # Print iteration info:
            print("Iteration: ", i, " / ", self.sim_loop_length, "    t: ", round(i * self.dt, 2))
            x = np.array([y_vec[:, -1]]).T

            # Get control input and obtain next state
            u, ref, pred_x, pred_ref = self.controller(x, i * self.dt)

            # Convert data to numpy array and collect barrier values
            u = np.asarray(u).reshape(6, 1)

            hp_c, hq_c = self.model.get_barrier_value(x.reshape(self.Nx, 1),
                                                      ref.reshape(self.Nx, 1),
                                                      u.reshape(self.Nu, 1))
            hp, hq = self.model.get_barrier_error_epsilon(x.reshape(self.Nx, 1),
                                                          ref.reshape(self.Nx, 1))
            # hp_c, hq_c = 0, 0
            # hp, hq = 0, 0

            # Prepare data for logging
            slv_time = np.append(slv_time,
                                 self.ctl_class.get_last_solve_time())
            ref_vec = np.append(ref_vec, np.array([ref]).T, axis=1)
            h1_h2_vec = np.append(h1_h2_vec,
                                  np.array([[hp_c, hq_c]]).reshape(2, 1),
                                  axis=1)
            h1_h2_e_vec = np.append(h1_h2_e_vec,
                                    np.array([[hp, hq]]).reshape(2, 1),
                                    axis=1)

            # Log in data_structure
            data['pred_ref'] = pred_ref
            data['pred_x'] = pred_x
            data['y_vec'] = y_vec
            data['u_vec'] = u_vec
            data['ref_vec'] = ref_vec
            data['h1_h2_vec'] = h1_h2_vec
            data['h1_h2_e_vec'] = h1_h2_e_vec
            data['t'] = t

            # Plot if plotting is required
            if self.animate is True:
                self.animated_plot(data, i)

            # Propagate state
            x_next = self.dynamics(x, u) + self.get_noise()

            # Store data
            t = np.append(t, i * self.dt)
            y_vec = np.append(y_vec, np.array(x_next), axis=1)
            u_vec = np.append(u_vec, np.array(u), axis=1)

        if self.collect is True:
            self.collect_plots(data)

        if self.animate is True or self.collect is True:
            plt.show(block=True)

        return t, y_vec, u_vec, np.average(slv_time)

    def get_noise(self):
        """
        Generate system noise and return the noise vector.

        :return: noise vector
        :rtype: np.array 13x1
        """

        if self.noise is not None:
            w_p = np.random.uniform(-self.noise["pos"] / 2.0,
                                    self.noise["pos"] / 2.0,
                                    (3, 1)) * self.dt
            w_t = np.random.uniform(-self.noise["att"] / 2.0,
                                    self.noise["att"] / 2.0,
                                    (3, 1)) * self.dt
            noise_vec = np.concatenate((w_p, w_t), axis=0)
        else:
            noise_vec = np.zeros((self.Nx, 1))

        return noise_vec

    def animated_plot(self, data, i):
        """
        Helper function to plot animated data.

        :param data: [description]
        :type data: [type]
        """

        # Extract data from argument structure
        pred_ref = data['pred_ref']
        pred_x = data['pred_x']
        y_vec = data['y_vec']
        u_vec = data['u_vec']
        ref_vec = data['ref_vec']
        h1_h2_vec = data['h1_h2_vec']
        t = data['t']

        # Get plot window values
        if self.plt_window != float("inf"):
            l_wnd = 0 if int(i + 1 - self.plt_window / self.dt) < 1 \
                else int(i + 1 - self.plt_window / self.dt)
        else:
            l_wnd = 0

        # Plot X-Y plane
        self.ax6.clear()
        self.ax6.set_title("3D Trajectory")
        x_pred = np.zeros(pred_ref.shape)
        for k in range(pred_ref.shape[1]):
            x_pred[:, k] = np.asarray(pred_x[k]).reshape(self.Nx,)

        self.ax6.plot(y_vec[0, l_wnd:-1], y_vec[1, l_wnd:-1],
                      y_vec[2, l_wnd:-1], color="r")
        self.ax6.plot(x_pred[0, :], x_pred[1, :], x_pred[2, :], color="r",
                      linestyle='-', marker='o')
        self.ax6.plot(pred_ref[0, :], pred_ref[1, :], pred_ref[2, :],
                      color="b", linestyle='--', marker='x')
        self.ax6.legend(["Past Trajectory",
                         "Predicted Trajectory",
                         "Reference"])
        self.ax6.set_xlabel("X [m]")
        self.ax6.set_ylabel("Y [m]")
        self.ax6.grid()

        # Plot barrier values
        self.ax7.clear()
        self.ax8.clear()
        self.ax7.set_title("Position and Attitude Barrier Values")
        self.ax7.plot(t[l_wnd:-1], h1_h2_vec[0, l_wnd:-1])
        self.ax8.plot(t[l_wnd:-1], h1_h2_vec[1, l_wnd:-1])
        self.ax7.legend(["Position Barrier"])
        self.ax8.legend(["Attitude Barrier"])
        self.ax8.set_xlabel("Time [s]")
        self.ax7.grid()
        self.ax8.grid()

        # Plot state info
        self.ax1.clear()
        self.ax1.set_title("Astrobee Testing")
        self.ax1.plot(t[l_wnd:-1], y_vec[0, l_wnd:-1] - ref_vec[0, l_wnd:-1],
                      t[l_wnd:-1], y_vec[1, l_wnd:-1] - ref_vec[1, l_wnd:-1],
                      t[l_wnd:-1], y_vec[2, l_wnd:-1] - ref_vec[2, l_wnd:-1])
        self.ax1.legend(["x", "y", "z"])
        self.ax1.set_ylabel("Error [m]")
        self.ax1.grid()

        self.ax2.clear()
        self.ax2.plot(t[l_wnd:-1], y_vec[3, l_wnd:-1] - ref_vec[3, l_wnd:-1],
                      t[l_wnd:-1], y_vec[4, l_wnd:-1] - ref_vec[4, l_wnd:-1],
                      t[l_wnd:-1], y_vec[5, l_wnd:-1] - ref_vec[5, l_wnd:-1])
        self.ax2.legend(["x", "y", "z"])
        self.ax2.set_ylabel("Attitude Error [rad]")
        self.ax2.grid()

        # Plot controls
        self.ax3.clear()
        self.ax3.plot(t[l_wnd:-1], u_vec[0, l_wnd:-1],
                      t[l_wnd:-1], u_vec[1, l_wnd:-1],
                      t[l_wnd:-1], u_vec[2, l_wnd:-1])
        self.ax3.set_xlabel("Time [s]")
        self.ax3.set_ylabel("U1 [m/s]")
        self.ax3.grid()

        self.ax4.clear()
        self.ax4.plot(t[l_wnd:-1], u_vec[3, l_wnd:-1],
                      t[l_wnd:-1], u_vec[4, l_wnd:-1],
                      t[l_wnd:-1], u_vec[5, l_wnd:-1])
        self.ax4.set_xlabel("Time [s]")
        self.ax4.set_ylabel("U2 [rad/s]")
        self.ax4.set_ylim(-0.06, 0.06)
        self.ax4.grid()

        plt.pause(0.001)

    def collect_plots(self, data):
        """
        Collect plots for paper.

        :param data: data dictionary for plots
        :type data: dict
        """

        # Extract data from argument structure
        y_vec = data['y_vec']
        u_vec = data['u_vec']
        ref_vec = data['ref_vec']
        h1_h2_e_vec = data['h1_h2_e_vec']
        t = data['t']

        # Plot properties
        plt.style.use('default')
        mpl.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}", r"\DeclareOldFontCommand{\rm}{\normalfont\rmfamily}{\mathrm}"]
        rc('text', usetex=True)
        rc('font', **{'family': 'serif',
                      'size': 22})
        rc('lines', linewidth=4)
        rc('legend', **{'fontsize': 14,
                        'handlelength': 2})
        rc('axes', titlesize='medium')

        # Get desired error bounds
        eps_p = self.model.eps_p

        eps_t = self.model.eps_t

        max_v = self.model.uub[0]  # maximum velocity allowed (same on all axis)
        max_w = self.model.uub[3]  # maximum angular velocity allowed (Same on all axis)

        # ---------------------------------------------------
        #      Third Figure with Compact Representation
        # ---------------------------------------------------

        # Create figures
        scale = 3.0
        fig1 = plt.figure(figsize=(3.5 * scale, 4.5 * scale), dpi=80)

        ax1 = fig1.add_subplot(321)  # Pos error
        ax2 = fig1.add_subplot(322)  # Att error

        ax3 = fig1.add_subplot(323)  # Pos barrier
        ax4 = fig1.add_subplot(324)  # Att barrier

        ax5 = fig1.add_subplot(325)  # U1
        ax6 = fig1.add_subplot(326)  # U2

        # Position error
        ax1.clear()
        ax1.plot(t, np.linalg.norm(y_vec[0:3, :] - ref_vec[0:3, :], axis=0))
        ax1.plot(t, np.ones((y_vec.shape[1],)) * (eps_p),
                 t, np.zeros((y_vec.shape[1],)),
                 color='k', linestyle='--')
        # ax1.legend(["X", "Y", "Z"])
        ax1.legend([r"$\| \tilde{e}_p \|$", r"$\varepsilon_p$"], loc="upper right")
        ax1.set_ylabel("{Norm Position Error [m]}")
        ax1.set_ylim(-0.3, eps_p * 1.3)
        ax1.grid()

        # Attitude error
        ax2.clear()
        ax2.plot(t, np.linalg.norm(y_vec[3:6, :] - ref_vec[3:6, :], axis=0))  # ,
        #         t, y_vec[4, :] - ref_vec[4, :],
        #         t, y_vec[5, :] - ref_vec[5, :])
        ax2.plot(t, np.ones((y_vec.shape[1],)) * (eps_t),
                 t, np.zeros((y_vec.shape[1],)),
                 color='k', linestyle='--')
        # ax2.legend([r"$\phi$", r"$\varphi$", r"$\psi$"])
        ax2.legend([r"$\| \tilde{e}_\theta \|$", r"$\varepsilon_\theta$"], loc="upper right")
        ax2.set_ylabel("{Norm Attitude Error [rad]}")
        ax2.yaxis.set_label_position("right")
        ax2.yaxis.tick_right()
        ax2.set_ylim(-0.1, eps_t * 1.35)
        ax2.grid()

        # Pos barrier
        ax3.clear()
        ax3.plot(t, h1_h2_e_vec[0, :])
        ax3.set_ylabel("{$h_1$ barrier value}")
        ax3.set_ylim(-0.3, eps_p**2 + 0.3)
        ax3.grid()

        # Att barrier
        ax4.clear()
        ax4.plot(t, h1_h2_e_vec[1, :])
        ax4.set_ylabel("{$h_2$ barrier value}")
        ax4.yaxis.set_label_position("right")
        ax4.yaxis.tick_right()
        ax4.set_ylim(-0.3, eps_t**2 + 0.3)
        ax4.grid()

        # Plot linear velocity inputs
        ax5.clear()
        ax5.plot(t, u_vec[0, :],
                 t, u_vec[1, :],
                 t, u_vec[2, :])
        ax5.plot(t, np.ones((u_vec.shape[1],)) * max_v,
                 t, -np.ones((u_vec.shape[1],)) * max_v,
                 color='k', linestyle='--')
        ax5.set_ylabel("{Linear Velocity - $u_1$ [m/s]}")
        ax5.legend(["X", "Y", "Z"])
        ax5.set_ylim(-max_v * 1.3, max_v * 1.3)
        ax5.set_xlabel("Time [s]")
        ax5.grid()

        # Plot angular velocity input
        ax6.clear()
        ax6.plot(t, u_vec[3, :],
                 t, u_vec[4, :],
                 t, u_vec[5, :])
        ax6.plot(t, np.ones((u_vec.shape[1],)) * max_w,
                 t, -np.ones((u_vec.shape[1],)) * max_w,
                 color='k', linestyle='--')
        ax6.set_ylabel("{Angular Velocity - $u_2$ [rad/s]}")
        ax6.yaxis.set_label_position("right")
        ax6.yaxis.tick_right()
        ax6.legend(["X", "Y", "Z"])
        ax6.set_ylim(-max_w * 1.3, max_w * 1.3)
        ax6.set_xlabel("Time [s]")
        ax6.grid()

        fig1.tight_layout()

    def prepare_animated_plots(self):
        """
        Helper function to create plot figures
        """

        # F1: state, F2: 3D trajectoty, F3: Barrier value
        self.fig1, (self.ax1, self.ax2, self.ax3, self.ax4) = plt.subplots(4)
        self.fig2 = plt.figure()
        self.ax6 = self.fig2.add_subplot(111, projection='3d')
        self.fig3 = plt.figure()
        self.ax7 = self.fig3.add_subplot(211)
        self.ax8 = self.fig3.add_subplot(212)
        plt.ion()

    def set_window(self, window):
        """
        Set the plot window length, in seconds.

        :param window: window length [s]
        :type window: float
        """
        self.plt_window = window

    def use_trajectory_control(self, value):
        """
        Helper function to set trajectory tracking controller.
        """
        self.using_trajectory_ref = value
