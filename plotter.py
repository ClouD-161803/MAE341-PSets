import numpy as np
import matplotlib.pyplot as plt

class SimulationPlotter:
    """
    Handles plotting of simulation results.
    """
    @staticmethod
    def plot_euler_angles(time_points, angles_history, dt):
        """
        Plots the evolution of Euler angles over time.
        """
        plt.figure(figsize=(12, 6))
        
        plt.plot(time_points, ((angles_history[:, 0] + np.pi) % (2 * np.pi)) - np.pi, label=r'$\phi(t)$ (Roll)')
        plt.plot(time_points, ((angles_history[:, 1] + np.pi) % (2 * np.pi)) - np.pi, label=r'$\theta(t)$ (Pitch)')
        plt.plot(time_points, ((angles_history[:, 2] + np.pi) % (2 * np.pi)) - np.pi, label=r'$\psi(t)$ (Yaw)')
        
        plt.title(fr'Euler Angle Evolution (3-2-1) with $\Delta t = {dt}$ s')
        plt.xlabel('Time (s)')
        plt.ylabel('Angle (rad) [-π, π)')
        plt.yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
                    [r'$-\pi$', r'$-\frac{\pi}{2}$', '0', r'$\frac{\pi}{2}$', r'$\pi$'])
        plt.grid(True)
        plt.legend(loc='lower right')
        plt.show()

    @staticmethod
    def plot_full_state(time_points, state_history, dt):
        """
        Plots the evolution of Euler angles and angular velocities.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Plot Euler Angles
        ax1.plot(time_points, ((state_history[:, 0] + np.pi) % (2 * np.pi)) - np.pi, label=r'$\phi(t)$ (Roll)')
        ax1.plot(time_points, ((state_history[:, 1] + np.pi) % (2 * np.pi)) - np.pi, label=r'$\theta(t)$ (Pitch)')
        ax1.plot(time_points, ((state_history[:, 2] + np.pi) % (2 * np.pi)) - np.pi, label=r'$\psi(t)$ (Yaw)')
        ax1.set_title(fr'Kinematic and Dynamic Evolution with $\Delta t = {dt}$ s')
        ax1.set_ylabel('Angle (rad) [-π, π)')
        ax1.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        ax1.set_yticklabels([r'$-\pi$', r'$-\frac{\pi}{2}$', '0', r'$\frac{\pi}{2}$', r'$\pi$'])
        ax1.grid(True)
        ax1.legend(loc='lower right')
        
        # Plot Angular Velocities
        ax2.plot(time_points, state_history[:, 3], label=r'$\omega_1(t)$')
        ax2.plot(time_points, state_history[:, 4], label=r'$\omega_2(t)$')
        ax2.plot(time_points, state_history[:, 5], label=r'$\omega_3(t)$')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Angular Velocity (rad/s)')
        ax2.grid(True)
        ax2.legend(loc='lower right')
        
        plt.tight_layout()
        plt.show()


class UnwrappedSimulationPlotter(SimulationPlotter):
    """
    Handles plotting of simulation results without modulo wrapping.
    """
    @staticmethod
    def plot_euler_angles(time_points, angles_history, dt):
        """
        Plots the evolution of Euler angles over time without wrapping.
        """
        plt.figure(figsize=(12, 6))
        
        plt.plot(time_points, np.unwrap(angles_history[:, 0]), label=r'$\phi(t)$ (Roll)')
        plt.plot(time_points, np.unwrap(angles_history[:, 1]), label=r'$\theta(t)$ (Pitch)')
        plt.plot(time_points, np.unwrap(angles_history[:, 2]), label=r'$\psi(t)$ (Yaw)')
        
        plt.title(fr'Euler Angle Evolution (3-2-1) with $\Delta t = {dt}$ s')
        plt.xlabel('Time (s)')
        plt.ylabel('Angle (rad)')
        plt.grid(True)
        plt.legend()
        plt.show()

    @staticmethod
    def plot_full_state(time_points, state_history, dt):
        """
        Plots the evolution of Euler angles and angular velocities without wrapping.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Plot Euler Angles
        ax1.plot(time_points, np.unwrap(state_history[:, 0]), label=r'$\phi(t)$ (Roll)')
        ax1.plot(time_points, np.unwrap(state_history[:, 1]), label=r'$\theta(t)$ (Pitch)')
        ax1.plot(time_points, np.unwrap(state_history[:, 2]), label=r'$\psi(t)$ (Yaw)')
        ax1.set_title(fr'Kinematic and Dynamic Evolution with $\Delta t = {dt}$ s')
        ax1.set_ylabel('Angle (rad)')
        ax1.grid(True)
        ax1.legend(loc='lower right')
        
        # Plot Angular Velocities
        ax2.plot(time_points, state_history[:, 3], label=r'$\omega_1(t)$')
        ax2.plot(time_points, state_history[:, 4], label=r'$\omega_2(t)$')
        ax2.plot(time_points, state_history[:, 5], label=r'$\omega_3(t)$')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Angular Velocity (rad/s)')
        ax2.grid(True)
        ax2.legend(loc='lower right')
        
        plt.tight_layout()
        plt.show()