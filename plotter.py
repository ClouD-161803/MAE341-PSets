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
        plt.legend(loc='lower right')
        plt.show()