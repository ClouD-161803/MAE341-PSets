import numpy as np

class NumericalSolver:
    """
    A base class for numerical integration of first-order ODEs.
    
    This class handles the time setup and the core integration loop.
    Subclasses must implement the '_step' method which defines the
    specifics of the system's dynamics (the RHS of the ODE).
    """
    def __init__(self, initial_conditions, t_initial, t_final, dt):
        self.initial_conditions = np.asarray(initial_conditions)
        self.t_initial = t_initial
        self.t_final = t_final
        self.dt = dt

        self.num_steps = int((t_final - t_initial) / dt)
        self.time_points = np.linspace(t_initial, t_final, self.num_steps + 1)
        
        self.history = np.zeros((self.num_steps + 1, len(self.initial_conditions)))
        self.history[0, :] = self.initial_conditions
        self.current_state = self.initial_conditions.copy()

    def _step(self):
        """
        This method must be implemented by subclasses. It should calculate
        the rates of change of the state variables.
        """
        raise NotImplementedError("The '_step' method must be implemented by a subclass.")

    def solve(self):
        """
        Runs the numerical integration using the Forward Euler method.
        """
        for i in range(self.num_steps):
            rates = self._step()
            self.current_state += rates * self.dt
            self.history[i + 1, :] = self.current_state
        return self.time_points, self.history

class EulerAngleSolver(NumericalSolver):
    """
    A numerical solver for 3-2-1 Euler angle kinematics.
    """
    def __init__(self, initial_conditions, t_initial, t_final, dt, omega):
        super().__init__(initial_conditions, t_initial, t_final, dt)
        self.omega = np.asarray(omega) # [w1, w2, w3]

    def _step(self):
        """
        Calculates the rates of change for 3-2-1 Euler angles.
        The state vector is assumed to be [phi, theta, psi].
        """
        phi, theta, psi = self.current_state[0], self.current_state[1], self.current_state[2]
        w1, w2, w3 = self.omega[0], self.omega[1], self.omega[2]
        
        cos_theta = np.cos(theta)
        if np.isclose(cos_theta, 0.0):
            return np.zeros(3) # Avoid singularity at theta = +/- 90 deg

        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        sin_theta = np.sin(theta)

        psi_dot = (w2 * sin_phi + w3 * cos_phi) / cos_theta
        theta_dot = w2 * cos_phi - w3 * sin_phi
        phi_dot = w1 + (w2 * sin_phi * sin_theta + w3 * cos_phi * sin_theta) / cos_theta
        
        return np.array([phi_dot, theta_dot, psi_dot])