import numpy as np


class DynamicsHandler:
    """Base class for dynamic model handlers."""
    def get_rates(self, omega):
        raise NotImplementedError("Subclasses must implement this method.")

class ConstantOmegaDynamics(DynamicsHandler):
    """
    Default dynamics handler that keeps angular velocity constant.
    This ensures backward compatibility with previous versions.
    """
    def get_rates(self, omega):
        return np.zeros(3) # Angular acceleration is zero

class EulerRigidBodyDynamics(DynamicsHandler):
    """
    Calculates angular acceleration for a rigid body using Euler's equations.
    """
    def __init__(self, inertia_tensor):
        self.I = np.asarray(inertia_tensor)
        self.I_inv = np.linalg.inv(self.I)
    
    def get_rates(self, omega):
        """
        Calculates omega_dot from the equation:
        omega_dot = -I_inv * (omega x (I * omega))
        """
        omega = np.asarray(omega)
        omega_dot = -self.I_inv @ (np.cross(omega, self.I @ omega))
        return omega_dot

    
class NumericalSolver:
    """
    A base class for numerical integration of first-order ODEs.
    Now uses a dynamics handler to update the full state.
    """
    def __init__(self, initial_conditions, t_initial, t_final, dt, dynamics_handler=None):
        self.initial_conditions = np.asarray(initial_conditions)
        self.t_initial = t_initial
        self.t_final = t_final
        self.dt = dt

        # Default to constant angular velocity if no handler is provided
        self.dynamics_handler = dynamics_handler if dynamics_handler is not None else ConstantOmegaDynamics()

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
    A numerical solver for 3-2-1 Euler angle kinematics combined with a dynamic model.
    The state vector is [phi, theta, psi, w1, w2, w3].
    """

    def _step(self):
        # Unpack the 6-element state vector
        kinematic_state = self.current_state[:3]
        dynamic_state = self.current_state[3:]
        
        phi, theta, psi = kinematic_state
        omega = dynamic_state
        w1, w2, w3 = omega
        
        # --- 1. Calculate Kinematic Rates ---
        cos_theta = np.cos(theta)
        if np.isclose(cos_theta, 0.0):
            kinematic_rates = np.zeros(3)
        else:
            sin_phi = np.sin(phi)
            cos_phi = np.cos(phi)
            sin_theta = np.sin(theta)
            psi_dot = (w2 * sin_phi + w3 * cos_phi) / cos_theta
            theta_dot = w2 * cos_phi - w3 * sin_phi
            phi_dot = w1 + (w2 * sin_phi * sin_theta + w3 * cos_phi * sin_theta) / cos_theta
            kinematic_rates = np.array([phi_dot, theta_dot, psi_dot])
        
        # --- 2. Calculate Dynamic Rates ---
        dynamic_rates = self.dynamics_handler.get_rates(omega)
        
        # --- 3. Combine and return full state rates ---
        return np.concatenate((kinematic_rates, dynamic_rates))