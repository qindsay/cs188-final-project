import numpy as np
from scipy.interpolate import interp1d

class CanonicalSystem:
    """
    Skeleton of the discrete canonical dynamical system.
    """
    def __init__(self, dt: float, ax: float = 1.0):
        """
        Args:
            dt (float): Timestep duration.
            ax (float): Gain on the canonical decay.
        """
        # Initialize time parameters
        self.dt: float = dt
        self.ax: float = ax
        self.run_time: float = 1.0  
        self.timesteps: int = int(self.run_time/dt)
        self.x: float = None  # phase variable

    def reset(self) -> None:
        """
        Reset the phase variable to its initial value.
        """
        self.x = 1.0

    def step(self, tau: float = 1.0, error_coupling: float = 1.0) -> float:
        """
        Advance the phase by one timestep.

        Returns:
            float: Updated phase value.
        """
        # TODO: implement update rule
        self.x = self.x + (-1 * self.ax * self.x * error_coupling) * self.dt * tau
        # print(self.x)
        return self.x

    def rollout(self, tau: float = 1.0, ec: float = 1.0) -> np.ndarray:
        """
        Generate the entire phase sequence.

        Returns:
            np.ndarray: Array of phase values over time.
        """
        # TODO: call reset() then repeatedly call step()
        out = np.zeros(self.timesteps)
        self.reset()
        for i in range(self.timesteps):
          out[i] = self.step(tau, ec)
        return out

class DMP:
    """
    Skeleton of the discrete Dynamic Motor Primitive.
    """
    def __init__(
        self,
        n_dmps: int,
        n_bfs: int,
        dt: float = 0.01,
        y0: float = 0.0,
        goal: float = 1.0,
        ay: float = 25.0,
        by: float = 6.25
    ):
        """
        Args:
            n_dmps (int): Number of dimensions.
            n_bfs (int): Number of basis functions per dimension.
            dt (float): Timestep duration.
            y0 (float|array): Initial state.
            goal (float|array): Goal state.
            ay (float|array): Attractor gain.
            by (float|array): Damping gain.
        """
        # TODO: initialize parameters
        self.n_dmps: int = n_dmps
        self.n_bfs: int = n_bfs
        self.dt: float = dt
        self.y0: np.ndarray = np.full(n_dmps, y0)
        self.goal: np.ndarray = np.full(n_dmps, goal)
        self.ay: np.ndarray = np.full(n_dmps, ay)
        self.by: np.ndarray = np.full(n_dmps, ay/4 if by is None else by)
        self.w: np.ndarray = np.zeros((n_dmps, n_bfs))  # weights
        self.cs: CanonicalSystem = CanonicalSystem(dt=self.dt, ax=float(self.ay[0]/self.by[0]))
        self.reset_state()
        self.centers, self.widths = None, None
    
    def reset_state(self) -> None:
        """
        Reset trajectories and canonical system state.
        """
        # TODO: reset y, dy, ddy and call self.cs.reset()
        self.y = self.y0.copy()
        self.dy = np.zeros(self.n_dmps)
        self.ddy = np.zeros(self.n_dmps)
        self.cs.reset()
    
    def __psi(self, x):
        return np.exp(-self.widths * (x[:,None] - self.centers)**2)

    def imitate(self, y_des: np.ndarray) -> np.ndarray:
        """
        Learn DMP weights from a demonstration.

        Args:
            y_des (np.ndarray): Desired trajectory, shape (D, T).

        Returns:
            np.ndarray: Interpolated demonstration (D x T').
        """
        # TODO: interpolate, compute forcing term, solve for w  
        if y_des.ndim == 1:
            y_des = y_des[None, :]
        D, T = y_des.shape
       
        interp = interp1d(np.linspace(0, 1, T), y_des, kind='cubic', axis=1)
        y_demo = interp(np.linspace(0, 1, self.cs.timesteps))   # (D,T')

        y = y_demo

        dy  = np.gradient(y,  self.dt, axis=1)
        ddy = np.gradient(dy, self.dt, axis=1)

        self.y0  = y[:, 0]
        self.goal = y[:, -1]

        self.centers = np.exp(-self.cs.ax * np.linspace(0, 1, self.n_bfs))
        self.widths  = self.n_bfs**1.5 / (self.centers * self.cs.ax)

        f_target = ddy - self.ay[:, None] * (
            self.by[:, None] * (self.goal[:, None] - y) - dy
        )

        # x_track = self.cs.rollout()
        # psi_raw = self.__psi(x_track)                     # (T', n_bfs)
        # psi = psi_raw / (psi_raw.sum(axis=1, keepdims=True) + 1e-10)
        # eps = 1e-6
        # self.w = np.vstack([
        #     np.linalg.lstsq(psi * x_track[:, None], f_target[d],
        #                     rcond=None)[0] /
        #     (self.goal[d] - self.y0[d] if
        #      abs(self.goal[d] - self.y0[d]) > eps else 1.0)
        #     for d in range(self.n_dmps)
        # ])

        return f_target, self.y0, self.goal
    
    def rollout(
        self,
        tau: float = 1.0,
        error: float = 0.0, 
        new_goal: np.ndarray = None
    ) -> np.ndarray:
        """
        Generate a new trajectory from the DMP.

        Args:
            tau (float): Temporal scaling.
            error (float): Feedback coupling.
            new_goal (np.ndarray, optional): Override goal.

        Returns:
            np.ndarray: Generated trajectory (T x D).
        """    
        if new_goal is not None:
            self.goal = np.atleast_1d(new_goal)
        
        self.reset_state()
        y_track = np.zeros((self.n_dmps, self.cs.timesteps))
        
        for t in range(self.cs.timesteps):
            phase = self.cs.step(tau=tau, error_coupling=1/(1+error))
            psi = np.exp(-self.widths * (phase - self.centers)**2) #does this ever get initialized correctly?
            
            for d in range(self.n_dmps):            
                
                f = (np.dot(psi, self.w[d]) * phase) / (psi.sum() + 1e-10)
                
                k = self.goal[d] - self.y0[d]
                self.ddy[d] = (
                    self.ay[d] * (self.by[d] * (self.goal[d] - self.y[d]) - self.dy[d]) + k * f
                )
                self.dy[d] += self.ddy[d] * self.dt * tau * (1.0 / (1.0 + error))
                self.y[d] += self.dy[d] * self.dt * tau * (1.0 / (1.0 + error))
            y_track[:, t] = self.y
            
        return y_track.T

    def rollout_adapted(
            self,
            weights, widths, centers, y0, tau: float = 1.0,
            error: float = 0.0, 
            new_goal: np.ndarray = None
        ) -> np.ndarray:
            """
            Generate a new trajectory from the DMP.

            Args:
                tau (float): Temporal scaling.
                error (float): Feedback coupling.
                new_goal (np.ndarray, optional): Override goal.

            Returns:
                np.ndarray: Generated trajectory (T x D).
            """    
            if new_goal is not None:
                self.goal = np.atleast_1d(new_goal)
            
            self.reset_state() #what is y0?
            y_track = np.zeros((self.n_dmps, self.cs.timesteps))
            
            for t in range(self.cs.timesteps):
                phase = self.cs.step(tau=tau, error_coupling=1/(1+error))
                psi = np.exp(-widths * (phase - centers)**2) #does this ever get initialized correctly?
                
                for d in range(self.n_dmps):            
            
                    f = (np.dot(psi, weights[d]) * phase) / (psi.sum() + 1e-10)
                    k = self.goal[d] - y0[d]
            
                    self.ddy[d] = (
                        self.ay[d] * (self.by[d] * (self.goal[d] - self.y[d]) - self.dy[d]) + k * f
                    )
                    self.dy[d] += self.ddy[d] * self.dt * tau * (1.0 / (1.0 + error))
                    self.y[d] += self.dy[d] * self.dt * tau * (1.0 / (1.0 + error))
                y_track[:, t] = self.y
                
            return y_track.T

# ==============================
# DMP Unit test
# ==============================
if __name__ == '__main__':

    import matplotlib.pyplot as plt

    # Test canonical system
    cs = CanonicalSystem(dt=0.05)
    x_track = cs.rollout()
    plt.figure()
    plt.plot(x_track, label='Canonical x')
    plt.title('Canonical System Rollout')
    plt.xlabel('Timestep')
    plt.ylabel('x')
    plt.legend()

    # Test DMP behavior with a sine-wave trajectory
    dt = 0.01
    T = 1.0
    t = np.arange(0, T, dt)
    y_des = np.sin(2 * np.pi * 2 * t)

    dmp = DMP(n_dmps=1, n_bfs=50, dt=dt)
    y_interp = dmp.imitate(y_des)
    y_run = dmp.rollout()

    plt.figure()
    plt.plot(t, y_des, 'k--', label='Original')
    plt.plot(np.linspace(0, T, y_interp.shape[1]), y_interp.flatten(), 'b-.', label='Interpolated')
    plt.plot(np.linspace(0, T, y_run.shape[0]), y_run.flatten(), 'r-', label='DMP Rollout')
    plt.title('DMP Imitation and Rollout')
    plt.xlabel('Time (s)')
    plt.ylabel('y')
    plt.legend()
    plt.tight_layout()
    plt.show()