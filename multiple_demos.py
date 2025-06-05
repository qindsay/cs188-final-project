import numpy as np
import scipy
from lindsay_dmp_copy import DMP, CanonicalSystem
import copy


#function that takes in all demo data and splits into 3 sections (move to nut, grab nut, put on peg). returns 3 splits
def split_demos(self, demos):
    pass
    
#function that takes in segmented demo data and interpolates, return as a list of segments
def interpolate_demos(self, demos):
    pass

#function that 

def imitate_path(self, x_des, dx_des = None, ddx_des = None, t_des = None,
        g_w = True, **kwargs):
        '''
        Takes in a desired trajectory and generates the set of system
        parameters that best realize this path.
        x_des array shaped num_timesteps x n_dmps
        t_des 1D array of num_timesteps component
        g_w boolean, used to separate the one-shot learning from the
                    regression over multiple demonstrations
        '''

        ## Set initial state and x_goal
        self.x_0 = x_des[0].copy()
        self.x_goal = x_des[-1].copy()

        ## Set t_span
        if t_des is None:
            # Default value for t_des
            t_des = np.linspace(0, self.cs.run_time, x_des.shape[0])
        else:
            # Warp time to start from zero and end up to T
            t_des -= t_des[0]
            t_des /= t_des[-1]
            t_des *= self.cs.run_time
        time = np.linspace(0., self.cs.run_time, self.cs.timesteps)

        ## Piecewise linear interpolation
        # Interpolation function
        path_gen = scipy.interpolate.interp1d(t_des, x_des.transpose())
        # Evaluation of the interpolant
        path = path_gen(time)
        x_des = path.transpose()

        ## Second order estimates of the derivatives
        ## (the last non centered, all the others centered)
        # if dx_des is None:
            # D1 = compute_D1(self.cs.timesteps, self.cs.dt)
            # dx_des = np.dot(D1, x_des)
        # else:
            # dpath = np.zeros([self.cs.timesteps, self.n_dmps])
            # dpath_gen = scipy.interpolate.interp1d(t_des, dx_des)
            # dpath = dpath_gen(time)
            # dx_des = dpath.transpose()
        # if ddx_des is None:
        #     D2 = compute_D2(self.cs.timesteps, self.cs.dt)
        #     ddx_des = np.dot(D2, x_des)
        # else:
        #     ddpath = np.zeros([self.cs.timesteps, self.n_dmps])
        #     ddpath_gen = scipy.interpolate.interp1d(t_des, ddx_des)
        #     ddpath = ddpath_gen(time)
        #     ddx_des = ddpath.transpose()
        dx_des = np.gradient(x_des,  self.dt, axis=1)
        ddx_des = np.gradient(dx_des, self.dt, axis=1)


        ## Find the force required to move along this trajectory
        s_track = self.cs.rollout()
        f_target = ((ddx_des / self.K - (self.x_goal - x_des) + 
            self.D / self.K * dx_des).transpose() +
            np.reshape((self.x_goal - self.x_0), [self.n_dmps, 1]) * s_track)
        if g_w:
            # Efficiently generate weights to realize f_target
            # (only if not called by paths_regression)
            self.gen_weights(f_target)
            self.reset_state()
            self.learned_position = self.x_goal - self.x_0
        return f_target

def gen_psi(x, n_bfs, cs_ax):
    '''
    Generates the activity of the basis functions for a given canonical
    system rollout.
        s : array containing the rollout of the canonical system
    '''
    centers = np.exp(-cs_ax * np.linspace(0, 1, n_bfs))
    widths  = n_bfs**1.5 / (centers * cs_ax)
    
    # c = np.reshape(centers, [n_bfs + 1, 1])
    # w = np.reshape(widths, [n_bfs + 1,1 ])
    # xi = w * (s - c) * (s - c)
    # psi_set = np.exp(- xi)
    # psi_set = np.nan_to_num(psi_set)
    # return psi_set

    return np.exp(-widths * (x[:,None] - centers)**2).T

    
def paths_regression(traj_set, n_dmps, n_bfs, dt, t_set = None):
    '''
    Takes in a set (list) of desired trajectories (with possibly the
    execution times) and generate the weight which realize the best
    approximation.
        each element of traj_set should be shaped num_timesteps x n_dim
        trajectories
    '''
    cs = CanonicalSystem(dt=dt)

    ## Step 1: Generate the set of the forcing terms
    f_set = np.zeros([len(traj_set), n_dmps, cs.timesteps])
    # g_new = np.ones(n_dmps)
    for it in range(len(traj_set)):
        traj = traj_set[it].T
        # if t_set is None:
        #     t_des_tmp = None
        # else:
        #     t_des_tmp = t_set[it]
        #     t_des_tmp -= t_des_tmp[0]
        #     t_des_tmp /= t_des_tmp[-1]
        #     t_des_tmp *= self.cs.run_time

        # Alignment of the trajectory so that
        # x_0 = [0; 0; ...; 0] and g = [1; 1; ...; 1].
        # x_des_tmp = copy.deepcopy(traj_set[it])
        # x_des_tmp -= x_des_tmp[0] # translation to x_0 = 0
        # g_old = x_des_tmp[-1] # original x_goal position
        # R = roto_dilatation(g_old, g_new) # rotodilatation

        # Rescaled and rotated trajectory
        # x_des_tmp = np.dot(x_des_tmp, np.transpose(R))

        # Learning of the forcing term for the particular trajectory
        tempDMP = DMP(n_dmps, n_bfs, dt)
        # f_tmp = self.imitate_path(x_des = x_des_tmp, t_des = t_des_tmp,
        #     g_w = False, add_force = None)        
        f_tmp = tempDMP.imitate(traj)
        f_set[it, :, :] = f_tmp.copy() # add the new forcing term to the set

    ## Step 2: Learning of the weights using linear regression
    w = np.zeros([n_dmps, n_bfs])
    s_track = cs.rollout()
    psi_set = gen_psi(s_track, n_bfs, cs.ax) 
    psi_sum = np.sum(psi_set, 0)
    psi_sum_2 = psi_sum * psi_sum
    s_track_2 = s_track * s_track
    
    A = np.zeros([n_bfs, n_bfs])
    for k in range(n_bfs):
        A[k, k] = scipy.integrate.simpson(
            psi_set[k, :] * psi_set[k, :] * s_track_2 / psi_sum_2, x=s_track)
        for h in range(k, n_bfs):
            A[h, k] = scipy.integrate.simpson(
                psi_set[k, :] * psi_set[h, :] * s_track_2 / psi_sum_2,
                x=s_track)
            A[k, h] = A[h, k].copy()
    A *= len(traj_set)
    LU = scipy.linalg.lu_factor(A)

    # The weights are learned dimension by dimension
    for d in range(n_dmps):
        f_d_set = f_set[:, d, :].copy()
        # Set up the minimization problem
        b = np.zeros([n_bfs])
        for k in range(n_bfs):
            
            b[k] = scipy.integrate.simpson(
                np.sum(f_d_set * psi_set[k, :] * s_track / psi_sum, 0),
                x=s_track)

        # Solve the minimization problem
        w[d, :] = scipy.linalg.lu_solve(LU, b)
    return w
    # self.learned_position = np.ones(self.n_dmps)

if __name__ == '__main__':
    
    import matplotlib.pyplot as plt

    # Test canonical system
    # cs = CanonicalSystem(dt=0.05)
    # x_track = cs.rollout()
    # plt.figure()
    # plt.plot(x_track, label='Canonical x')
    # plt.title('Canonical System Rollout')
    # plt.xlabel('Timestep')
    # plt.ylabel('x')
    # plt.legend()

    # Test DMP behavior with a sine-wave trajectory
    dt = 0.01
    T = 1.0
    t = np.arange(0, T, dt)
    y_des = np.sin(2 * np.pi * 2 * t)
    
    traj_set = []
    for _ in range(10):
        freq = np.random.uniform(1.5, 2.5)         # vary frequency a bit
        amp = np.random.uniform(0.8, 1.2)          # vary amplitude
        phase = np.random.uniform(-0.5, 0.5) * np.pi  # vary phase shift

        y_des = amp * np.sin(2 * np.pi * freq * t + phase)
        traj_set.append(y_des[:, None])  # shape (T, 1)

    traj_set = np.array(traj_set)  # shape (10, T, 1)

    weights = paths_regression(traj_set, n_dmps=1, n_bfs=50, dt=dt)

    # dmp = DMP(n_dmps=1, n_bfs=50, dt=dt)
    # y_interp = dmp.imitate(y_des)
    # y_run = dmp.rollout()

    # plt.figure()
    # plt.plot(t, y_des, 'k--', label='Original')
    # plt.plot(np.linspace(0, T, y_interp.shape[1]), y_interp.flatten(), 'b-.', label='Interpolated')
    # plt.plot(np.linspace(0, T, y_run.shape[0]), y_run.flatten(), 'r-', label='DMP Rollout')
    # plt.title('DMP Imitation and Rollout')
    # plt.xlabel('Time (s)')
    # plt.ylabel('y')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()