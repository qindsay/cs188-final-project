import numpy as np
import scipy


def detect_grasp_segments(grasp_flags: np.ndarray) -> list:
    """
    Identify segments based on grasp toggles.

    Args:
        grasp_flags (np.ndarray): (T,1) array of grasp signals.

    Returns:
        List[Tuple[int,int]]: start and end indices per segment.
    """
    # TODO: implement boundary detection
    res = []
    start = 0
    cur = grasp_flags[0]
    length = len(grasp_flags)
    for i in range(length):
        if grasp_flags[i] != cur:
            res.append((start, i)) #consider not doing i-1 bc of offset in init
            start = i
            cur = grasp_flags[i]
    
    res.append((start, length))   
    return res

# function that takes in all demo data and splits into 3 sections (move to nut, grab nut, put on peg). returns 3 splits
def split_demos(demos):
    splits = {0: [], 1: [], 2: []}

    for key in demos:
        demo = demos[key]
        ee_pos = demo['obs_robot0_eef_pos']
        grasp_flags = demo['actions'][:, -1:].astype(int) 
        segments = detect_grasp_segments(grasp_flags)

        if len(segments) >= 3:
            for i in range(3):
                start, end = segments[i]
                splits[i].append(ee_pos[start:end])

    return splits

import numpy as np

# function that computes the average trajectory segments
def compute_avg_traj(segments):
    if not segments:
        raise ValueError("Empty segment list!")
    
    min_len = float('inf')
    for seg in segments:
        seg_len = len(seg)
        if seg_len < min_len:
            min_len = seg_len

    modified_segs = []
    for seg in segments:
        truncated_seg = seg[:min_len]
        modified_segs.append(truncated_seg)

    segs = np.stack(modified_segs, axis=0)
    avg_traj = np.mean(segs, axis=0)
    return avg_traj


def compute_avg_obj_start(demos):
    obj_start = []
    for key in demos:
        demo = demos[key]
        obj_start.append(demo['obs_object'][0, :3])
    avg_obj_start = np.stack(obj_start, axis=0)
    return np.mean(avg_obj_start, axis=0)

def compute_avg_eef_start(demos):
    eef_start = []
    for key in demos:
        demo = demos[key]
        eef_start.append(demo['obs_robot0_eef_pos'][0, :3])
    avg_eef_start = np.stack(eef_start, axis=0)
    return np.mean(avg_eef_start, axis=0)
        
# #function that takes in segmented demo data and interpolates, return as a list of segments
# def interpolate_demos(demos):
#     pass

#function that 

def paths_regression(self, traj_set, t_set = None):
    '''
    Takes in a set (list) of desired trajectories (with possibly the
    execution times) and generate the weight which realize the best
    approximation.
        each element of traj_set should be shaped num_timesteps x n_dim
        trajectories
    '''

    ## Step 1: Generate the set of the forcing terms
    f_set = np.zeros([len(traj_set), self.n_dmps, self.cs.timesteps])
    g_new = np.ones(self.n_dmps)
    for it in range(len(traj_set)):
        if t_set is None:
            t_des_tmp = None
        else:
            t_des_tmp = t_set[it]
            t_des_tmp -= t_des_tmp[0]
            t_des_tmp /= t_des_tmp[-1]
            t_des_tmp *= self.cs.run_time

        # Alignment of the trajectory so that
        # x_0 = [0; 0; ...; 0] and g = [1; 1; ...; 1].
        x_des_tmp = copy.deepcopy(traj_set[it])
        x_des_tmp -= x_des_tmp[0] # translation to x_0 = 0
        g_old = x_des_tmp[-1] # original x_goal position
        R = roto_dilatation(g_old, g_new) # rotodilatation

        # Rescaled and rotated trajectory
        x_des_tmp = np.dot(x_des_tmp, np.transpose(R))

        # Learning of the forcing term for the particular trajectory
        f_tmp = self.imitate_path(x_des = x_des_tmp, t_des = t_des_tmp,
            g_w = False, add_force = None)
        f_set[it, :, :] = f_tmp.copy() # add the new forcing term to the set

    ## Step 2: Learning of the weights using linear regression
    self.w = np.zeros([self.n_dmps, self.n_bfs + 1])
    s_track = self.cs.rollout()
    psi_set = self.gen_psi(s_track)
    psi_sum = np.sum(psi_set, 0)
    psi_sum_2 = psi_sum * psi_sum
    s_track_2 = s_track * s_track
    A = np.zeros([self.n_bfs + 1, self.n_bfs + 1])
    for k in range(self.n_bfs + 1):
        A[k, k] = scipy.integrate.simps(
            psi_set[k, :] * psi_set[k, :] * s_track_2 / psi_sum_2, s_track)
        for h in range(k + 1, self.n_bfs + 1):
            A[h, k] = scipy.integrate.simps(
                psi_set[k, :] * psi_set[h, :] * s_track_2 / psi_sum_2,
                s_track)
            A[k, h] = A[h, k].copy()
    A *= len(traj_set)
    LU = scipy.linalg.lu_factor(A)

    # The weights are learned dimension by dimension
    for d in range(self.n_dmps):
        f_d_set = f_set[:, d, :].copy()
        # Set up the minimization problem
        b = np.zeros([self.n_bfs + 1])
        for k in range(self.n_bfs + 1):
            b[k] = scipy.integrate.simps(
                np.sum(f_d_set * psi_set[k, :] * s_track / psi_sum, 0),
                s_track)

        # Solve the minimization problem
        self.w[d, :] = scipy.linalg.lu_solve(LU, b)
    self.learned_position = np.ones(self.n_dmps)