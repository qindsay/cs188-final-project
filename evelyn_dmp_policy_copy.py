import numpy as np
from collections import defaultdict
from dmp import DMP
from pid import PID
from load_data import reconstruct_from_npz
from multiple_demos import combine, split_demos, compute_avg_traj


class DMPPolicyWithPID:
    """
    A policy that follows a demonstrated path with DMPs and PID control.

    The demonstration is split into segments based on grasp toggles.  
    The first segment's endpoint is re-targeted to a new object pose.
    Subsequent segments replay the original DMP rollouts.

    Args:
        square_obs (dict): 'SquareNut_pos' observed
        demo_path (str): path to .npz file with demo data.
        dt (float): control timestep.
        n_bfs (int): number of basis functions per DMP.
    """
    def __init__(self, square_pos, demo_path='demos.npz', dt=0.01, n_bfs=20):
        # Load and parse demo [CA3 CODE]
            # raw = np.load(demo_path)
            # demos = defaultdict(dict)
            # demo = np.mean(np.stack(demos), axis=0)
            # print(list(demos.keys())[:10])
            # print(list(demos.values())[:10])
            
            # for key in raw.files:
            #     prefix, trial, field = key.split('_', 2)
            #     demos[f"{prefix}_{trial}"][field] = raw[key]
            # demo = demos['demo_98']
        
        demos = reconstruct_from_npz(demo_path)
        demo = combine(demos)
    
        # Extract trajectories and grasp
        ee_pos = demo['obs_robot0_eef_pos']  # (T,3)
        T, _ = ee_pos.shape
        ee_grasp = demo['actions'][:, -1:].astype(int)  # (T,1)
        segments = self.detect_grasp_segments(ee_grasp)

        # Compute offset for first segment to new object pose
        demo_obj_pos = demo['obs_object'][0, :3]
        new_obj_pos = square_pos
        start, end = segments[0]
        offset = ee_pos[end-1] - demo_obj_pos

        # TODO: Fit DMPs and generate segment trajectories
        self.dt = dt
        self.grasp = [-1, 1, -1]
        dmp0 = DMP(n_dmps=3, n_bfs=n_bfs, dt=dt)
        # imitate take in the average segment trajectories
        dmp0.imitate((ee_pos[start:end]).T)
        self.traj0 = dmp0.rollout(new_goal=new_obj_pos + offset)
        self.len0 = len(self.traj0)
        self.segments = [[0, self.len0-1]]
        
        start1, end1 = segments[1]
        dmp1 = DMP(n_dmps=3, n_bfs=n_bfs, dt=dt)
        dmp1.imitate((ee_pos[start1:end1]).T)
        self.traj1 = dmp1.rollout()
        self.len1 = len(self.traj1)
        self.segments.append([self.segments[-1][1]+1, self.segments[-1][1] + self.len1])
        
        start2, end2 = segments[2]
        dmp2 = DMP(n_dmps=3, n_bfs=n_bfs, dt=dt)
        dmp2.imitate((ee_pos[start2:end2]).T)
        self.traj2 = dmp2.rollout()
        self.len2 = len(self.traj2)
        self.segments.append([self.segments[-1][1]+1, self.segments[-1][1] + self.len2])
        
        self.pid = PID(kp=2.0, ki=0.4, kd=0.4, target=self.traj0[0])
        self.stage = 0
        self.step = 0
        self.justChanged = False
        
    def detect_grasp_segments(self, grasp_flags: np.ndarray) -> list:
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

    def get_action(self, robot_eef_pos: np.ndarray) -> np.ndarray:
        """
        Compute next action for the robot's end-effector.

        Args:
            robot_eef_pos (np.ndarray): Current end-effector position [x,y,z].

        Returns:
            np.ndarray: Action vector [dx,dy,dz,0,0,0,grasp].
        """
        # TODO: select current segment and step
        # TODO: compute PID-based delta_pos
        # TODO: assemble action (zero rotation + grasp)
        
        OBJ_DIST_THRESH = 0.03
        
        if self.stage == 0:  
            self.pid.target = self.traj0[self.step]
            delta_pos = np.linalg.norm(self.pid.target - robot_eef_pos)
            
            if delta_pos < OBJ_DIST_THRESH:
                self.step += 1
        elif self.stage == 1:
            if self.justChanged:
                self.pid.reset(target=self.traj1[0])
                self.justChanged = False                  
            
            delta_pos = np.linalg.norm(self.pid.target - robot_eef_pos)
            
            if delta_pos < OBJ_DIST_THRESH:
                self.pid.target = self.traj1[self.step-self.len0]
                self.step += 1
        elif self.stage == 2:
            if self.justChanged:
                self.pid.reset(target=self.traj2[0])
                self.justChanged = False
                
            delta_pos = np.linalg.norm(self.pid.target - robot_eef_pos)
            
            if delta_pos < OBJ_DIST_THRESH:
                self.pid.target = self.traj2[self.step-(self.len1+self.len0)]
                self.step += 1

            
        if self.stage < 3 and self.step > self.segments[self.stage][1]:
            self.stage += 1
            self.justChanged = True
            print("Moved onto", self.stage)
        
        output = self.pid.update(robot_eef_pos, self.dt)
        action = np.zeros(7)
        action[0:3] = output
        if self.stage < 3:
            action[6] = self.grasp[self.stage]
        else:
            action[6] = 0
        
        return action