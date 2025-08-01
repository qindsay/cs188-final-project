import numpy as np
from collections import defaultdict
from dmp import DMP
from pid import PID
from load_data import reconstruct_from_npz
from multiple_demos import get_closest_demo
from scipy.spatial.transform import Rotation as R


def rotate_quat_sequence(quats, R_delta):
    R_q = R.from_quat(quats)
    R_rot = R.from_matrix(R_delta)
    R_new = R_q * R_rot  # Apply delta after originalAdd commentMore actions

    quats_new = R_new.as_quat()

    # Fix potential discontinuities
    for i in range(1, len(quats_new)):
        if np.dot(quats_new[i], quats_new[i-1]) < 0:
            quats_new[i] = -quats_new[i]

    return quats_new

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
    def __init__(self, obs, demo_path='demos.npz', dt=0.01, n_bfs=20):
        square_pos = obs['SquareNut_pos']
        
        demos = reconstruct_from_npz(demo_path)
        closest_demo = get_closest_demo(demos, square_pos)
        # print(closest_demo.keys())

        # Extract trajectories and grasp
        ee_pos = closest_demo['obs_robot0_eef_pos']  # (T,3)
        # ee_quat = closest_demo['obs_robot0_eef_quat']
        # print("ee_quat shape:", ee_quat.shape)
        # print("ee_quat[0:5]:", ee_quat[0:5])
        T, _ = ee_pos.shape
        ee_grasp = closest_demo['actions'][:, -1:].astype(int)  # (T,1)
        segments = self.detect_grasp_segments(ee_grasp)
        
        

        # Compute offset for first segment to new object pose
        closest_demo_obj_pos = closest_demo['obs_object'][0,:3]
        new_obj_pos = square_pos
        start0, end0 = segments[0]
        offset = ee_pos[end0-1] - closest_demo_obj_pos
        
        # eef_quat = closest_demo['obs_robot0_eef_quat']  # (T, 4)Add commentMore actions

        # eef_quat0 = ee_quat[start0:end0]
        # eef_quat0_rot = rotate_quat_sequence(eef_quat0, R_delta)
        # print("R_delta:\n", R_delta)
        # print("eef_quat0[0]:", eef_quat0[0])
        # print("Rotated quat0[0]:", eef_quat0_rot[0])
        # print("Norm:", np.linalg.norm(eef_quat0_rot[0]))
    
        # TODO: Fit DMPs and generate segment trajectories
        self.dt = dt
        self.grasp = [-1, 1, -1]

        
        dmp0 = DMP(n_dmps=3, n_bfs=n_bfs, dt=dt)
        dmp0.imitate((ee_pos[start0:end0]).T)
        # obj_traj = min(closest_demo['obs_robot0_eef_pos'], key=lambda x: x[0])
        obj_traj = min(ee_pos, key=lambda x: x[0])
        traj_right = obj_traj[1] > 0.14 #demo picks up from the right side of the nut
        new_goal = new_obj_pos + offset
                
        if traj_right: 
            new_goal[2] -= 0.02  # Move 2cm to make sure we actually pick up nut
            print("Adjusting goal down due to traj_right")
        
        #TODO: if right approach, shift goal left. if left approach, shift goal right!!!!!!!
        self.traj0 = dmp0.rollout(new_goal=new_goal)
        # self.quat0 = eef_quat0_rot
        self.len0 = len(self.traj0)
        self.segments = [[0, self.len0-1]]
        
        # eef_quat1 = ee_quat[start1:end1]
        # eef_quat1_rot = rotate_quat_sequence(eef_quat1, R_delta)
        start1, end1 = segments[1]
        dmp1 = DMP(n_dmps=3, n_bfs=n_bfs, dt=dt)
        # seg1 = ee_pos[start1:end1] - closest_demo_obj_pos
        # seg1_rotated = seg1 @ R_delta.T
        # new_segment1 = seg1_rotated + new_obj_pos
        # dmp1.imitate(new_segment1.T)
        dmp1.imitate((ee_pos[start1:end1]).T)
        self.traj1 = dmp1.rollout()
        # self.quat1 = eef_quat1_rot

        self.len1 = len(self.traj1)
        self.segments.append([self.segments[-1][1]+1, self.segments[-1][1] + self.len1])
        
        start2, end2 = segments[2]
        dmp2 = DMP(n_dmps=3, n_bfs=n_bfs, dt=dt)
        dmp2.imitate((ee_pos[start2:end2]).T)
        orig_orientation = ee_pos[end2 - 1].copy()
        adjusted_orientation = orig_orientation.copy()
        if traj_right:
            adjusted_orientation[1] -= 0.05 
            adjusted_orientation[0] += 0.02
            adjusted_orientation[2] += 0.03
            print("if coming from right-side")
        else:
            adjusted_orientation[1] += 0.05
            adjusted_orientation[0] -= 0.06
            adjusted_orientation[2] += 0.03
            print("if coming from left-side approach")
        self.traj2 = dmp2.rollout(new_goal=adjusted_orientation)
        # self.traj2 = dmp2.rollout()
        # self.quat2 = ee_quat[start2:end2]
        self.len2 = len(self.traj2)
        self.segments.append([self.segments[-1][1]+1, self.segments[-1][1] + self.len2])
        
        # self.quaternions = [self.quat0, self.quat1, self.quat2]

        self.pid = PID(kp=8.0, ki=0.4, kd=0.4, target=self.traj0[0])
        # self.pid = PID(kp=10.0, ki=0.1, kd=0.1, target=np.zeros(len(self.traj0[0])))
        self.stage = 0
        self.step = 0
        self.justChanged = False

        # nudge to push it down
        self.nudge = False
        self.nudge_count = 0
        self.nudge_aligned = None

        # self.delay_counter = 0
        # self.delay_duration = 100

        
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

        if self.stage >= 3:
            if not self.rotation_applied:
                angle = np.deg2rad(5) # adjust angle by 5
                rot_matrix = R.from_euler('z', angle).as_matrix()
            
                rot_quat = R.from_matrix(rot_matrix).as_quat()
                rot_quat = rot_quat / np.linalg.norm(rot_quat)

                action = np.zeros(7)
                action[0:3] = np.zeros(3)
                action[3:7] = rot_quat
                action[6] = 0.0
                self.rotation_applied = True
                print("Rotate a bit before drop")
                return action
            if not self.nudge:
                self.nudge = True
                self.nudge_aligned = robot_eef_pos.copy()
                self.nudge_count = 0
                self.nudge_direction = np.array([0.05, 0.00, -0.03]) 
                self.pid.reset(target=self.nudge_aligned + self.nudge_direction)
                print("Starting nudge")
            if self.nudge_count < 10:
                output = self.pid.update(robot_eef_pos, self.dt)

                action = np.zeros(7)
                action[0:3] = output
                action[6] = 0.0
                self.nudge_count += 1
                return action
            # if self.delay_counter < self.delay_duration:
            #     # after nudge is done, delay
            #     self.delay_counter += 1
            #     return np.zeros(7)
            else:
                return np.zeros(7)

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
            if self.stage == 2:
                self.pending_rotation = True
                self.rotation_applied = False
            self.stage += 1
            self.justChanged = True
            print("Moved onto", self.stage)
        
        output = self.pid.update(robot_eef_pos, self.dt)
        action = np.zeros(7)
        action[0:3] = output
        # action[3:7] = self.target_quat[self.step]
        if self.stage < 3:
            # quat_idx = self.step - self.segments[self.stage][0]Add commentMore actions
            # quat_idx = np.clip(quat_idx, 0, len(self.quaternions[self.stage]) - 1)
            # current_quat = self.quaternions[self.stage][quat_idx]
            # current_quat = current_quat / np.linalg.norm(current_quat)
            # action[3:7] = current_quat
            
            action[6] = self.grasp[self.stage]
        else:
            action[6] = 0
        
        return action