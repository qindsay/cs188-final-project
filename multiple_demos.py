import numpy as np
import scipy

def get_closest_demo(demos, obj_pos):
    closest = None
    closest_dist = np.inf
    
    for d in demos:
        demo = demos[d]
        nut_dist = min(demo['obs_robot0_eef_pos'], key=lambda x: x[0]) #focus on replicating eef behavior, find traj that gets closest to nut
        
        cur_dist = np.linalg.norm(nut_dist - obj_pos) 
        if cur_dist < closest_dist:
            closest_dist = cur_dist
            closest = demo
    
    return closest