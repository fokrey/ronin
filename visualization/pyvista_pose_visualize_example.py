import pyvista as pv
import numpy as np

from pathlib import Path
import sys
sys.path.insert(0,str(Path('./source').resolve()))
import time

import mrob



from transformations import change_cf

if __name__ == "__main__":
    file_path = "./visualization/poses.txt"

    data = np.genfromtxt(file_path)

    traj = data[:,1:4] # smartphone trajectory
    quat = data[:,4:] # smartphone attitute

    # downsample = 20

    p = pv.Plotter(off_screen=False)

    # Dimensions	13: H: 146.7 mm (5.78 in) W: 71.5 mm (2.81 in) D: 7.65 mm (0.301 in) 
    smartphone = p.add_mesh(pv.Box((-0.0715,0.0715,-0.1467/2,0.1467/2,-0.00765,0.00765)))
    p.add_camera_orientation_widget()
    p.show_grid()
    p.add_axes()

    # TODO add call backs to control playback

    p.add_lines(traj.repeat(2,axis=0)[1:-1],color='black')

    oX = p.add_mesh(pv.Arrow(np.array([0,0,0]),np.array([1,0,0]),scale=0.5),color='red')
    oY = p.add_mesh(pv.Arrow(np.array([0,0,0]),np.array([0,1,0]),scale=0.5),color='green')
    oZ = p.add_mesh(pv.Arrow(np.array([0,0,0]),np.array([0,0,1]),scale=0.5),color='blue')

    p.show(interactive_update=True)
    for i in range(0, len(data)-1):
        p.update()
        time.sleep(0.5)

        R_m = mrob.geometry.quat_to_so3(quat[i])

        # R = change_cf(quat[i],np.eye(3))
        pose = np.identity(4)
        pose[:3,:3] = R_m
        pose[:3, 3:] = traj[i].reshape(-1,1)

        oX.user_matrix = pose
        oY.user_matrix = pose
        oZ.user_matrix = pose

        smartphone.user_matrix = pose
