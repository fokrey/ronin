import pyvista as pv
import numpy as np

from pathlib import Path
import sys
sys.path.insert(0,str(Path('./source').resolve()))

import time

import mrob
import imageio

def stop_animation():
    global running
    running = False

def continue_animation():
    global running
    running = True


if __name__ == "__main__":
    file_path = "visualization/poses/poses_circle2.txt"

    data = np.genfromtxt(file_path)
    

    traj = data[:, 1:4]  # smartphone trajectory
    quat = data[:, 4:]  # smartphone attitude

    p = pv.Plotter(off_screen=False, notebook=False)
    
    # Dimensions 13: H: 146.7 mm (5.78 in) W: 71.5 mm (2.81 in) D: 7.65 mm (0.301 in)
    #smartphone = p.add_mesh(pv.Box((-0.00765, 0.00765, -0.1467 / 2, 0.1467 / 2, -0.0715, 0.0715)))
    smartphone = p.add_mesh(pv.Box((-0.765, 0.765, -14.67/2, 14.67/2, -7.15/2, 7.15/2)))
    p.add_camera_orientation_widget()
    p.show_grid()
    p.add_axes()

    # TODO: Add callbacks to control playback
    
    p.add_lines(traj.repeat(2, axis=0)[1:-1], color='black')

    oX = p.add_mesh(pv.Arrow(np.array([0, 0, 0]), np.array([1, 0, 0]), scale=10), color='red')
    oY = p.add_mesh(pv.Arrow(np.array([0, 0, 0]), np.array([0, 1, 0]), scale=10), color='green')
    oZ = p.add_mesh(pv.Arrow(np.array([0, 0, 0]), np.array([0, 0, 1]), scale=10), color='blue')

    p.show(interactive_update=True)
    
    p.add_key_event("s", stop_animation)
    p.add_key_event("c", continue_animation)

    running = True
    current_frame = 0  # Start from the first frame
    frames = [] 

    while current_frame < len(data) - 1:    
        p.update()
        time.sleep(0.05)  # Reduced delay for smoother interaction
        
        if not running:
            continue

        R_m = mrob.geometry.quat_to_so3(quat[current_frame])

        pose = np.identity(4)
        pose[:3, :3] = R_m
        pose[:3, 3:] = traj[current_frame].reshape(-1, 1)

        smartphone.user_matrix = pose
        oX.user_matrix = pose
        oY.user_matrix = pose
        oZ.user_matrix = pose  
        
        screenshot = p.screenshot()  # Capture the current frame as an image
        frames.append(screenshot)
        
        current_frame += 1
        
    gif_filename = "animation.gif"
    with imageio.get_writer(gif_filename, mode='I', duration=0.05) as writer:
        for frame in frames:
            writer.append_data(frame)