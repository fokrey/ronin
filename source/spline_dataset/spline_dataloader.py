# from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np

from spline_dataset.spline_diff import generate_imu_data
from spline_dataset.spline_generation import generate_batch_of_splines

# from spline_diff import generate_imu_data
# from spline_generation import generate_batch_of_splines

import glob
from tqdm import tqdm
import mrob
import os
import pickle

import torch
from torch.utils.data import Dataset

def get_gt_se3_Poses(poses):
    result = [None]*len(poses)
    angles = np.zeros((len(poses),3))
    angles[:,2] = np.arctan2(poses[:,3],poses[:,2])
    xyz = np.zeros((len(poses),3))
    xyz[:,:2] = poses[:,:2]

    for i in range(len(poses)):
        result[i] = mrob.SE3(mrob.SO3(angles[i]),xyz[i])

    return result

def get_gt_se3vel_Poses(poses, velocities):
    result = [None]*len(poses)
    angles = np.zeros((len(poses),3))
    angles[:,2] = np.arctan2(poses[:,3],poses[:,2])
    xyz = np.zeros((len(poses),3))
    xyz[:,:2] = poses[:,:2]

    v_xyz = np.zeros((len(poses),3))
    v_xyz[:,:2] = velocities

    for i in range(len(poses)):
        result[i] = mrob.SE3vel(mrob.SO3(angles[i]),xyz[i], v_xyz[i])
    return result

# class that reads generated IMU data for spline curves on a 2d plane
class Spline_2D_Dataset():
    def __init__(self, spline_path, window = 100, enable_noise = True):

        self.window = window

        self.bias_acc = np.array([0,0])
        if enable_noise:
            self.Q_acc = np.array([0.9**2,0,0,0.9**2]).reshape(2,2)
        else:
            self.Q_acc = np.zeros((2,2))

        self.bias_w = np.array([0])
        if enable_noise:
            self.Q_w = np.array([0.15**2]).reshape(1,1)
        else:
            self.Q_w = np.zeros((1,1))

        paths = glob.glob(f'{spline_path}spline_*.txt')
        print(f"Found {len(paths)} splines in path: {spline_path}")

        B = len(paths) # this is our batch size

        self.data = np.zeros((B,window,3))#np.concatenate((acc,gyro),axis=1)

        self.slices = [[] for _ in range(B)]
        self.gt_odometry = [[] for _ in range(B)]
        self.gt_traj = [[] for _ in range(B)]
        self.gt_poses = [[] for _ in range(B)]
        self.gt_velocity = [[] for _ in range(B)]
        self.time = [[] for _ in range(B)]

        for b in tqdm(range(len(paths))):
            # 1 sample == 1 spline

            spline_points = np.genfromtxt(paths[b])

            self.gt_traj[b] = spline_points

            acc, gyro, tau, n, velocity, time = generate_imu_data(spline_points)
            # TODO inject noise:
            # -additive
            # -multiplicative

            tmp_data = np.concatenate((acc,gyro),axis=1)
            
            # splitting 1 track into several slices
            slice_num = tmp_data.shape[0] // self.window
            for i in range(slice_num):

                temp_slice = tmp_data[i*self.window : (i+1)*self.window]

                acc_noise = np.random.multivariate_normal(self.bias_acc,self.Q_acc,self.window)
                temp_slice[:,:2] = temp_slice[:,:2] + acc_noise

                omega_noise = np.random.multivariate_normal(self.bias_w, self.Q_w,self.window)
                temp_slice[:,2:] = temp_slice[:,2:] + omega_noise

                self.slices[b].append(temp_slice) # adding i-th slice into b-th track

                # need to rotate points using first tau vector orientation
                c = tau[i*self.window][0]
                s = tau[i*self.window][1]

                R = np.array([[c,-s],[s,c]])

                tmp = spline_points[i*self.window : (i + 1)*self.window]@R

                tmp = tmp - tmp[0]

                # calculating orientation increment

                tmp = np.concatenate((tmp,tau[i*self.window : (i + 1)*self.window]@R),axis=1)

                self.gt_odometry[b].append(tmp)

            self.gt_poses[b] = np.concatenate((self.gt_traj[b][:-1], tau),axis=1)[::window]
            self.gt_velocity[b] = velocity[::window]
            self.time[b] = time[::window]

        self.X = np.array(self.slices)
        self.y = np.array(self.gt_odometry)
        self.gt_traj = np.array(self.gt_traj)
        self.gt_poses = np.array(self.gt_poses)
        self.gt_velocity = np.array(self.gt_velocity)
        self.gt_velocity.reshape(-1, 2).shape
        self.time = np.array(self.time)
        
        self.adjust_shape()
        
        self.X = self.X.reshape(-1, self.X.shape[2], self.X.shape[3])
        self.gt_velocity = self.gt_velocity.reshape(-1, self.gt_velocity.shape[-1])
        
    def normalize(self):
        imu_data = np.concatenate(self.X, axis=0)
        self.imu_mean = imu_data.mean(axis=(0, 1))
        self.imu_std = imu_data.std(axis=(0, 1))
        for i in range(len(self.X)):
            self.x[i] = (self.X[i] - self.imu_mean) / self.imu_std
            
    def adjust_shape(self):
        min_length = min(self.X.shape[1], self.gt_velocity.shape[1])
        self.X = self.X[:, :min_length, :, :]
        self.gt_velocity = self.gt_velocity[:, :min_length, :]
        
        #self.X = self.X.squeeze(2)
        

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self,idx):
        # xy = self.gt_poses[idx][:,:2]
        # cs = self.gt_poses[idx][:,-2:]
        # angles = np.array(np.arctan2(cs[:,1],cs[:,0]))
        
        # result = {
        #     'imu': self.X[idx],
        #     'gt_odometry' : self.y[idx],
        #     'gt_traj': self.gt_traj[idx],
        #     'gt_poses' : self.gt_poses[idx],
        #     'gt_velocity': self.gt_velocity[idx],
        #     'gt_orientation' : angles,
        #     'gt_se3' : get_gt_se3_Poses(self.gt_poses[idx]),
        #     #'gt_se3vel' : get_gt_se3vel_Poses(self.gt_poses[idx], self.gt_velocity[idx]),
        #     'time': self.time[idx]
        # }
        
        imu = torch.tensor(self.X[idx], dtype=torch.float32).permute(1, 0)
        velocity = torch.tensor(self.gt_velocity[idx], dtype=torch.float32)
        return imu, velocity
    

if __name__ == "__main__":
    output_path = './out/'

    if not os.path.exists(output_path):
        os.makedirs(output_path,exist_ok=True)
    path_to_splines = output_path + 'splines/'

    number_of_splines = 10
    if not os.path.exists(path_to_splines):
        number_of_control_nodes = 10
        generate_batch_of_splines(path_to_splines, number_of_splines, number_of_control_nodes, 100)

    if not os.path.isfile(path_to_splines + f'spline_dataset_{number_of_splines}.pkl'):
        dataset = Spline_2D_Dataset(path_to_splines, window=10, enable_noise = not True)
        pickle.dump(dataset,open(path_to_splines + f'spline_dataset_{number_of_splines}.pkl','wb'))
    else:
        dataset = pickle.load(open(path_to_splines + f'spline_dataset_{number_of_splines}.pkl','rb'))
    dataset = Spline_2D_Dataset(path_to_splines, window=10, enable_noise= not True)

    print(f"{dataset.__len__()=}")
    print(f"{dataset.__getitem__(0)=}")

    # for i in range(len(dataset)):
    #     #sample: dict_keys(['imu', 'y', 'gt_traj', 'gt_poses', 'gt_velocity', 'gt_orientation', 'gt_se3', 'gt_se3vel', 'time])

    #     s = dataset.__getitem__(i)

    #     plt.plot(s['time'],s['imu'][...,0],label='acc_x')
    #     plt.plot(s['time'],s['imu'][...,1],label='acc_y')
    #     plt.plot(s['time'],s['imu'][...,2],label='omega_z')
    #     plt.grid()
    #     plt.legend()


    #     plt.figure()
    #     plt.plot(s['gt_traj'][...,0],s['gt_traj'][...,1],label='gt traj')
    #     plt.grid()
    #     plt.axis('equal')
    #     plt.legend()

    #     plt.show()

    N = len(dataset)

    for i in range(N):
        print(dataset.__getitem__(i)[0].shape)
        print(dataset.__getitem__(i)[1].shape)