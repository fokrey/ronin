import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from tqdm import tqdm

from spline_dataset.spline_diff import generate_imu_data
from spline_dataset.spline_generation import generate_batch_of_splines
from spline_dataset.spline_dataloader import Spline_2D_Dataset

import mrob
from mrob_num_diff.num_diff_3d import read_graph_toro_description_3d, compose_graph_3d, numerical_diff2_3d, numerical_diff1_3d
from mrob_num_diff.graph_generator import ToRoContainer


def pose_to_se3(pose):
    x, y, cos_theta, sin_theta = pose
    theta = np.arctan2(sin_theta, cos_theta)

    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, 0]
    return T


output_path = './out/'
# toro_file = os.path.join(output_path, 'toro_file.txt')

def print_2d_graph(graph, gt_poses):
    x = graph.get_estimated_state()
    
    prev_p = np.array(graph.get_estimated_state()[0][:2, 3])
    plt.figure()

    for p in x:
        p = p[:2, 3]
        plt.plot(p[0], p[1], 'ob')
        plt.plot((prev_p[0], p[0]), (prev_p[1], p[1]), '-b', label='estimated')
        
        prev_p = p
    
    plt.plot(gt_poses[:, :2][:, 0], gt_poses[:, :2][:, 1], label='GT', color='red')
    plt.title("2D Pose Graph")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid()
    plt.show()


def visualize_gradient(gradient, title, dir_to_save, dx = None, dz=None):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    ax[0].imshow(gradient)
    ax[1].spy(gradient, precision=1e-7)
    if dx == None:
        plt.suptitle(f'{title}\n {dz=}')
        plt.savefig(os.path.join(dir_to_save, f'gradient1_dz={dz}.png'))
    else:
        plt.suptitle(f'{title}\n {dx=}, {dz=}')
        plt.savefig(os.path.join(dir_to_save, f'gradient2_dx={dx}_dz={dz}.png'))
    


def normalize_matrix(matrix):
    print(f'Norm of matrix: {np.linalg.norm(matrix)}')
    return matrix / np.linalg.norm(matrix)


def mean_squared_error(matrix1, matrix2):
    return np.linalg.norm(matrix1 - matrix2)


def compare_gradients(gradient1, gradient2, dx=None, dz=None):
    print('Norm of gradient1:', np.linalg.norm(gradient1))
    print('Norm of gradient2:', np.linalg.norm(gradient2))
    
    vmin = min(gradient1.min(), gradient2.min())
    vmax = max(gradient1.max(), gradient2.max())
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    im1 = ax[0].imshow(gradient1, vmin=vmin, vmax=vmax, cmap='viridis')
    ax[0].set_title(f'Gradient #1, {dx=}')
    im2 = ax[1].imshow(gradient2, vmin=vmin, vmax=vmax, cmap='viridis')
    ax[1].set_title(f'Gradient #2, {dx=}, {dz=}')
    fig.colorbar(im1, ax=ax, orientation='vertical', fraction=0.02, pad=0.04)
    plt.show()
    
    mse_value = mean_squared_error(gradient1, gradient2)
    print(f'MSE value: {mse_value}')



def plot_velocities(predictions, velocities):
    plt.figure(figsize=(12, 6))
    plt.plot(predictions[:, 0], label="Predicted V_x", linestyle="--")
    plt.plot(predictions[:, 1], label="Predicted V_y", linestyle="--")
    plt.plot(velocities[:, 0], label="Ground Truth V_x")
    plt.plot(velocities[:, 1], label="Ground Truth V_y")
    plt.xlabel('Time')
    plt.ylabel('Velocity')
    plt.legend()
    plt.show()

    
def plot_losses(losses, title):
    plt.figure(figsize=(12, 6))
    plt.plot(losses, marker='o', linestyle='-', color='b')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel(title, fontsize=12)
    plt.title(f'{title} per Epoch', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()   


def populate_graph(imu, gt_velocity, gt_poses, pred_velocity, gt_poses_se3, dt=0.1):
    np.random.seed(24)
    graph = mrob.FGraph()
    toro_container = ToRoContainer()

    W_odo = np.eye(6) * 0.01
    W_gps = np.eye(6) * 10.0

    node_ids = []

    # Adding all nodes
    traj = np.cumsum(gt_velocity.detach().numpy()*dt, axis=0) + gt_poses[0, :2].detach().numpy()
    traj_odo = np.cumsum(pred_velocity.detach().numpy()*dt, axis=0) + gt_poses[0, :2].detach().numpy()
    T_nodes = []
    for i in range(len(gt_poses)):
        if isinstance(gt_poses[i], mrob.SE3):
            T_i = gt_poses[i]
        else:
            # Convert [x, y, cos, sin] to SE3
            x, y, c, s = gt_poses[i]
            # R = np.array([[c,s],[-s,c]])
            yaw = np.arctan2(s, c)
            #T_i = mrob.SE3([0.0, 0.0, yaw, x, y, 0.0])
            # dx = vx * dt
            # dy = vy * dt
            # x, y = traj_odo[i]
            # dx = dx * c + dy * s
            # dy = -dx * s + dy * c
            
            T_i = mrob.SE3(pose_to_se3(np.array([x, y, c, s] + 0.15 * np.random.rand(4))))
            #TODO perturb gt_pose
            # perturbation = np.zeros_like(gt_poses_se3[i].detach().numpy())
            # perturbation[0, 3] = 0.012
            # perturbation[1, 3] = 0.01
            # gt_pose_se3_perturbed = gt_poses_se3[i].detach().numpy() + perturbation
            # T_i = mrob.SE3(gt_pose_se3_perturbed)
            T_nodes.append(T_i)
               
        node_id = graph.add_node_pose_3d(T_i)
        node_ids.append(node_id)
            
        toro_container.add_node_pose_3d(node_id, T_i.Ln())

        # Add "GPS" factor
        graph.add_factor_1pose_3d(T_i, node_id, W_gps)
        toro_container.add_factor_1pose_3d(node_id, T_i.Ln(), W_gps)

    # Adding odometry factors (predicted velocity)
    for i in range(len(node_ids) - 1):
        vx, vy = pred_velocity[i]
        #TODO rotate pred_velocity to local frame (pred_velocity is now in global frame)
        dx = vx * dt
        dy = vy * dt
        # dx, dy = traj_odo[i]
        rotation = T_nodes[i].R()
        rotated_xyz = np.array([dx.item(), dy.item(), 0]) @ rotation
        gt_rotated_xyz = np.array([gt_velocity[i][0].item() * dt, gt_velocity[i][1].item() * dt, 0]) @ rotation
        
        odo = mrob.SE3(pose_to_se3(np.array([rotated_xyz[0], rotated_xyz[1], 0, 0])))
        gt_odo = mrob.SE3(pose_to_se3(np.array([gt_rotated_xyz[0], gt_rotated_xyz[1], 0, 0])))
        
        graph.add_factor_2poses_3d(odo, node_ids[i], node_ids[i + 1], W_odo)
        toro_container.add_factor_2poses_3d(node_ids[i], node_ids[i + 1], odo.Ln(), W_odo) # .Ln() [x, y, z, roll, pitch, yaw]
        
    for i in range(len(node_ids) - 1):
        vx, vy = gt_velocity[i]
        dx = vx * dt
        dy = vy * dt
        # dx, dy = traj_odo[i]
        rotation = T_nodes[i].R()
        rotated_xyz = np.array([dx.item(), dy.item(), 0]) @ rotation
        gt_odo = mrob.SE3(pose_to_se3(np.array([rotated_xyz[0], rotated_xyz[1], 0, 0]))) 
    
    # print_2d_graph(graph, gt_poses)
    graph.solve(mrob.LM, verbose=False)
    # print_2d_graph(graph, gt_poses)
    
    return graph, toro_container.get_lines()


if not os.path.exists(output_path):
    os.makedirs(output_path,exist_ok=True)
path_to_splines = output_path + 'splines/'

number_of_splines = 1
if not os.path.exists(path_to_splines):
    number_of_control_nodes = 10
    generate_batch_of_splines(path_to_splines, number_of_splines, number_of_control_nodes, 100)

if not os.path.isfile(path_to_splines + f'spline_dataset_{number_of_splines}.pkl'):
    dataset = Spline_2D_Dataset(path_to_splines, window=10, enable_noise = not True)
    pickle.dump(dataset,open(path_to_splines + f'spline_dataset_{number_of_splines}.pkl','wb'))
else:
    dataset = pickle.load(open(path_to_splines + f'spline_dataset_{number_of_splines}.pkl','rb'))
    
dataset = Spline_2D_Dataset(path_to_splines, window=10, enable_noise= not True)
dataloader = DataLoader(dataset, batch_size=79, shuffle=False)

from ronin_resnet import ResNet1D, BasicBlock1D, FCOutputModule

model = ResNet1D(
    num_inputs=3,       
    num_outputs=2,         
    block_type=BasicBlock1D,
    group_sizes=[2, 2, 2],   
    base_plane=64,
    output_block=FCOutputModule,  
    kernel_size=3,
    fc_dim=512,             
    in_dim=1,            
    dropout=0.5,
    trans_planes=128
)
model.load_state_dict(torch.load("out/model.pth"))

optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
def loss_delta_x(gt_poses, estimated_poses):
    result_se3 = gt_poses @ np.linalg.inv(estimated_poses) #change to mrob.SE3.inv()
    result_Ln = np.array([mrob.SE3(result_se3[i]).Ln() for i in range(len(result_se3))])
    return result_Ln.reshape(-1)


def compute_gradient(matrix, vector):
    N = matrix.shape[0]
    mult = (vector @ matrix)[:, 0:N].reshape(-1, 6)
    v_grad = mult[:, 3:5]
    return v_grad

def rotate_velocity(poses, velocity):
    N = len(poses)
    velocity_detach = velocity.detach().numpy()
    rotated_velocity = np.zeros_like(velocity_detach)
    for i in range(N):
        x, y, c, s = poses[i]
        yaw = np.arctan2(s, c)
        R = np.array([[np.cos(yaw), -np.sin(yaw)],
                       [np.sin(yaw), np.cos(yaw)]])
        rotated_velocity[i] = R @ velocity_detach[i].T
    return rotated_velocity    

num_epochs = 20
pred_velocity = []

losses = []
chi2_errors = []
predictions = velocities = None
graph_to_plot = None
gt_poses = None

for epoch in (range(num_epochs)):
    epoch_loss = 0
    # model.train()
    for imu, gt_velocity, gt_poses, gt_poses_se3 in dataloader:
        model.train()
        pred_velocity = model(imu)
        # rotated_pred_velocity = rotate_velocity(gt_poses, pred_velocity)
        graph, toro_lines = populate_graph(imu, gt_velocity, gt_poses, pred_velocity, gt_poses_se3)
        graph.solve(mrob.LM)
        print(f"Chi-squared error after optimization: {graph.chi2()}")
        print_2d_graph(graph, gt_poses)
        toro_file = os.path.join(output_path, 'toro_file.txt')
        with open(toro_file,'w') as f:
            f.writelines(toro_lines)
            f.close()

        chi2_dx_dz = numerical_diff2_3d(toro_file, dx=1e-4, dz=1e-4)
        dx_dz = numerical_diff1_3d(toro_file, dz=1e-4)
        # compare_gradients(dx_dz, chi2_dx_dz, dx=1e-4, dz=1e-4)
        chi2_errors.append(graph.chi2())
        
        delta_x = loss_delta_x(gt_poses_se3.detach().cpu().numpy(), graph.get_estimated_state())
        N = chi2_dx_dz.shape[0]
        mult = (delta_x @ chi2_dx_dz)[:, 0:N].reshape(-1, 6)
        v_grad = mult[:, 3:5]
        v_grad = v_grad / 0.01

        optimizer.zero_grad()
        pred_velocity.backward(gradient=torch.tensor(v_grad, dtype=torch.float32))
        optimizer.step()
        
        model.eval()
        valid_velocity = model(imu)
        graph_valid, toro_lines = populate_graph(imu, gt_velocity, gt_poses, valid_velocity, gt_poses_se3)
        # graph_valid.solve(mrob.LM, verbose=False)
        graph_to_plot = graph_valid
        # print_2d_graph(graph_valid, gt_poses)
        velocity_loss = F.mse_loss(valid_velocity, gt_velocity)
        epoch_loss += velocity_loss.item()
        
        predictions = valid_velocity.cpu().detach().numpy().copy()
        velocities = gt_velocity.cpu().detach().numpy().copy()
        # predictions =  rotated_pred_velocity
    avg_loss = epoch_loss / len(dataloader)
    losses.append(avg_loss)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.6f}")

plot_velocities(predictions, velocities)
    
plot_losses(losses, 'MSE loss')
plot_losses(chi2_errors, 'chi2')
print_2d_graph(graph_to_plot, gt_poses)