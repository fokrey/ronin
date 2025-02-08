from ronin_resnet import ResNet1D, BasicBlock1D, FCOutputModule
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np

from spline_dataset.spline_diff import generate_imu_data
from spline_dataset.spline_generation import generate_batch_of_splines
from spline_dataset.spline_dataloader import Spline_2D_Dataset
import os
import pickle

import matplotlib.pyplot as plt

def plot_velocities(predictions, velocities):
    plt.figure(figsize=(12, 6))
    plt.plot(predictions[:, 0], label="Predicted V_x", linestyle="--")
    plt.plot(predictions[:, 1], label="Predicted V_y", linestyle="--")
    plt.plot(velocities[:, 0], label="Ground Truth V_x")
    plt.plot(velocities[:, 1], label="Ground Truth V_y")
    plt.legend()
    plt.grid(True)
    plt.xlabel('Time')
    plt.ylabel('Velocity')
    plt.savefig('velocities.png')

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

#model = model.to(device)

criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=1e-4)


output_path = './out/'

if not os.path.exists(output_path):
    os.makedirs(output_path,exist_ok=True)
path_to_splines = output_path + 'splines_train/'

number_of_splines = 10
if not os.path.exists(path_to_splines):
    number_of_control_nodes = 10
    generate_batch_of_splines(path_to_splines, number_of_splines, number_of_control_nodes, 100)

if not os.path.isfile(path_to_splines + f'spline_dataset_{number_of_splines}.pkl'):
    dataset = Spline_2D_Dataset(path_to_splines, window=100, enable_noise = not True)
    pickle.dump(dataset,open(path_to_splines + f'spline_dataset_{number_of_splines}.pkl','wb'))
else:
    dataset = pickle.load(open(path_to_splines + f'spline_dataset_{number_of_splines}.pkl','rb'))
    
dataset = Spline_2D_Dataset(path_to_splines, window=10, enable_noise= not True)

train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, valid_dataset = random_split(dataset, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=16, shuffle=False)

test_dataset = Spline_2D_Dataset(path_to_splines, window=10, enable_noise= not True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

print(f"Train size: {len(train_dataset)}, Test size: {len(valid_dataset)}")

# for X, y in train_dataloader:
#     print(X.shape)
#     print(y.shape)    

# Training loop
train_losses = []
num_epochs = 30
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0

    for imu, velocity, _, _ in train_dataloader:
        imu, velocity = imu, velocity

        predictions = model(imu)

        loss = criterion(predictions, velocity)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_dataloader):.4f}")
    train_losses.append(epoch_loss/len(train_dataloader))

torch.save(model.state_dict(), "out/model.pth")

model.eval()
total_loss = 0.0
valid_losses = []
predictions = velocities = None
with torch.no_grad():
    for i, (imu, velocity, _, _) in enumerate(test_dataloader):
        imu, velocity = imu, velocity

        pred = model(imu)
        if predictions is None:
            predictions = pred.cpu().detach().numpy().copy()
        else:
            predictions = np.vstack((predictions, pred.cpu().detach().numpy()))
            
        if velocities is None:
            velocities = velocity.cpu().detach().numpy().copy()
        else:
            velocities = np.vstack((velocities, velocity.cpu().detach().numpy()))

        loss = criterion(pred, velocity)
        total_loss += loss.item()
       
    avg_loss = total_loss/len(valid_dataloader)
    valid_losses.append(avg_loss) 
    print(f"Validation Loss: {avg_loss:.4f}")

#df = pd.DataFrame(velocity, columns=['V_x, ])


delta_t = np.full(test_dataset.time.shape, fill_value=0.1)
traj_x_gt = np.cumsum(velocities[:, 0] * delta_t)
traj_y_gt = np.cumsum(velocities[:, 1] * delta_t)

traj_x_pred = np.cumsum(predictions[:, 0] * delta_t)
traj_y_pred = np.cumsum(predictions[:, 1] * delta_t)


plot_velocities(predictions, velocities)

plt.figure(figsize=(10, 6))
plt.plot(traj_x_pred, traj_y_pred, label='predicted')
plt.plot(traj_x_gt, traj_y_gt, label="GT traj", linestyle="--")
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.grid()
plt.savefig('trajectories.png')

def plot_losses(losses, title):
    plt.figure(figsize=(12, 6))
    plt.plot(losses, marker='o', linestyle='-', color='b')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel(title, fontsize=12)
    plt.title(f'{title}', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
    
plot_losses(train_losses, 'MSE loss')