from ronin_resnet import ResNet1D, BasicBlock1D, FCOutputModule
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from spline_dataset.spline_diff import generate_imu_data
from spline_dataset.spline_generation import generate_batch_of_splines
from spline_dataset.spline_dataloader import Spline_2D_Dataset
import os
import pickle

import matplotlib.pyplot as plt

model = ResNet1D(
    num_inputs=3,            # IMU features: 3 channels
    num_outputs=2,           # Velocity outputs: vx, vy
    block_type=BasicBlock1D, # Basic ResNet block
    group_sizes=[2, 2, 2],   # Adjusted number of residual groups
    base_plane=64,
    output_block=FCOutputModule,  # Use FCOutputModule
    kernel_size=3,
    fc_dim=512,              # Fully connected dimensions
    in_dim=1,              # Adjusted for 100 time steps
    dropout=0.5,
    trans_planes=128
)

#model = model.to(device)

criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=1e-4)


output_path = './out/'

if not os.path.exists(output_path):
    os.makedirs(output_path,exist_ok=True)
path_to_splines = output_path + 'splines/'

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

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, valid_dataset = random_split(dataset, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=16, shuffle=False)

print(f"Train size: {len(train_dataset)}, Test size: {len(valid_dataset)}")

# for X, y in train_dataloader:
#     print(X.shape)
#     print(y.shape)    

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0

    for imu, velocity in train_dataloader:
        imu, velocity = imu, velocity

        predictions = model(imu)

        loss = criterion(predictions, velocity)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_dataloader):.4f}")

model.eval()
total_loss = 0.0

predictions = []
velocity = []
with torch.no_grad():
    for imu, velocity in valid_dataloader:
        imu, velocity = imu, velocity

        predictions = model(imu)

        loss = criterion(predictions, velocity)
        total_loss += loss.item()

    print(f"Validation Loss: {total_loss/len(valid_dataloader):.4f}")


predictions = predictions.cpu().detach().numpy()
velocity = velocity.cpu().numpy()

plt.figure(figsize=(12, 6))
plt.plot(predictions[:, 0], label="Predicted V_x", marker='o', linestyle="--", color='blue')
plt.plot(predictions[:, 1], label="Predicted V_y", marker='o', linestyle="--", color='orange')
plt.plot(velocity[:, 0], label="Ground Truth V_x", marker='x', color='blue')
plt.plot(velocity[:, 1], label="Ground Truth V_y", marker='x', color='orange')
plt.legend()
plt.show()
plt.savefig('velocity.png')