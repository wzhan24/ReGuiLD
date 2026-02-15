import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def augment_rotations(data):
    augmented = []
    for cube in data:
        # Original3
        augmented.append(cube)
        # Rotate 90 along z axis
        augmented.append(np.rot90(cube, k=1, axes=(0, 1)))
        # Rotate 90 degrees along x axis (switch y and z)
        augmented.append(np.rot90(cube, k=1, axes=(1, 2)))
        augmented.append(np.rot90(np.rot90(cube, k=1, axes=(1, 2)), k=1, axes=(0, 2)))
        # Rotate 90 degrees along y axis (switch x and z)
        augmented.append(np.rot90(cube, k=1, axes=(0, 2)))
        augmented.append(np.rot90(np.rot90(cube, k=1, axes=(0, 2)), k=1, axes=(1, 2)))
    return np.array(augmented)

def fake_augment(data):
    augmented = []
    for cube in data:
        lengthx = cube.shape[0] // 2
        lengthy = cube.shape[1] // 2
        lengthz = cube.shape[2] // 2
        start_x = np.random.randint(0, cube.shape[0] - lengthx)
        start_y = np.random.randint(0, cube.shape[1] - lengthy)
        start_z = np.random.randint(0, cube.shape[2] - lengthz)
        rand = np.random.randint(0, 5)
        if rand == 0:
            scale = np.random.uniform(0.1, 0.3)
            copy_cube = np.copy(cube)+ np.random.normal(0, scale, cube.shape)
            copy_cube[copy_cube > 1.2] = 0
            copy_cube[copy_cube < -0.4] = 1
            copy_cube[copy_cube < 0.5] = 0
            copy_cube[copy_cube >= 0.5] = 1
            augmented.append(copy_cube)
        elif rand == 1:
            copy_cube = np.copy(cube)
            copy_cube[start_x:start_x + lengthx, start_y:start_y + lengthy, start_z:start_z + lengthz] = 0
            augmented.append(copy_cube)
        elif rand == 2:
            idx = np.random.randint(0, cube.shape[0])
            copy_cube = np.copy(cube)
            copy_cube[start_x:start_x + lengthx, start_y:start_y + lengthy, start_z:start_z + lengthz] = \
                data[idx][start_x:start_x + lengthx, start_y:start_y + lengthy, start_z:start_z + lengthz]
            augmented.append(copy_cube)
        else:
            scale = np.random.uniform(0.3, 0.6)
            copy_cube = np.copy(cube)
            copy_cube[start_x:start_x + lengthx, start_y:start_y + lengthy, start_z:start_z + lengthz] = \
                np.random.normal(0, scale, (lengthx, lengthy, lengthz))
            copy_cube[copy_cube > 0.5] = 1
            copy_cube[copy_cube <= 0.5] = 0
            augmented.append(copy_cube)
    return np.array(augmented)

class VoxelDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        voxel = self.data[idx]
        # Optionally convert to torch.Tensor if needed
        return torch.tensor(voxel, dtype=torch.float32)
    
class LatentDataset(Dataset):
    def __init__(self, latents):
        self.latents = latents

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        return torch.tensor(self.latents[idx], dtype=torch.float32)

def get_latents(loader, encoder, device):
    encoder.eval()
    latents_list = []
    with torch.no_grad():
        for batch in loader:
            if batch.ndim == 4:
                batch = batch.unsqueeze(1)
            batch = batch.to(device)
            latents = encoder(batch)
            latents_list.append(latents.cpu().numpy())
    latents_array = np.concatenate(latents_list, axis=0)
    return latents_array