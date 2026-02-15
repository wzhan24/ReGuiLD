import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt

class VoxEvaluation(nn.Module):
    def __init__(self, dataset):
        super(VoxEvaluation, self).__init__()
        self.dataset = dataset

    @staticmethod
    def iou_score(a, b, threshold=0.5):
        a_bin = a > threshold
        b_bin = b > threshold
        intersection = np.sum(a_bin & b_bin)
        union = np.sum(a_bin | b_bin)
        return intersection / (union + 1e-6)

    def symmetry_score(self, voxel):
        flipped_voxel = voxel.flip(dims=[0, 1, 2])
        symmetry = torch.sum((voxel * flipped_voxel) > 0).float() / (torch.sum(voxel).float() + 1e-6)
        return symmetry.item()

    def periodicity_score(self, voxel):
        x0, xl = voxel[0, :, :], voxel[-1, :, :]
        y0, yl = voxel[:, 0, :], voxel[:, -1, :]
        z0, zl = voxel[:, :, 0], voxel[:, :, -1]
        x_inter = torch.sum((x0 * xl) > 0).float()
        y_inter = torch.sum((y0 * yl) > 0).float()
        z_inter = torch.sum((z0 * zl) > 0).float()
        x_union = torch.sum((x0 + xl) > 0).float()
        y_union = torch.sum((y0 + yl) > 0).float()
        z_union = torch.sum((z0 + zl) > 0).float()
        score = (x_inter / (x_union + 1e-6) +
                 y_inter / (y_union + 1e-6) +
                 z_inter / (z_union + 1e-6)) / 3
        return score.item()

    def connection_score(self, voxel):
        def largest_component_size(v):
            visited = np.zeros(v.shape, dtype=bool)
            total_size = np.sum(v > 0)
            largest_size = 0
            for x in range(v.shape[0]):
                for y in range(v.shape[1]):
                    for z in range(v.shape[2]):
                        if v[x, y, z] > 0 and not visited[x, y, z]:
                            stack = [(x, y, z)]
                            visited[x, y, z] = True
                            comp_size = 1
                            while stack:
                                cx, cy, cz = stack.pop()
                                for dx, dy, dz in [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]:
                                    nx = (cx + dx) % v.shape[0]
                                    ny = (cy + dy) % v.shape[1]
                                    nz = (cz + dz) % v.shape[2]
                                    if v[nx, ny, nz] > 0 and not visited[nx, ny, nz]:
                                        visited[nx, ny, nz] = True
                                        stack.append((nx, ny, nz))
                                        comp_size += 1
                            largest_size = max(largest_size, comp_size)
            return largest_size / total_size if total_size > 0 else 0

        v = voxel.cpu().numpy()
        return largest_component_size(v)

    def novelty_score(self, voxel):
        v = voxel.cpu().numpy()
        ious = [self.iou_score(v, data.numpy()) for data in self.dataset]
        max_iou = max(ious)
        return 1 - max_iou

    def coverage_score(self, voxel):
        v = voxel.cpu().numpy()
        ious = [self.iou_score(v, data.numpy()) for data in self.dataset]
        best_idx = np.argmax(ious)
        return best_idx

    def forward(self, voxels, valid_threshold=0.5):
        # voxels: (batch_size, D, H, W)
        batch_size = len(voxels)
        symmetry_scores = []
        periodicity_scores = []
        connection_scores = []
        novelty_scores = []
        coverage_indices = []

        for i in tqdm(range(batch_size)):
            voxel = voxels[i]
            voxel = (voxel - torch.min(voxel)) / (torch.max(voxel) - torch.min(voxel) + 1e-6)
            voxel[voxel < 0.5] = 0
            voxel[voxel >= 0.5] = 1
            if torch.sum(torch.abs(voxel)) == 0:
                symmetry_scores.append(0)
                periodicity_scores.append(0)
                connection_scores.append(0)
                novelty_scores.append(0)
                coverage_indices.append(-1)
                continue
            symmetry_scores.append(self.symmetry_score(voxel))
            periodicity_scores.append(self.periodicity_score(voxel))
            connection_scores.append(self.connection_score(voxel))
            if symmetry_scores[-1] > valid_threshold and periodicity_scores[-1] > valid_threshold and connection_scores[-1] > valid_threshold:
                novelty_scores.append(self.novelty_score(voxel))
                coverage_indices.append(self.coverage_score(voxel))
            else:
                novelty_scores.append(0)
                coverage_indices.append(-1)

        # Coverage score: how many unique dataset samples are covered by batch
        coverage = len(set(coverage_indices)) / batch_size if batch_size > 0 else 0

        print(f"Average Symmetry Score: {np.mean(symmetry_scores):.4f}")
        print(f"Average Periodicity Score: {np.mean(periodicity_scores):.4f}")
        print(f"Average Connection Score: {np.mean(connection_scores):.4f}")
        print(f"Average Novelty Score: {np.mean(novelty_scores):.4f}")
        print(f"Batch Coverage Score: {coverage:.4f}")

        return symmetry_scores, periodicity_scores, connection_scores, novelty_scores, coverage
    
def visualizeVox(voxel, elev=30, azim=25):
    # Generate binary voxel data
    voxel_data = voxel > 0  # Only plot non-zero voxels

    # Create a 3D plot
    fig = plt.figure(dpi=200)
    ax = fig.add_subplot(111, projection='3d')

    # Get the coordinates of the voxels
    dimensions = voxel.shape
    x, y, z = np.indices(dimensions)

    # Use a more vivid light blue color for all voxels
    #light_blue = (0.6, 0.8, 1.0, 0.9)
    vivid_blue = (0.3, 0.75, 1.0, 0.95)  # RGBA, higher blue and alpha
    colors = np.empty(voxel_data.shape + (4,), dtype=float)
    colors[:] = vivid_blue

    # Plot the voxels with thinner edges
    ax.voxels(voxel_data, facecolors=colors, edgecolor=(0.2, 0.6, 1.0, 0.4))  # More blue edge

    # Set the elevation (elev) and azimuth (azim) angles of the plot
    ax.view_init(elev=elev, azim=azim)

    # Set labels
    ax.set_xlabel('voxel X')
    ax.set_ylabel('voxel Y')
    ax.set_zlabel('voxel Z')

    # Show the plot
    plt.show()