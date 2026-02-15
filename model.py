import torch
from torch import nn
from sklearn.cluster import KMeans
import numpy as np
import math

class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim, dropout=0.1):
        super(SimpleTransformer, self).__init__()
        self.input_fc = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_fc = nn.Linear(model_dim, output_dim)

    def forward(self, x):
        # x: (batch, N, input_dim)
        tokens = x
        tokens = self.input_fc(tokens)           # (batch, N, model_dim)
        tokens = tokens.transpose(0, 1)          # (N, batch, model_dim)

        # Apply transformer
        tokens = self.transformer_encoder(tokens)  # (N, batch, model_dim)
        tokens = tokens.transpose(0, 1)            # (batch, N, model_dim)

        out = self.output_fc(tokens)               # (batch, N, output_dim)
        return out

class TransformerEncoder(nn.Module):
    def __init__(self, patch_dim=512, patch_embed_dim=128, num_tokens=216, latent_dim=128, model_dim=128, num_heads=8, num_layers=4):
        super().__init__()
        self.patch_embed = nn.Linear(8*8*8, patch_dim)
        self.patch_reduce = nn.Sequential(
            nn.Linear(patch_dim, patch_embed_dim),
            nn.ReLU()
        )
        self.transformer = SimpleTransformer(
            input_dim=patch_embed_dim,
            model_dim=model_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            output_dim=patch_embed_dim
        )
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_tokens * patch_embed_dim, 1024),
            # nn.ReLU(),
            # nn.Linear(num_tokens * patch_embed_dim // 8, 1024),
            nn.ReLU(),
            nn.Linear(1024, latent_dim)
        )
        self.num_tokens = num_tokens
        self.patch_embed_dim = patch_embed_dim

    def get_positional_encoding(self, seq_len, dim, device):
        pe = torch.zeros(seq_len, dim, device=device)
        position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, device=device).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe  # (seq_len, dim)

    def forward(self, x):
        # x: (batch, 1, 48, 48, 48)
        batch_size = x.size(0)
        patches = x.unfold(2, 8, 8).unfold(3, 8, 8).unfold(4, 8, 8)  # (batch, 1, 6, 6, 6, 8, 8, 8)
        patches = patches.contiguous().view(batch_size, -1, 8, 8, 8)  # (batch, 216, 8, 8, 8)
        patches = patches.view(batch_size, 216, -1)  # (batch, 216, 512)
        tokens = self.patch_embed(patches)           # (batch, 216, 512)
        tokens = self.patch_reduce(tokens)           # (batch, 216, 128)
        tokens = tokens + 0.1 * self.get_positional_encoding(self.num_tokens, self.patch_embed_dim, tokens.device).unsqueeze(0)  # Add positional encoding
        tokens = self.transformer(tokens)            # (batch, 216, 128)
        latent = self.mlp(tokens)                    # (batch, latent_dim)
        return latent

class TransformerDecoder(nn.Module):
    def __init__(self, latent_dim=128, patch_embed_dim=128, patch_dim=512, num_tokens=216, model_dim=128, num_heads=8, num_layers=4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            # nn.ReLU(),
            # nn.Linear(1024, num_tokens * patch_embed_dim // 8),
            nn.ReLU(),
            nn.Linear(1024, num_tokens * patch_embed_dim)
        )
        self.transformer = SimpleTransformer(
            input_dim=patch_embed_dim,
            model_dim=model_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            output_dim=patch_embed_dim
        )
        self.patch_expand = nn.Sequential(
            nn.Linear(patch_embed_dim, patch_dim),
            nn.ReLU()
        )
        self.patch_proj = nn.Linear(patch_dim, 8*8*8)
        self.num_tokens = num_tokens
        self.patch_embed_dim = patch_embed_dim
        self.latent_dim = latent_dim

    def forward(self, z):
        # z: (batch, latent_dim)
        batch_size = z.size(0)
        tokens = self.mlp(z).view(batch_size, self.num_tokens, -1)   # (batch, 216, 128)
        tokens = self.transformer(tokens)                 # (batch, 216, 128)
        tokens = self.patch_expand(tokens)                # (batch, 216, 512)
        patches = self.patch_proj(tokens)                 # (batch, 216, 512) -> (batch, 216, 512)
        patches = patches.view(batch_size, self.num_tokens, 8, 8, 8)  # (batch, 216, 8, 8, 8)
        voxels = torch.zeros((batch_size, 1, 48, 48, 48), device=patches.device)
        idx = 0
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    voxels[:, 0, i*8:(i+1)*8, j*8:(j+1)*8, k*8:(k+1)*8] = patches[:, idx]
                    idx += 1
        return voxels
    
class RepelEnergy(nn.Module):
    def __init__(self):
        super(RepelEnergy, self).__init__()

    def pairwise_energy(self, latents_a, latents_b, weight=1.0):
        # latents_a: (N, D), latents_b: (M, D)
        # Compute pairwise distances
        dists = torch.cdist(latents_a, latents_b, p=2)  # (N, M)
        energy = weight * torch.sum(1.0 / (dists + 0.1))
        return energy/(dists.numel() + 1)

    def intra_energy(self, latents, weight=0.01):
        # Compute pairwise distances within latents
        dists = torch.cdist(latents, latents, p=2)  # (N, N)
        # Exclude diagonal (self-distances)
        mask = ~torch.eye(dists.size(0), dtype=torch.bool, device=dists.device)
        energy = weight * torch.sum(1.0 / (dists[mask] + 0.1))
        return energy/(mask.sum() + 1)

    def forward(self, posi_latents, nega_latents, inter_weight=1.0, intra_weight=1e-4):
        # Inter energy: between posi and nega
        inter_energy = self.pairwise_energy(posi_latents, nega_latents, weight=inter_weight)
        # Intra energy: within posi and within nega
        intra_energy = self.intra_energy(posi_latents, weight=intra_weight) + \
                       self.intra_energy(nega_latents, weight=intra_weight)
        total_energy = inter_energy + intra_energy
        return total_energy, inter_energy, intra_energy
    
class ResidualMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super(ResidualMLP, self).__init__()
        self.input_layer = nn.Linear(input_dim+1, hidden_dim)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim+1, hidden_dim) for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(hidden_dim+1, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x, t):
        x = torch.cat([x, t], dim=-1)
        x = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            residual = x
            x = torch.cat([x, t], dim=-1)
            x = self.activation(layer(x))
            x = x + residual  # Residual connection
        x = torch.cat([x, t], dim=-1)
        x = self.output_layer(x)
        return x

class Repellor:
    def __init__(self, latent_dataset, n_clusters=10):
        # latent_dataset: instance of LatentDataset
        self.latents = np.array([latent.numpy() for latent in latent_dataset])
        self.n_clusters = n_clusters
        self.centroids = None
        self.radii = None
        self.cluster_indices = None

    def cluster(self):
        # Cluster the latent vectors
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=10)
        labels = kmeans.fit_predict(self.latents)
        self.centroids = []
        self.radii = []
        self.cluster_indices = []
        for i in range(self.n_clusters):
            indices = np.where(labels == i)[0]
            cluster_points = self.latents[indices]
            centroid = np.mean(cluster_points, axis=0)
            radius = np.max(np.linalg.norm(cluster_points - centroid, axis=1))
            self.centroids.append(centroid)
            self.radii.append(radius)
            self.cluster_indices.append(indices)
        self.centroids = np.array(self.centroids)
        self.radii = np.array(self.radii)
        #print(self.radii)

    def get_clusters(self):
        return self.centroids, self.radii, self.cluster_indices
    
    def repel(self, x, threshold):
        """
        x: np.ndarray, shape (latent_dim,)
        threshold: float
        Returns: np.ndarray, shape (latent_dim,) - summed repelling force
        """
        threshold = np.max(self.radii)
        force = np.zeros_like(x)
        # Find which cluster x is in (within radius)
        for i in range(self.n_clusters):
            centroid = self.centroids[i]
            radius = self.radii[i]
            dist_to_centroid = np.linalg.norm(x - centroid)
            if dist_to_centroid <= radius:
                # x is in this cluster
                indices = self.cluster_indices[i]
                cluster_points = self.latents[indices]
                for neighbor in cluster_points:
                    dist = np.linalg.norm(x - neighbor)
                    if dist < threshold and dist > 1e-8:
                        direction = x - neighbor
                        direction = direction / (dist + 1e-3)
                        #force += direction / (dist**2 + 1e-3)
                        force -= direction/np.log(max(1 - dist / threshold, 1e-8))
                break  # Only consider the first cluster x is in
        return force