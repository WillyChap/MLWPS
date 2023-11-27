import torch
from torch import nn
from math import sqrt

from einops import rearrange, reduce, repeat, pack, unpack
from einops.layers.torch import Rearrange


def load_positional_encoder(conf):
    pe_type = conf["model"]["positional_encoder"]
    
    if pe_type == "continuous":
        dim = conf["model"]["dim"]
        heads = conf["model"]["heads"]
        return ContinuousPositionBias(dim = dim // 4, heads = heads)

    if pe_type == "graph":
        dim = conf["model"]["dim"]
        max_frames = conf["model"]["frames"]
        max_width = conf["model"]["image_width"]
        max_height = conf["model"]["image_height"]
        num_nodes = conf["model"]["num_nodes"]
        num_edges = conf["model"]["num_edges"]
        return GraphPositionBias(dim, max_frames, max_width, max_height, num_nodes, num_edges)


class ContinuousPositionBias(nn.Module):
    """ from https://arxiv.org/abs/2111.09883 """

    def __init__(self, *, dim, heads, layers = 2):
        super().__init__()
        self.net = nn.ModuleList([])
        self.net.append(nn.Sequential(nn.Linear(2, dim), leaky_relu()))

        for _ in range(layers - 1):
            self.net.append(nn.Sequential(nn.Linear(dim, dim), leaky_relu()))

        self.net.append(nn.Linear(dim, heads))
        self.register_buffer('rel_pos', None, persistent = False)

    def forward(self, x):
        n, device = x.shape[-1], x.device
        fmap_size = int(sqrt(n))

        if not exists(self.rel_pos):
            pos = torch.arange(fmap_size, device = device)
            grid = torch.stack(torch.meshgrid(pos, pos, indexing = 'ij'))
            grid = rearrange(grid, 'c i j -> (i j) c')
            rel_pos = rearrange(grid, 'i c -> i 1 c') - rearrange(grid, 'j c -> 1 j c')
            rel_pos = torch.sign(rel_pos) * torch.log(rel_pos.abs() + 1)
            self.register_buffer('rel_pos', rel_pos, persistent = False)

        rel_pos = self.rel_pos.float()

        for layer in self.net:
            rel_pos = layer(rel_pos)

        bias = rearrange(rel_pos, 'i j h -> h i j')
        return x + bias


class GraphPositionBias(nn.Module):
    def __init__(self, d_model, max_frames, max_width, max_height, num_nodes, num_edges):
        super(LearnablePositionalEmbeddingWithGraph, self).__init__()
        self.d_model = d_model
        
        # Learnable positional embeddings for frames, latitude, and longitude
        self.frame_embedding = nn.Parameter(torch.randn(1, max_frames, d_model // 3))
        self.width_embedding = nn.Parameter(torch.randn(1, max_width, d_model // 3))
        self.height_embedding = nn.Parameter(torch.randn(1, max_height, d_model // 3))
        
        # Learnable graph embeddings
        self.node_embedding = nn.Parameter(torch.randn(1, num_nodes, d_model // 3))
        self.edge_embedding = nn.Parameter(torch.randn(1, num_edges, d_model // 3))
        
    def forward(self, x, edge_index):
        batch_size, _, frames, width, height = x.size()
        
        # Repeat the learnable positional embeddings to match the input batch size
        frame_embedding = self.frame_embedding.repeat(batch_size, 1, 1)
        width_embedding = self.width_embedding.repeat(batch_size, 1, 1)
        height_embedding = self.height_embedding.repeat(batch_size, 1, 1)
        
        # Combine the embeddings and add them to the input data
        x = x + frame_embedding.unsqueeze(-2).unsqueeze(-1) + width_embedding.unsqueeze(-2) + height_embedding.unsqueeze(-1)
        
        # Learnable graph embeddings
        node_embedding = self.node_embedding.repeat(batch_size, 1, 1)
        edge_embedding = self.edge_embedding.repeat(batch_size, 1, 1)
        
        # Apply graph embeddings
        graph_embedding = torch.matmul(node_embedding, edge_embedding.transpose(-1, -2))
        x = x + graph_embedding.unsqueeze(-2).unsqueeze(-1)
        
        # Normalize the embeddings
        x = x / torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        return x



def posemb_sincos_3d(patches, temperature = 10000, dtype = torch.float32):
    _, f, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    z, y, x = torch.meshgrid(
        torch.arange(f, device = device),
        torch.arange(h, device = device),
        torch.arange(w, device = device),
    indexing = 'ij')

    fourier_dim = dim // 6

    omega = torch.arange(fourier_dim, device = device) / (fourier_dim - 1)
    omega = 1. / (temperature ** omega)

    z = z.flatten()[:, None] * omega[None, :]
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :] 

    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos(), z.sin(), z.cos()), dim = 1)

    pe = F.pad(pe, (0, dim - (fourier_dim * 6))) # pad if feature dimension not cleanly divisible by 6
    return pe.type(dtype)