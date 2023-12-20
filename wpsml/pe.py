import torch
from torch import nn
import torch.nn.functional as F
from math import sqrt

from einops import rearrange, reduce, repeat, pack, unpack
from einops.layers.torch import Rearrange


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        # dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # attn = self.attend(dots)
        # attn = self.dropout(attn)

        # out = torch.matmul(attn, v)

        with torch.backends.cuda.sdp_kernel(
            enable_flash=True, 
            enable_math=False, 
            enable_mem_efficient=True
        ):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask = None,
                dropout_p = 0.0
            )

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


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

    def __init__(self, *, dim, heads, layers = 2, p = 0.2):
        super().__init__()
        self.net = nn.ModuleList([])
        self.net.append(nn.Sequential(nn.Linear(2, dim), nn.LeakyReLU(p)))

        for _ in range(layers - 1):
            self.net.append(nn.Sequential(nn.Linear(dim, dim), nn.LeakyReLU(p)))

        self.net.append(nn.Linear(dim, heads))
        self.register_buffer('rel_pos', None, persistent = False)

    def forward(self, x):
        n, device = x.shape[-1], x.device
        fmap_size = int(sqrt(n))
        
        if not (self.rel_pos is not None):
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
        super(GraphPositionBias, self).__init__()
        self.d_model = d_model
        
        # Learnable positional embeddings for frames, latitude, and longitude
        self.frame_embedding = nn.Parameter(torch.randn(1, max_frames, d_model // 3))
        self.width_embedding = nn.Parameter(torch.randn(1, max_width, d_model // 3))
        self.height_embedding = nn.Parameter(torch.randn(1, max_height, d_model // 3))
        
        # Learnable graph embeddings
        self.node_embedding = nn.Parameter(torch.randn(1, num_nodes, d_model // 3))
        self.edge_embedding = nn.Parameter(torch.randn(1, num_edges, d_model // 3))
        
    def forward(self, x):
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


class TokenizationAggregation(nn.Module):
    def __init__(self, channels, patch_height, patch_width, emb_dim):
        super(TokenizationAggregation, self).__init__()

        self.patch_height = patch_height
        self.patch_width = patch_width

        self.tokenization = nn.Linear(patch_height * patch_width, emb_dim)
        self.aggregation = Attention(emb_dim)
        self.query = nn.Parameter(torch.randn(emb_dim))

    def forward(self, x):
        B, V, H, W = x.shape

        token_height = H // self.patch_height
        token_width = W // self.patch_width

        x = x.unfold(2, self.patch_height, self.patch_height).unfold(3, self.patch_width, self.patch_width)
        x = x.contiguous().view(B, V, token_height, token_width, -1)

        x = x.permute(0, 2, 3, 1, 4)  # Adjust permutation
        x = x.reshape(B, token_height, token_width, V, -1)

        # Linearly embed each variable independently
        x = self.tokenization(x)
        
        # Reshape to include token dimensions
        x = x.reshape(B, token_height, token_width, V, -1)
        
        B, N, M, V, D = x.shape
        
        # Average over the color channels
        x = x.mean(dim=3)

        # Flatten the token and variable dimensions for the attention mechanism
        x = x.reshape(B, N * M, D)

        # Attention mechanism using the Attention class
        x = self.aggregation(x)

        return x
    

class PosEmb3D(nn.Module):
    def __init__(self, frames, image_height, image_width, frame_patch_size, patch_height, patch_width, dim, temperature=10000):
        super(PosEmb3D, self).__init__()
        z, y, x = torch.meshgrid(
            torch.arange(frames // frame_patch_size),
            torch.arange(image_height // patch_height),
            torch.arange(image_width // patch_width),
            indexing='ij'
        )

        fourier_dim = dim // 6
        omega = torch.arange(fourier_dim) / (fourier_dim - 1)
        omega = 1. / (temperature ** omega)

        z = z.flatten()[:, None] * omega[None, :]
        y = y.flatten()[:, None] * omega[None, :]
        x = x.flatten()[:, None] * omega[None, :]

        pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos(), z.sin(), z.cos()), dim=1)
        pe = F.pad(pe, (0, dim - (fourier_dim * 6)))  # pad if feature dimension not cleanly divisible by 6
        self.embedding = pe

    def forward(self, x):
        return x + self.embedding.to(dtype=x.dtype, device=x.device)


class SurfacePosEmb2D(nn.Module):
    def __init__(self, image_height, image_width, patch_height, patch_width, dim, temperature=10000):
        super(SurfacePosEmb2D, self).__init__()
        y, x = torch.meshgrid(
            torch.arange(image_height // patch_height),
            torch.arange(image_width // patch_width),
            indexing="ij"
        )

        assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
        omega = torch.arange(dim // 4) / (dim // 4 - 1)
        omega = 1.0 / (temperature ** omega)

        y = y.flatten()[:, None] * omega[None, :]
        x = x.flatten()[:, None] * omega[None, :]
        pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
        self.embedding = pe

    def forward(self, x):
        return x + self.embedding.to(dtype=x.dtype, device=x.device)


class CombinedPosEmb(nn.Module):
    def __init__(self, frames, image_height, image_width, frame_patch_size, patch_height, patch_width, dim, temperature=10000):
        super(CombinedPosEmb, self).__init__()
        
        self.num_patches = (image_height // patch_height) * (image_width // patch_width) * (frames // frame_patch_size)  
        self.num_patches_surf = (image_height // patch_height) * (image_width // patch_width)
        
        # Initialize the 3D position embedding
        self.pos_emb_3d = PosEmb3D(
            frames, image_height, image_width, frame_patch_size, patch_height, patch_width, dim, temperature
        )
        
        # Initialize the 2D position embedding
        self.pos_emb_2d = SurfacePosEmb2D(
            image_height, image_width, patch_height, patch_width, dim, temperature
        )

    def forward(self, x):
        # Split the input back into x and x_surf
        x, x_surf = torch.split(x, [x.size(1) - self.num_patches_surf, self.num_patches_surf], dim=1)

        # Apply the 3D position embedding to x and the 2D position embedding to x_surf
        x = self.pos_emb_3d(x)
        x_surf = self.pos_emb_2d(x_surf)

        # Concatenate x and x_surf along the appropriate dimension
        x_combined = torch.cat((x, x_surf), dim=1)

        return x_combined
