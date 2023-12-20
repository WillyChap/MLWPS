import torch
import torch.fft
from torch import nn
from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

import torch.nn.functional as F
from wpsml.pe import SurfacePosEmb2D, TokenizationAggregation
#from wpsml.visual_ssl import SimSiam, MLP
from vector_quantize_pytorch import VectorQuantize


class FeedForward(nn.Module):
    
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


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


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class PatchDropout(nn.Module):
    
    def __init__(self, prob):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob

    def forward(self, x):
        if not self.training or self.prob == 0.:
            return x

        b, n, _, device = *x.shape, x.device

        batch_indices = torch.arange(b, device = device)
        batch_indices = rearrange(batch_indices, '... -> ... 1')
        num_patches_keep = max(1, int(n * (1 - self.prob)))
        patch_indices_keep = torch.randn(b, n, device = device).topk(num_patches_keep, dim = -1).indices

        return x[batch_indices, patch_indices_keep]


class ViT2D(nn.Module):
    def __init__(
        self,
        image_height, 
        patch_height,
        image_width,
        patch_width,
        frames, 
        frame_patch_size,
        dim,
        channels = 67,
        surface_channels = 7,
        depth = 4,
        heads = 8,
        dim_head = 32,
        mlp_dim = 32, 
        dropout = 0.0,
        use_registers = False,
        num_register_tokens = 0,
        token_dropout = 0.0,
        use_codebook = False,
        vq_codebook_size = 128,
        vq_decay = 0.1,
        vq_commitment_weight = 1.0,
        vq_kmeans_init = True,
        vq_use_cosine_sim = True
    ):
                 
        super().__init__()
        
        self.channels = channels
        self.surface_channels = surface_channels
        self.frames = frames
        self.use_codebook = use_codebook

        # Encoder-decoder layers 
        self.transformer_encoder = Transformer(
            dim = dim,
            depth = depth,
            dim_head = dim_head,
            heads = heads,
            mlp_dim = mlp_dim,
            dropout = dropout
        )
        self.transformer_decoder = Transformer(
            dim = dim,
            depth = depth,
            dim_head = dim_head,
            heads = heads,
            mlp_dim = mlp_dim,
            dropout = dropout
        )
                
        # Input/output dimensions
        input_dim = (channels * frames + surface_channels) * patch_height * patch_width
        
        # Encoder layers
        self.encoder_embed = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
                                       p1=patch_height, 
                                       p2=patch_width) 
        
        self.encoder_linear = nn.Linear(input_dim, dim)
        self.encoder_layer_norm = nn.LayerNorm(dim)
        
        # Decoder layers                 
        self.decoder_linear_1 = nn.Linear(dim, dim * 4)
        self.decoder_linear_2 = nn.Linear(dim * 4, input_dim)
        self.decoder_layer_norm_1 = nn.LayerNorm(4 * dim)
        self.decoder_layer_norm_2 = nn.LayerNorm(input_dim)
        self.decoder_rearrange = Rearrange('b (h w) (p1 p2 c) -> b c (p1 h) (w p2)',  
                                                   h=(image_height // patch_height),
                                                   w=(image_width // patch_width),
                                                   p1=patch_height, 
                                                   p2=patch_width)
        
        # Positional embeddings
        # self.pos_embedding_enc = TokenizationAggregation(channels, patch_height, patch_width, dim)
        self.pos_embedding_enc = SurfacePosEmb2D(
            image_height, image_width, patch_height, patch_width, dim
        ) 
        self.pos_embedding_dec = SurfacePosEmb2D(
            image_height, image_width, patch_height, patch_width, dim
        ) 
        
        # Token / patch drop
        self.token_dropout = PatchDropout(token_dropout) if token_dropout > 0.0 else nn.Identity()

        # Vision Transformers Need Registers, https://arxiv.org/abs/2309.16588
        self.use_registers = use_registers
        if self.use_registers:
            self.register_tokens = nn.Parameter(torch.randn(num_register_tokens, dim))

        # codebook
        self.use_codebook = use_codebook
        if self.use_codebook:
            self.vq = VectorQuantize(
                dim = dim,
                codebook_size = vq_codebook_size,     # codebook size
                decay = vq_decay,             # the exponential moving average decay, lower means the dictionary will change faster
                commitment_weight = vq_commitment_weight, # the weight on the commitment loss
                vq_kmeans_init = vq_kmeans_init,
                vq_use_cosine_sim = vq_use_cosine_sim
                
            )
                                                   
    def encode(self, x):      
        # encode
        x = self.encoder_embed(x)
        x = self.encoder_linear(x)
        x = self.encoder_layer_norm(x)
        x = self.pos_embedding_enc(x)
        x = self.token_dropout(x)
        if self.use_registers:
            r = repeat(self.register_tokens, 'n d -> b n d', b = x.shape[0])
            x, ps = pack([x, r], 'b * d')
        x = self.transformer_encoder(x)
        if self.use_registers:
            x, _ = unpack(x, ps, 'b * d')
        
        return x
    
    def decode(self, x):
        x = self.pos_embedding_dec(x)
        if self.use_registers:
            r = repeat(self.register_tokens, 'n d -> b n d', b = x.shape[0])
            x, ps = pack([x, r], 'b * d')
        x = self.transformer_decoder(x)
        if self.use_registers:
            x, _ = unpack(x, ps, 'b * d')
        
        x = self.decoder_linear_1(x) 
        #x = F.tanh(x)
        x = self.decoder_layer_norm_1(x)
        x = self.decoder_linear_2(x)
        x = self.decoder_layer_norm_2(x)
        x = self.decoder_rearrange(x)
        return x
    
    def forward(self, x):
        z = self.encode(x)
            
        if self.use_codebook:
            z, indices, commit_loss = self.vq(z)
            x = self.decode(z)
            return x, commit_loss
            
        x = self.decode(z)    
        return x

    def concat_and_reshape(self, x1, x2):
        x1 = x1.view(x1.shape[0], -1, x1.shape[3], x1.shape[4])
        x_concat = torch.cat((x1, x2), dim=1)
        return x_concat

    def split_and_reshape(self, tensor):
        tensor1 = tensor[:, :int(self.channels*self.frames), :, :]
        tensor2 = tensor[:, -int(self.surface_channels):, :, :]
        tensor1 = tensor1.view(tensor1.shape[0], self.channels, self.frames, tensor1.shape[2], tensor1.shape[3])
        return tensor1, tensor2
    
    def codebook(self):
        if self.use_codebook:
            return self.vq.codebook
        return None



if __name__ == "__main__":
    image_height = 640 # 640
    patch_height = 64
    image_width = 1280  # 1280
    patch_width = 64
    frames = 15
    frame_patch_size = 3

    channels = 4
    surface_channels = 7
    dim = 64
    layers = 4
    dim_head = 30
    mlp_dim = 30
    heads = 8
    depth = 4

    input_tensor = torch.randn(1, channels, frames, image_height, image_width)
    input_tensor_surf = torch.randn(1, surface_channels, image_height, image_width)

    model = ViT2D(
        image_height, 
        patch_height,
        image_width,
        patch_width,
        frames, 
        frame_patch_size,
        dim,
        channels,
        surface_channels,
        depth,
        heads,
        dim_head,
        mlp_dim,
    )
    
    loss = model(input_tensor, input_tensor_surf, 
             input_tensor, input_tensor_surf, 
             return_loss=True)

    print("Loss:", loss)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters in the model: {num_params}")