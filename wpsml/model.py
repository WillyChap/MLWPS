import math
import torch
import torch.fft
from torch import nn
from math import sqrt
from vector_quantize_pytorch import LFQ
from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

import torch.nn.functional as F
from torch.autograd import grad as torch_grad
import torchvision
from wpsml.pe import PosEmb3D, SurfacePosEmb2D, CombinedPosEmb
from wpsml.visual_ssl import SimSiam, MLP
from wpsml.loss import SpectralLoss, SpectralLossSurface, MSLELoss, WeightedMSELoss, WeightedMSELossSurface
from functools import partial
from vector_quantize_pytorch import VectorQuantize


class SimpleModel(nn.Module):
    def __init__(self, color_dim, surface_dim):
        super(SimpleModel, self).__init__()

        # Shared layers
        self.conv = nn.Conv3d(color_dim, 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

        # Color prediction head
        self.conv_transpose_color = nn.ConvTranspose3d(64, color_dim, kernel_size=3, stride=1, padding=1)
        self.loss_fn_color = nn.MSELoss()

        # Surface detail prediction head
        self.conv_transpose_surface = nn.ConvTranspose2d(1, surface_dim, kernel_size=3, stride=1, padding=1)  # Using ConvTranspose2d for 2D prediction
        self.loss_fn_surface = nn.MSELoss()

    def forward(self, x, x_surface, y, y_surface):
        # Process 3D input
        x = self.conv(x)
        x = self.relu(x)
        x = self.conv_transpose_color(x)
        loss = self.loss_fn_color(x, y)

        # Process 2D input
        x_surface = self.conv_transpose_surface(x_surface)  # Add channel dimension
        x_surface = x_surface.squeeze(1)  # Remove channel dimension after prediction
        loss_surface = self.loss_fn_surface(x_surface, y_surface)

        return x, x_surface, loss+loss_surface



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
        inner_dim = dim_head *  heads
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


class TokenDropout(nn.Module):
    
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


class ViTEncDecSurface(nn.Module):
    def __init__(
        self,
        image_height, 
        patch_height,
        image_width,
        patch_width,
        frames, 
        frame_patch_size,
        dim,
        channels = 4,
        surface_channels = 7,
        depth = 4,
        heads = 8,
        dim_head = 32,
        mlp_dim = 32, 
        use_registers = False,
        num_register_tokens = 0,
        token_dropout = 0.0,
        use_codebook = False,
        vq_codebook_size = 128,
        vq_decay = 0.1,
        vq_commitment_weight = 1.0
    ):
                 
        super().__init__()

        # Encoder-decoder layers 
        self.transformer_encoder = Transformer(
            dim = dim,
            depth = depth,
            dim_head = dim_head,
            heads = heads,
            mlp_dim = mlp_dim
        )
        self.transformer_encoder_surf = Transformer(
            dim = dim,
            depth = depth,
            dim_head = dim_head,
            heads = heads,
            mlp_dim = mlp_dim
        )
        self.transformer_decoder = Transformer(
            dim = dim,
            depth = depth,
            dim_head = dim_head,
            heads = heads,
            mlp_dim = mlp_dim
        )
        self.transformer_decoder_surf = Transformer(
            dim = dim,
            depth = depth,
            dim_head = dim_head,
            heads = heads,
            mlp_dim = mlp_dim
        )
        
        # Input/output dimensions
        num_patches = (image_height // patch_height) * (image_width // patch_width) * (frames // frame_patch_size)  
        input_dim = channels * patch_height * patch_width * frame_patch_size
        input_dim_surface = surface_channels * patch_height * patch_width
        
        # Encoder layers 
        self.encoder_embed = Rearrange('b c (f pf) (h p1) (w p2) -> b (f h w) (p1 p2 pf c)', p1=patch_height, p2=patch_width, pf=frame_patch_size)
        self.encoder_linear = nn.Linear(input_dim, dim)
        
        self.encoder_layer_norm = nn.LayerNorm(dim)
        self.encoder_surface_layer_norm = nn.LayerNorm(dim)
        
        self.encoder_surface_embed = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width, c=surface_channels) 
        self.encoder_surface_linear = nn.Linear(input_dim_surface, dim)
        
        # Decoder layers
        self.decoder_linear_1 = nn.Linear(dim, dim * 4)
        self.decoder_linear_2 = nn.Linear(dim * 4, input_dim)
        self.decoder_rearrange = Rearrange('b (f h w) (p1 p2 pf c) -> b c (pf f) (p1 h) (w p2)',  
                                           h=(image_height // patch_height), f=(frames // frame_patch_size),  
                                           p1=patch_height, p2=patch_width, pf=frame_patch_size)
        
        self.decoder_layer_norm = nn.LayerNorm(4 * dim)
        self.decoder_surface_layer_norm = nn.LayerNorm(4 * dim)
                                           
        self.decoder_surface_linear_1 = nn.Linear(dim, dim * 4)
        self.decoder_surface_linear_2 = nn.Linear(dim * 4, input_dim_surface)
        self.decoder_surface_rearrange = Rearrange('b (h w) (p1 p2 c) -> b c (p1 h) (w p2)',  
                                                   w=(image_width // patch_width), c=surface_channels,  
                                                   p1=patch_height, p2=patch_width)

        # Positional embeddings
        self.pos_embedding = PosEmb3D(
            frames, image_height, image_width, frame_patch_size, patch_height, patch_width, dim
        )

        self.surface_pos_embedding = SurfacePosEmb2D(
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
        if  self.use_codebook:
            self.vq = VectorQuantize(
                dim = dim,
                codebook_size = vq_codebook_size,     # codebook size
                decay = vq_decay,             # the exponential moving average decay, lower means the dictionary will change faster
                commitment_weight = vq_commitment_weight  # the weight on the commitment loss
            )
            self.vq_surf = VectorQuantize(
                dim = dim,
                codebook_size = vq_codebook_size,     # codebook size
                decay = vq_decay,             # the exponential moving average decay, lower means the dictionary will change faster
                commitment_weight = vq_commitment_weight  # the weight on the commitment loss
            )
                                                   
    def encode(self, x, x_surf):
        x = self.encoder_embed(x)
        x = self.encoder_linear(x)
        x = self.encoder_layer_norm(x)
        x = self.pos_embedding(x)
        x = self.token_dropout(x)
        if self.use_registers:
            r = repeat(self.register_tokens, 'n d -> b n d', b = x.shape[0])
            x, ps = pack([x, r], 'b * d')
        x = self.transformer_encoder(x)
        if self.use_registers:
            x, _ = unpack(x, ps, 'b * d')
        
        x_surf = self.encoder_surface_embed(x_surf)
        x_surf = self.encoder_surface_linear(x_surf)
        x_surf = self.encoder_surface_layer_norm(x_surf)
        x_surf = self.surface_pos_embedding(x_surf)
        x_surf = self.token_dropout(x_surf)
        if self.use_registers:
            r = repeat(self.register_tokens, 'n d -> b n d', b = x_surf.shape[0])
            x_surf, ps = pack([x_surf, r], 'b * d')
        x_surf = self.transformer_encoder_surf(x_surf)
        if self.use_registers:
            x_surf, _ = unpack(x_surf, ps, 'b * d')
        
        return x, x_surf
    
    def decode(self, x, x_surf):
        x = self.pos_embedding(x)
        if self.use_registers:
            r = repeat(self.register_tokens, 'n d -> b n d', b = x.shape[0])
            x, ps = pack([x, r], 'b * d')
        x = self.transformer_decoder(x)
        if self.use_registers:
            x, _ = unpack(x, ps, 'b * d')
        
        x = self.decoder_linear_1(x) 
        #x = F.tanh(x)
        x = self.decoder_layer_norm(x)
        x = self.decoder_linear_2(x)
        x = self.decoder_rearrange(x)

        x_surf = self.surface_pos_embedding(x_surf)
        if self.use_registers:
            r = repeat(self.register_tokens, 'n d -> b n d', b = x_surf.shape[0])
            x_surf, ps = pack([x_surf, r], 'b * d')
        x_surf = self.transformer_decoder_surf(x_surf)
        if self.use_registers:
            x_surf, _ = unpack(x_surf, ps, 'b * d')
        
        x_surf = self.decoder_surface_linear_1(x_surf)
        #x_surf = F.tanh(x_surf)
        x_surf = self.decoder_surface_layer_norm(x_surf)
        x_surf = self.decoder_surface_linear_2(x_surf)
        x_surf = self.decoder_surface_rearrange(x_surf)
        
        return x, x_surf
    
    
class CombinedViT(nn.Module):
    def __init__(
        self,
        image_height, 
        patch_height,
        image_width,
        patch_width,
        frames, 
        frame_patch_size,
        dim,
        channels = 4,
        surface_channels = 7,
        depth = 4,
        heads = 8,
        dim_head = 32,
        mlp_dim = 32, 
        use_registers = False,
        num_register_tokens = 0,
        token_dropout = 0.0,
        use_codebook = False,
        vq_codebook_size = 128,
        vq_decay = 0.1,
        vq_commitment_weight = 1.0
    ):
                 
        super().__init__()

        # Encoder-decoder layers 
        self.transformer_encoder = Transformer(
            dim = dim,
            depth = depth,
            dim_head = dim_head,
            heads = heads,
            mlp_dim = mlp_dim
        )
        self.transformer_decoder = Transformer(
            dim = dim,
            depth = depth,
            dim_head = dim_head,
            heads = heads,
            mlp_dim = mlp_dim
        )
        
        # Input/output dimensions
        self.num_patches = (image_height // patch_height) * (image_width // patch_width) * (frames // frame_patch_size)  
        self.num_patches_surf = (image_height // patch_height) * (image_width // patch_width)
        input_dim = channels * patch_height * patch_width * frame_patch_size
        input_dim_surface = surface_channels * patch_height * patch_width
        
        # Encoder layers 
        self.encoder_embed = Rearrange('b c (f pf) (h p1) (w p2) -> b (f h w) (p1 p2 pf c)', p1=patch_height, p2=patch_width, pf=frame_patch_size)
        self.encoder_linear = nn.Linear(input_dim, dim)
        self.encoder_layer_norm = nn.LayerNorm(dim)
        self.encoder_layer_norm_cat = nn.LayerNorm(dim)
        
        self.encoder_surface_embed = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width, c=surface_channels) 
        self.encoder_surface_linear = nn.Linear(input_dim_surface, dim)
        self.encoder_surface_layer_norm = nn.LayerNorm(dim)
        
        # Decoder layers
        self.decoder_linear_1 = nn.Linear(dim, dim * 4)
        self.decoder_linear_2 = nn.Linear(dim * 4, input_dim)
        self.decoder_layer_norm_1 = nn.LayerNorm(4 * dim)
        self.decoder_layer_norm_2 = nn.LayerNorm(input_dim)
        self.decoder_rearrange = Rearrange('b (f h w) (p1 p2 pf c) -> b c (pf f) (p1 h) (w p2)',  
                                           h=(image_height // patch_height), f=(frames // frame_patch_size),  
                                           p1=patch_height, p2=patch_width, pf=frame_patch_size)
        
                                           
        self.decoder_surface_linear_1 = nn.Linear(dim, dim * 4)
        self.decoder_surface_linear_2 = nn.Linear(dim * 4, input_dim_surface)
        self.decoder_surface_layer_norm_1 = nn.LayerNorm(4 * dim)
        self.decoder_surface_layer_norm_2 = nn.LayerNorm(input_dim_surface)
        self.decoder_surface_rearrange = Rearrange('b (h w) (p1 p2 c) -> b c (p1 h) (w p2)',  
                                                   w=(image_width // patch_width), c=surface_channels,  
                                                   p1=patch_height, p2=patch_width)

        # Positional embeddings
        self.pos_embedding = PosEmb3D(
            frames, image_height, image_width, frame_patch_size, patch_height, patch_width, dim
        )

        self.surface_pos_embedding = SurfacePosEmb2D(
            image_height, image_width, patch_height, patch_width, dim
        )
        # Decoder PE
        self.decoder_pos_embedding = CombinedPosEmb(
            frames, image_height, image_width, frame_patch_size, patch_height, patch_width, dim
        )
        
        # Token / patch drop
        self.token_dropout = PatchDropout(token_dropout) if token_dropout > 0.0 else nn.Identity()

        # Vision Transformers Need Registers, https://arxiv.org/abs/2309.16588
        self.use_registers = use_registers
        if self.use_registers:
            self.register_tokens = nn.Parameter(torch.randn(num_register_tokens, dim))

        # codebook
        self.use_codebook = use_codebook
        if  self.use_codebook:
            self.vq = VectorQuantize(
                dim = dim,
                codebook_size = vq_codebook_size,     # codebook size
                decay = vq_decay,             # the exponential moving average decay, lower means the dictionary will change faster
                commitment_weight = vq_commitment_weight  # the weight on the commitment loss
            )
 

    def encode(self, x, x_surf):
        x = self.encoder_embed(x)
        x = self.encoder_linear(x)
        x = self.encoder_layer_norm(x)
        x = self.pos_embedding(x)

        x_surf = self.encoder_surface_embed(x_surf)
        x_surf = self.encoder_surface_linear(x_surf)
        x_surf = self.encoder_surface_layer_norm(x_surf)
        x_surf = self.surface_pos_embedding(x_surf)
        
        # Concatenate x and x_surf along the sequence dimension then normalize
        x = torch.cat((x, x_surf), dim=1)
        x = self.encoder_layer_norm_cat(x)
        
        # Attend
        x = self.token_dropout(x)
        if self.use_registers:
            r = repeat(self.register_tokens, 'n d -> b n d', b = x.shape[0])
            x, ps = pack([x, r], 'b * d')
        x = self.transformer_encoder(x)
        if self.use_registers:
            x, _ = unpack(x, ps, 'b * d')
        
        return x
        
    def decode(self, x):
        x = self.decoder_pos_embedding(x)

        # Attend
        if self.use_registers:
            r = repeat(self.register_tokens, 'n d -> b n d', b = x.shape[0])
            x, ps = pack([x, r], 'b * d')
        x = self.transformer_decoder(x)
        if self.use_registers:
            x, _ = unpack(x, ps, 'b * d')

        # Split the output back into x and x_surf
        x, x_surf = torch.split(x, [self.num_patches, self.num_patches_surf], dim=1)

        # Now reshape the two terms
        # x
        x = self.decoder_linear_1(x) 
        x = self.decoder_layer_norm_1(x)
        x = self.decoder_linear_2(x)
        x = self.decoder_layer_norm_2(x)
        x = self.decoder_rearrange(x)

        # x_surf
        x_surf = self.decoder_surface_linear_1(x_surf)
        x_surf = self.decoder_surface_layer_norm_1(x_surf)
        x_surf = self.decoder_surface_linear_2(x_surf)
        x_surf = self.decoder_surface_layer_norm_2(x_surf)
        x_surf = self.decoder_surface_rearrange(x_surf)

        return x, x_surf
    

class ViTEncoderDecoder(nn.Module):
    def __init__(
        self,
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
        rk4_integration=True,
        use_registers = False,
        num_register_tokens = 0,
        token_dropout = 0.0,
        use_codebook = False,
        vq_codebook_size = 128,
        vq_decay = 0.1,
        vq_commitment_weight = 1.0,
        use_vgg = False,
        use_visual_ssl = True,
        visual_ssl_weight = 0.05,
        use_spectral_loss=True,
        spectral_wavenum_init=20,
        spectral_lambda_reg=1.0,
        l2_recon_loss = False,
        use_hinge_loss = True,
        **kwargs
    ):
        super().__init__()
        
        self.channels = channels
        
        self.enc_dec = CombinedViT( #ViTEncDecSurface(
            image_height, 
            patch_height, 
            image_width,
            patch_width,
            frames, 
            frame_patch_size,
            dim,
            channels=channels,
            surface_channels=surface_channels,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            use_registers = use_registers,
            num_register_tokens = num_register_tokens,
            token_dropout = token_dropout,
            use_codebook = use_codebook,
            vq_codebook_size = vq_codebook_size,
            vq_decay = vq_decay,
            vq_commitment_weight = vq_commitment_weight
        )
        
        # integration type
        self.rk4_integration = rk4_integration

        # reconstruction loss
        if l2_recon_loss:
            self.recon_loss = nn.MSELoss(reduction='none')
            self.recon_loss_surf = nn.MSELoss(reduction='none')
        else:
            self.recon_loss = F.l1_loss
            self.recon_loss_surf = F.l1_loss
            
        #self.recon_loss = F.mse_loss if l2_recon_loss else F.l1_loss
        #self.recon_loss_surf = F.mse_loss if l2_recon_loss else F.l1_loss

        # ssl -- makes more sense to move this to the model class above which contains all layers
        self.visual_ssl = None
        self.visual_ssl_weight = visual_ssl_weight
        if use_visual_ssl:
            ssl_type = partial(SimSiam, 
                           channels = channels, 
                           surface_channels = surface_channels, 
                           device = next(self.enc_dec.parameters()).device)
            
            self.visual_ssl = ssl_type(
                self.enc_dec.encode,
                image_height = image_height,
                image_width = image_width,
                hidden_layer = -1
            )

        # perceptual loss -- possibly the same here
        self.use_vgg = use_vgg
        if self.use_vgg:
            self.vgg = torchvision.models.vgg16(pretrained = True)
            self.vgg.features[0] = torch.nn.Conv2d(channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.vgg.classifier = nn.Sequential(*self.vgg.classifier[:-2])
            
            # Freeze the weights of the pre-trained layers
            for param in self.vgg.parameters():
                param.requires_grad = False

        # spectral loss
        self.use_spectral_loss = use_spectral_loss
        self.spectral_lambda_reg = spectral_lambda_reg if self.use_spectral_loss else 1.0
        if self.use_spectral_loss:
            self.spectral_loss = SpectralLoss(wavenum_init=spectral_wavenum_init)
            self.spectral_loss_surface = SpectralLossSurface(wavenum_init=spectral_wavenum_init)
            

    def encode(self, fmap, fmap_surface):
        return self.enc_dec.encode(fmap, fmap_surface)

    def decode(self, z):
        
        if self.enc_dec.use_codebook:
            fmap, indices, commit_loss = self.enc_dec.vq(z)
            #fmap_surface, indices_surface, commit_loss_surface = self.enc_dec.vq_surf(fmap_surface)
            fmap, fmap_surface = self.enc_dec.decode(z)
            return fmap, fmap_surface, commit_loss
        
        fmap, fmap_surface = self.enc_dec.decode(z)

        return fmap, fmap_surface
    
    def forward(
        self,
        img,
        img_surface,
        y_img = False,
        y_img_surface = False,
        tendency_scaler = None,
        atmosphere_weights = None,
        surface_weights = None,
        return_loss = False,
        return_recons = False,
        return_ssl_loss = False
    ):
        batch, channels, frames, height, width, device = *img.shape, img.device

        # ssl loss (only using ERA5, not model predictions)
        
        if return_ssl_loss and self.visual_ssl is not None:
            return self.visual_ssl(img, img_surface) * self.visual_ssl_weight

        # autoencoder

        if self.rk4_integration:
            
            # RK4 steps for encoded result
            
            k1, k1_surf = self.decode(self.encode(img, img_surface))
            k2, k2_surf = self.decode(self.encode(img + 0.5 * k1, img_surface + 0.5 * k1_surf))
            k3, k3_surf = self.decode(self.encode(img + 0.5 * k2, img_surface + 0.5 * k2_surf))
            k4, k4_surf = self.decode(self.encode(img + k3, img_surface + k3_surf))
            
            # should be z-scores of tendencies
            
            pred_tendency = (k1 + 2 * k2 + 2 * k3 + k4) / 6
            pred_tendency_surf = (k1_surf + 2 * k2_surf + 2 * k3_surf + k4_surf) / 6
            
            # Undo the z-score output (tendency) of the model
            
            unscaled_tend, unscaled_tend_surf = tendency_scaler.inverse_transform(
                pred_tendency,
                pred_tendency_surf
            )
            
            # Step
            
            fmap = img + unscaled_tend
            fmap_surface = img_surface + unscaled_tend_surf

        else:
            z = self.encode(img, img_surface)
            
            if self.enc_dec.use_codebook:
                fmap, fmap_surface, cm_loss = self.decode(z)
            else:
                fmap, fmap_surface = self.decode(z)
                
            pred_tendency, pred_tendency_surf = tendency_scaler.transform(fmap-img, fmap_surface-img_surface)
            
            
        if not return_loss:
            return fmap, fmap_surface
        
        
        # Convert true tendencies to z-scores
        
        true_tendency, true_tendency_surf = tendency_scaler.transform(y_img-img, y_img_surface-img_surface)
        
        # reconstruction loss
        
        if atmosphere_weights is not None and surface_weights is not None:
            recon_loss = (self.recon_loss(true_tendency, pred_tendency) * atmosphere_weights).mean()
            recon_loss_surface = (self.recon_loss_surf(true_tendency_surf, pred_tendency_surf) * surface_weights).mean()
        
        else:
            recon_loss = self.recon_loss(true_tendency, pred_tendency).mean()
            recon_loss_surface = self.recon_loss_surf(true_tendency_surf, pred_tendency_surf).mean()

        # fourier spectral loss
        
        spec_loss = 0.0
        if self.use_spectral_loss:
            spec_loss_1 = self.spectral_loss(true_tendency, pred_tendency)
            spec_loss_2 = self.spectral_loss_surface(true_tendency_surf, pred_tendency_surf)
            spec_loss = 0.5 * (spec_loss_1 + spec_loss_2)
        
        # Add terms
        
        loss = self.spectral_lambda_reg * (recon_loss + recon_loss_surface) + (1 - self.spectral_lambda_reg) * spec_loss
        
        if self.enc_dec.use_codebook:
            loss += cm_loss.squeeze()

        if return_recons:
            return fmap, fmap_surface, loss

        return loss
    
    
    
#         # reconstruction loss
        
#         recon_loss = self.recon_loss_fn(y_img, fmap)
#         recon_loss_surface = self.recon_loss_fn(y_img_surface, fmap_surface)

#         # fourier spectral loss
        
#         spec_loss = 0.0
#         if self.use_spectral_loss:
#             spec_loss_1 = self.spectral_loss(y_img, fmap)
#             spec_loss_2 = self.spectral_loss_surface(y_img_surface, fmap_surface)
#             spec_loss = 0.5 * (spec_loss_1 + spec_loss_2)
        
#         # Add terms
        
#         loss = self.spectral_lambda_reg * (recon_loss + recon_loss_surface) + (1 - self.spectral_lambda_reg) * spec_loss
        
#         if self.enc_dec.use_codebook:
#             loss += (cm_loss + cm_loss_surf).squeeze()

#         if return_recons:
#             return fmap + img, fmap_surface + img_surface, loss

#         return loss



if __name__ == "__main__":
    image_height = 640  # 640
    patch_height = 64
    image_width = 1280  # 1280
    patch_width = 64
    frames = 16
    frame_patch_size = 4

    channels = 5
    dim = 128
    layers = 4
    dim_head = 30
    mlp_dim = 30
    heads = 8
    depth = 4

    vq_codebook_dim = 128
    vq_codebook_size = 2 ** 16
    vq_entropy_loss_weight = 0.8
    vq_diversity_gamma = 1.
    discr_layers = 4

    input_tensor = torch.randn(1, channels, frames, image_height, image_width).to("cuda")

    model = VQGanVAE(
        image_height,
        patch_height,
        image_width,
        patch_width,
        frames,
        frame_patch_size,
        dim,
        channels,
        depth,
        heads,
        dim_head,
        mlp_dim,
        vq_codebook_dim=vq_codebook_dim,
        vq_codebook_size=vq_codebook_size,
        vq_entropy_loss_weight=vq_entropy_loss_weight,
        vq_diversity_gamma=vq_diversity_gamma,
        discr_layers=discr_layers
    ).to("cuda")

    loss = model(input_tensor, input_tensor, return_loss=True)

    print("Loss:", loss)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters in the model: {num_params}")