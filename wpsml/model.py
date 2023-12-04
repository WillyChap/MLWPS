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
from wpsml.pe import PosEmb3D, SurfacePosEmb2D, GraphPositionBias
from wpsml.visual_ssl import SimSiam, MLP
from wpsml.loss import SpectralLoss, SpectralLossSurface
from functools import partial



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
        self.transformer_decoder = Transformer(
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
        
        self.encoder_surface_embed = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width, c=surface_channels) 
        self.encoder_surface_linear = nn.Linear(input_dim_surface, dim)
        
        # Decoder layers
        self.decoder_linear_1 = nn.Linear(dim, dim * 4)
        self.decoder_linear_2 = nn.Linear(dim * 4, input_dim)
        self.decoder_rearrange = Rearrange('b (f h w) (p1 p2 pf c) -> b c (pf f) (p1 h) (w p2)',  
                                           h=(image_height // patch_height), f=(frames // frame_patch_size),  
                                           p1=patch_height, p2=patch_width, pf=frame_patch_size)
                                           
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
                                                   
    def encode(self, x, x_surf):
        x = self.encoder_embed(x)
        x = self.encoder_linear(x)  
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
        x_surf = self.surface_pos_embedding(x_surf)
        x_surf = self.token_dropout(x_surf)
        if self.use_registers:
            r = repeat(self.register_tokens, 'n d -> b n d', b = x_surf.shape[0])
            x_surf, ps = pack([x_surf, r], 'b * d')
        x_surf = self.transformer_encoder(x_surf)
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
        x = F.tanh(x)
        x = self.decoder_linear_2(x)
        x = self.decoder_rearrange(x)

        x_surf = self.surface_pos_embedding(x_surf)
        if self.use_registers:
            r = repeat(self.register_tokens, 'n d -> b n d', b = x_surf.shape[0])
            x_surf, ps = pack([x_surf, r], 'b * d')
        x_surf = self.transformer_decoder(x_surf)
        if self.use_registers:
            x_surf, _ = unpack(x_surf, ps, 'b * d')
        
        x_surf = self.decoder_surface_linear_1(x_surf)
        x_surf = F.tanh(x_surf)
        x_surf = self.decoder_surface_linear_2(x_surf)
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
        
        self.enc_dec = ViTEncDecSurface(
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
        self.recon_loss_fn = F.mse_loss if l2_recon_loss else F.l1_loss

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
        fmap, fmap_surface = self.enc_dec.encode(fmap, fmap_surface)
        return fmap, fmap_surface

    def decode(self, fmap, fmap_surface):
        #fmap, indices, commit_loss = self.enc_dec.vq(fmap)
        fmap, fmap_surface = self.enc_dec.decode(fmap, fmap_surface)

        return fmap, fmap_surface
    
    def forward(
        self,
        img,
        img_surface,
        y_img = False,
        y_img_surface = False,
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
            k1, k1_surf = self.decode(*self.encode(img, img_surface))
            k2, k2_surf = self.decode(*self.encode(img + 0.5 * k1, img_surface + 0.5 * k1_surf))
            k3, k3_surf = self.decode(*self.encode(img + 0.5 * k2, img_surface + 0.5 * k2_surf))
            k4, k4_surf = self.decode(*self.encode(img + k3, img_surface + k3_surf))
            fmap = img + (k1 + 2 * k2 + 2 * k3 + k4) / 6
            fmap_surface = img_surface + (k1_surf + 2 * k2_surf + 2 * k3_surf + k4_surf) / 6

        else:
            fmap, fmap_surface = self.encode(img, img_surface)
            fmap, fmap_surface = self.decode(fmap, fmap_surface)

        
        if not return_loss:
            return fmap, fmap_surface

        # reconstruction loss
        
        recon_loss = self.recon_loss_fn(y_img, fmap)
        recon_loss_surface = self.recon_loss_fn(y_img_surface, fmap_surface)

        # fourier spectral loss
        spec_loss = 0.0
        if self.use_spectral_loss:
            spec_loss_1 = self.spectral_loss(y_img, fmap)
            spec_loss_2 = self.spectral_loss_surface(y_img_surface, fmap_surface)
            spec_loss = 0.5 * (spec_loss_1 + spec_loss_2)
        
        # Add terms
        
        loss = self.spectral_lambda_reg * (recon_loss + recon_loss_surface) + (1 - self.spectral_lambda_reg) * spec_loss

        if return_recons:
            return fmap, fmap_surface, loss
        
        # perceptual
        # img_vgg_input = img
        # fmap_vgg_input = fmap
        
        # # in the dall-e example, there are no frames (pressure levels). here we loop over them and sum the losses
        # for i in range(img_vgg_input.shape[2]):
        #     # Get the i-th frame from the original and reconstructed videos
        #     img_frame = img_vgg_input[:, :, i, :, :]
        #     fmap_frame = fmap_vgg_input[:, :, i, :, :]

        #     # Compute VGG features for the original and reconstructed frames
        #     img_vgg_feats = self.vgg(img_frame)
        #     recon_vgg_feats = self.vgg(fmap_frame)

        #     # Compute the perceptual loss (MSE loss between VGG features)
        #     perceptual_loss = F.mse_loss(img_vgg_feats, recon_vgg_feats)

        #     #print(i, perceptual_loss)
        #     # generator loss
        #     #gen_loss = self.gen_loss(self.discr(fmap_frame))

        #     # calculate adaptive weight
        #     #last_dec_layer = self.enc_dec.last_dec_layer
        #     #norm_grad_wrt_gen_loss = 1.0 #grad_layer_wrt_loss(gen_loss, last_dec_layer).norm(p = 2)
        #     #norm_grad_wrt_perceptual_loss = 1.0 #grad_layer_wrt_loss(perceptual_loss, last_dec_layer).norm(p = 2)

        #     #print(norm_grad_wrt_perceptual_loss, norm_grad_wrt_gen_loss)
        #     #adaptive_weight = safe_div(norm_grad_wrt_perceptual_loss, norm_grad_wrt_gen_loss)
        #     #adaptive_weight.clamp_(max = 1e4)

        #     # combine losses
        #     loss += perceptual_loss #(perceptual_loss + adaptive_weight * gen_loss)

        return loss


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