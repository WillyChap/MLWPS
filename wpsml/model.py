import torch
from torch import nn
from math import sqrt
from vector_quantize_pytorch import LFQ
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import torch.nn.functional as F
from torch.autograd import grad as torch_grad
import torchvision


class CustomLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super(CustomLayerNorm, self).__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        normalized_x = (x - mean) / (std + self.eps)

        # Apply weight and bias without in-place modification
        weighted_normalized_x = self.weight * normalized_x
        output = weighted_normalized_x + self.bias

        return output


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

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)

        # with torch.backends.cuda.sdp_kernel(
        #     enable_flash=True, 
        #     enable_math=False, 
        #     enable_mem_efficient=False
        # ):
        #     out = F.scaled_dot_product_attention(
        #         q, k, v,
        #         attn_mask = None,
        #         dropout_p = 0.0
        #     )

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

    
class ContinuousPositionBias(nn.Module):
    """ from https://arxiv.org/abs/2111.09883 """

    def __init__(self, *, dim, heads, layers = 2):
        super().__init__()
        self.net = MList([])
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
    
    
class ViTAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        dropout = 0.
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.dropout = nn.Dropout(dropout)
        self.pre_norm = LayerNormChan(dim)

        self.cpb = ContinuousPositionBias(dim = dim // 4, heads = heads)
        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(inner_dim, dim, 1, bias = False)

    def forward(self, x):
        h = self.heads
        height, width, residual = *x.shape[-2:], x.clone()

        x = self.pre_norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim = 1)

        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = h), (q, k, v))

        sim = einsum('b h c i, b h c j -> b h i j', q, k) * self.scale

        sim = self.cpb(sim)

        attn = stable_softmax(sim, dim = -1)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h c j -> b h c i', attn, v)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', x = height, y = width)
        out = self.to_out(out)

        return out + residual
    
    
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


class ViTEncDec(nn.Module):
    def __init__(
        self,
        image_height, 
        patch_height,
        image_width,
        patch_width,
        frames, 
        frame_patch_size,
        dim,
        channels = 3,
        depth = 4,
        heads = 8,
        dim_head = 32,
        mlp_dim = 32
    ):
        super().__init__()
        
        num_patches = (image_height // patch_height) * (image_width // patch_width) * (frames // frame_patch_size)
        input_dim = channels * patch_height * patch_width * frame_patch_size
        
        self.encoder = nn.Sequential(
            Rearrange('b c (f pf) (h p1) (w p2) -> b (f h w) (p1 p2 pf c)', p1 = patch_height, p2 = patch_width, pf = frame_patch_size),
            nn.Linear(input_dim, dim),
            Transformer(
                dim = dim,
                depth = depth,
                dim_head = dim_head,
                heads = heads,
                mlp_dim = mlp_dim
            ),
             #Rearrange('b (f h w) c -> b c f h w', h = patch_height, w = patch_width)
        )
        
        self.decoder = nn.Sequential(
            #Rearrange('b c f h w -> b (f h w) c'),
            Transformer(
                dim = dim,
                depth = depth,
                dim_head = dim_head,
                heads = heads,
                mlp_dim = mlp_dim
            ),
            nn.Sequential(
                nn.Linear(dim, dim * 4, bias = False),
                nn.Tanh(),
                nn.Linear(dim * 4, input_dim, bias = False),
            ),
            Rearrange('b (f h w) (p1 p2 pf c) -> b c (pf f) (p1 h) (w p2)', 
                      h = (image_height // patch_height), f = (frames // frame_patch_size), p1 = patch_height, p2 = patch_width, pf = frame_patch_size)
        )
        
    def get_encoded_fmap_size(self, image_size):
        return image_size // self.patch_size

    @property
    def last_dec_layer(self):
        return self.decoder[-2][-1].weight

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)


class Discriminator(nn.Module):
    def __init__(
        self,
        dims,
        channels = 3,
        groups = 16,
        init_kernel_size = 5
    ):
        super().__init__()
        dim_pairs = zip(dims[:-1], dims[1:])

        self.layers = nn.ModuleList([nn.Sequential(nn.Conv2d(channels, dims[0], init_kernel_size, padding = init_kernel_size // 2), nn.LeakyReLU(0.1))])

        for dim_in, dim_out in dim_pairs:
            self.layers.append(nn.Sequential(
                nn.Conv2d(dim_in, dim_out, 4, stride = 2, padding = 1),
                nn.GroupNorm(groups, dim_out),
                nn.LeakyReLU(0.1)
            ))

        dim = dims[-1]
        self.to_logits = nn.Sequential( # return 5 x 5, for PatchGAN-esque training
            nn.Conv2d(dim, dim, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(dim, 1, 4)
        )

    def forward(self, x):
        for net in self.layers:
            x = net(x)

        return self.to_logits(x)


def hinge_discr_loss(fake, real):
    return (F.relu(1 + fake) + F.relu(1 - real)).mean()

def hinge_gen_loss(fake):
    return -fake.mean()

def grad_layer_wrt_loss(loss, layer):
    return 1.0
    return torch_grad(
        outputs = loss,
        inputs = layer,
        grad_outputs = torch.ones_like(loss),
        retain_graph = True
    )[0].detach()

def safe_div(numer, denom, eps = 1e-8):
    return numer / (denom + eps)

def bce_discr_loss(fake, real):
    return (-log(1 - torch.sigmoid(fake)) - log(torch.sigmoid(real))).mean()

def bce_gen_loss(fake):
    return -log(torch.sigmoid(fake)).mean()

def gradient_penalty(images, output, weight = 10):
    batch_size = images.shape[0]
    gradients = torch_grad(outputs = output, inputs = images,
                           grad_outputs = torch.ones(output.size(), device = images.device),
                           create_graph = True, retain_graph = True, only_inputs = True)[0]

    gradients = rearrange(gradients, 'b ... -> b (...)')
    return weight * ((gradients.norm(2, dim = 1) - 1) ** 2).mean()


class VQGanVAE(nn.Module):
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
        depth,
        heads,
        dim_head,
        mlp_dim,
        l2_recon_loss = False,
        use_hinge_loss = True,
        vgg = None,
        use_vgg_and_gan = True,
        vq_codebook_dim = 256,
        vq_codebook_size = 65536,
        vq_entropy_loss_weight = 0.8,
        vq_diversity_gamma = 1.,
        discr_layers = 4,
        **kwargs
    ):
        super().__init__()
        
        self.channels = channels
        self.codebook_size = vq_codebook_size

        self.enc_dec = ViTEncDec(
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
            mlp_dim
        )

        self.vq = self._vq_vae = LFQ(
            codebook_size = vq_codebook_size,      # codebook size, must be a power of 2
            dim = dim,                  # this is the input feature dimension, defaults to log2(codebook_size) if not defined
            entropy_loss_weight = vq_entropy_loss_weight,  # how much weight to place on entropy loss
            diversity_gamma = vq_diversity_gamma        # within entropy loss, how much weight to give to diversity of codes, taken from https://arxiv.org/abs/1911.05894
        )

        # reconstruction loss
        self.recon_loss_fn = F.mse_loss if l2_recon_loss else F.l1_loss
        self.vgg = torchvision.models.vgg16(pretrained = True)
        self.vgg.features[0] = torch.nn.Conv2d(channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.vgg.classifier = nn.Sequential(*self.vgg.classifier[:-2])
        
        # Freeze the weights of the pre-trained layers
        for param in self.vgg.parameters():
            param.requires_grad = False

        # gan related losses
        layer_mults = list(map(lambda t: 2 ** t, range(discr_layers)))
        layer_dims = [dim * mult for mult in layer_mults]
        dims = (dim, *layer_dims)

        self.discr = Discriminator(dims = dims, channels = channels)
        self.discr_loss = hinge_discr_loss if use_hinge_loss else bce_discr_loss
        self.gen_loss = hinge_gen_loss if use_hinge_loss else bce_gen_loss
        
        self.use_vgg_and_gan = use_vgg_and_gan

    @property
    def encoded_dim(self):
        return self.enc_dec.encoded_dim

    def get_encoded_fmap_size(self, image_size):
        return self.enc_dec.get_encoded_fmap_size(image_size)

    def copy_for_eval(self):
        device = next(self.parameters()).device
        vae_copy = copy.deepcopy(self.cpu())

        if vae_copy.use_vgg_and_gan:
            del vae_copy.discr
            del vae_copy.vgg

        vae_copy.eval()
        return vae_copy.to(device)

    def state_dict(self, *args, **kwargs):
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return super().load_state_dict(*args, **kwargs)

    @property
    def codebook(self):
        return self.vq.codebook

    def encode(self, fmap):
        fmap = self.enc_dec.encode(fmap)
        return fmap

    def decode(self, fmap, return_indices_and_loss = False):
        fmap, indices, commit_loss = self.vq(fmap)

        fmap = self.enc_dec.decode(fmap)

        if not return_indices_and_loss:
            return fmap

        return fmap, indices, commit_loss

    def forward(
        self,
        img,
        target,
        return_loss = False,
        return_discr_loss = False,
        return_recons = False,
        add_gradient_penalty = True
    ):
        batch, channels, frames, height, width, device = *img.shape, img.device
        
        #assert height == self.image_size and width == self.image_size, 'height and width of input image must be equal to {self.image_size}'
        #assert channels == self.channels, 'number of channels on image or sketch is not equal to the channels set on this VQGanVAE'

        fmap = self.encode(img)
        fmap, indices, commit_loss = self.decode(fmap, return_indices_and_loss = True)
        
        # For next time step
        img = target

        if not return_loss and not return_discr_loss:
            return fmap

        assert return_loss ^ return_discr_loss, 'you should either return autoencoder loss or discriminator loss, but not both'

        # whether to return discriminator loss

        if return_discr_loss:
            assert self.discr is not None, 'discriminator must exist to train it'

            #fmap.detach_()
            #img.requires_grad_()
            
            # Initialize an empty list to store discriminator losses for each level
            discr_losses = []

            for i in range(img.size(2)):
                img_frame = img[:, :, i, :, :]
                fmap_frame = fmap[:, :, i, :, :]

                img_frame.requires_grad_()
                fmap_frame.requires_grad_()

                fmap_discr_logits, img_discr_logits = map(self.discr, (fmap_frame, img_frame))

                discr_loss = self.discr_loss(fmap_discr_logits, img_discr_logits)

                if add_gradient_penalty:
                    gp = gradient_penalty(img_frame, img_discr_logits)
                    loss = discr_loss + gp
                else:
                    loss = discr_loss

                # Append the discriminator loss for the current level to the list
                discr_losses.append(loss)

            # Sum the accumulated discriminator losses along the third dimension
            loss = torch.mean(torch.stack(discr_losses))
    
            #fmap_discr_logits, img_discr_logits = map(self.discr, (fmap, img))

            #discr_loss = self.discr_loss(fmap_discr_logits, img_discr_logits)

            #if add_gradient_penalty:
            #    gp = gradient_penalty(img, img_discr_logits)
            #    loss = discr_loss + gp

            if return_recons:
                return loss, fmap

            return loss

        # reconstruction loss
        recon_loss = self.recon_loss_fn(fmap, img)

        # early return if training on grayscale
        if not self.use_vgg_and_gan:
            if return_recons:
                return recon_loss, fmap

            return recon_loss

#         # perceptual loss
#         img_vgg_input = img
#         fmap_vgg_input = fmap

#         # in the dall-e example, there are no frames (pressure levels). here we reshape instead of loop over levels
#         # first get the vae loss
#         loss = recon_loss + commit_loss

#         # Reshape inputs
#         num_channels = img_vgg_input.shape[1]
#         num_frames = img_vgg_input.shape[2]
#         height = img_vgg_input.shape[3]
#         width = img_vgg_input.shape[4]

#         img_vgg_input_reshaped = img_vgg_input.permute(0, 2, 1, 3, 4).contiguous().view(-1, num_channels, height, width)
#         fmap_vgg_input_reshaped = fmap_vgg_input.permute(0, 2, 1, 3, 4).contiguous().view(-1, num_channels, height, width)
        
#         # Compute VGG features for the original and reconstructed frames
#         img_vgg_feats = self.vgg(img_vgg_input_reshaped)
#         recon_vgg_feats = self.vgg(fmap_vgg_input_reshaped)
        
#         # Compute the perceptual loss (MSE loss between VGG features)
#         perceptual_loss = F.mse_loss(img_vgg_feats, recon_vgg_feats)
        
#         # Combine losses
#         loss += perceptual_loss

        # perceptual loss
        img_vgg_input = img
        fmap_vgg_input = fmap
        
        # in the dall-e example, there are no frames (pressure levels). here we loop over them and sum the losses
        loss = recon_loss + commit_loss
        for i in range(img_vgg_input.shape[2]):
            # Get the i-th frame from the original and reconstructed videos
            img_frame = img_vgg_input[:, :, i, :, :]
            fmap_frame = fmap_vgg_input[:, :, i, :, :]

            # Compute VGG features for the original and reconstructed frames
            img_vgg_feats = self.vgg(img_frame)
            recon_vgg_feats = self.vgg(fmap_frame)

            # Compute the perceptual loss (MSE loss between VGG features)
            perceptual_loss = F.mse_loss(img_vgg_feats, recon_vgg_feats)

            #print(i, perceptual_loss)
            # generator loss
            #gen_loss = self.gen_loss(self.discr(fmap_frame))

            # calculate adaptive weight
            #last_dec_layer = self.enc_dec.last_dec_layer
            #norm_grad_wrt_gen_loss = 1.0 #grad_layer_wrt_loss(gen_loss, last_dec_layer).norm(p = 2)
            #norm_grad_wrt_perceptual_loss = 1.0 #grad_layer_wrt_loss(perceptual_loss, last_dec_layer).norm(p = 2)

            #print(norm_grad_wrt_perceptual_loss, norm_grad_wrt_gen_loss)
            #adaptive_weight = safe_div(norm_grad_wrt_perceptual_loss, norm_grad_wrt_gen_loss)
            #adaptive_weight.clamp_(max = 1e4)

            # combine losses
            loss += perceptual_loss #(perceptual_loss + adaptive_weight * gen_loss)

        if return_recons:
            return loss, fmap

        return loss


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
        channels = 3,
        depth = 4,
        heads = 8,
        dim_head = 32,
        mlp_dim = 32
    ):
        super().__init__()
        
        num_patches = (image_height // patch_height) * (image_width // patch_width) * (frames // frame_patch_size)
        input_dim = channels * patch_height * patch_width * frame_patch_size
        input_dim_surface = patch_height * patch_width
        
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
        
        self.encoder = nn.Sequential(
            Rearrange('b c (f pf) (h p1) (w p2) -> b (f h w) (p1 p2 pf c)', p1 = patch_height, p2 = patch_width, pf = frame_patch_size),
            nn.Linear(input_dim, dim),
            self.transformer_encoder,
             #Rearrange('b (f h w) c -> b c f h w', h = patch_height, w = patch_width)
        )
        
        self.encoder_surface = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width, c = 1),
            nn.Linear(input_dim_surface, dim),
            self.transformer_encoder,
             #Rearrange('b (f h w) c -> b c f h w', h = patch_height, w = patch_width)
        )
        
        self.decoder = nn.Sequential(
            #Rearrange('b c f h w -> b (f h w) c'),
            self.transformer_decoder,
            nn.Sequential(
                nn.Linear(dim, dim * 4, bias = False),
                nn.Tanh(),
                nn.Linear(dim * 4, input_dim, bias = False),
            ),
            Rearrange('b (f h w) (p1 p2 pf c) -> b c (pf f) (p1 h) (w p2)', 
                      h = (image_height // patch_height), f = (frames // frame_patch_size), p1 = patch_height, p2 = patch_width, pf = frame_patch_size)
        )
        
        self.decoder_surface = nn.Sequential(
            #Rearrange('b c f h w -> b (f h w) c'),
            self.transformer_decoder,
            nn.Sequential(
                nn.Linear(dim, dim * 4, bias = False),
                nn.Tanh(),
                nn.Linear(dim * 4, input_dim_surface, bias = False),
            ),
            Rearrange('b (h w) (p1 p2 c) -> b c (p1 h) (w p2)', w = (image_width // patch_width),
                      c = 1, p1 = patch_height, p2 = patch_width)
        )

    def encode(self, x, x_surf):
        encoded = self.encoder(x)
        encoded_surf = self.encoder_surface(x_surf)
        return encoded, encoded_surf

    def decode(self, x, x_surf):
        decoded = self.decoder(x)
        decoded_surf = self.decoder_surface(x_surf)
        return decoded, decoded_surf
    

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
        depth,
        heads,
        dim_head,
        mlp_dim,
        l2_recon_loss = False,
        use_hinge_loss = True,
        vgg = None,
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
            channels,
            depth,
            heads,
            dim_head,
            mlp_dim
        )

        # reconstruction loss
        self.recon_loss_fn = F.mse_loss if l2_recon_loss else F.l1_loss

    def state_dict(self, *args, **kwargs):
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return super().load_state_dict(*args, **kwargs)

    def encode(self, fmap, fmap_surface):
        fmap, fmap_surface = self.enc_dec.encode(fmap, fmap_surface)
        return fmap, fmap_surface

    def decode(self, fmap, fmap_surface):

        fmap, fmap_surface = self.enc_dec.decode(fmap, fmap_surface)

        return fmap, fmap_surface
    
    def forward(
        self,
        img,
        img_surface,
        y_img,
        y_img_surface,
        return_loss = False,
        return_recons = False
    ):
        batch, channels, frames, height, width, device = *img.shape, img.device
        
        fmap, fmap_surface = self.encode(img, img_surface)
        fmap, fmap_surface = self.decode(fmap, fmap_surface)

        if not return_loss:
            return fmap, fmap_surface

        # reconstruction loss
        recon_loss = self.recon_loss_fn(y_img, fmap)
        recon_loss_surface = self.recon_loss_fn(y_img_surface, fmap_surface)
        
        loss = recon_loss + recon_loss_surface

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