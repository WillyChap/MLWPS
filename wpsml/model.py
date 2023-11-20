import torch
from torch import nn
from math import sqrt
from vector_quantize_pytorch import LFQ
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import torch.nn.functional as F
from torch.autograd import grad as torch_grad
import torchvision


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
            assert exists(self.discr), 'discriminator must exist to train it'

            fmap.detach_()
            img.requires_grad_()

            fmap_discr_logits, img_discr_logits = map(self.discr, (fmap, img))

            discr_loss = self.discr_loss(fmap_discr_logits, img_discr_logits)

            if add_gradient_penalty:
                gp = gradient_penalty(img, img_discr_logits)
                loss = discr_loss + gp

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

            # generator loss
            gen_loss = self.gen_loss(self.discr(fmap_frame))

            # calculate adaptive weight
            last_dec_layer = self.enc_dec.last_dec_layer
            norm_grad_wrt_gen_loss = grad_layer_wrt_loss(gen_loss, last_dec_layer).norm(p = 2)
            norm_grad_wrt_perceptual_loss = grad_layer_wrt_loss(perceptual_loss, last_dec_layer).norm(p = 2)

            #print(norm_grad_wrt_perceptual_loss, norm_grad_wrt_gen_loss)
            adaptive_weight = safe_div(norm_grad_wrt_perceptual_loss, norm_grad_wrt_gen_loss)
            adaptive_weight.clamp_(max = 1e4)

            # combine losses
            loss += (perceptual_loss + adaptive_weight * gen_loss)

        if return_recons:
            return loss, fmap

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