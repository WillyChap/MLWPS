import copy
import random
from functools import wraps

import torch
from torch import nn
import torch.nn.functional as F

from torchvision import transforms as T
from einops import rearrange

# augmentations

class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)

def get_default_aug(image_height, image_width, channels = 4):
    is_rgb = channels == 3
    is_greyscale = channels == 1
    rgb_or_greyscale = is_rgb or is_greyscale

    return torch.nn.Sequential(
        RandomApply(
            T.ColorJitter(0.8, 0.8, 0.8, 0.2),
            p = 0.3
        ) if rgb_or_greyscale else nn.Identity(),
        T.RandomGrayscale(p = 0.2) if is_rgb else nn.Identity(),
        T.RandomHorizontalFlip(),
        RandomApply(
            T.GaussianBlur((3, 3), (1.0, 2.0)),
            p = 0.2
        ),
        T.RandomResizedCrop((image_height, image_width)),
        T.Normalize(
            mean=torch.tensor([0.485, 0.456, 0.406]),
            std=torch.tensor([0.229, 0.224, 0.225])
        ) if is_rgb else nn.Identity(),
    )

# helper functions

def default(val, def_val):
    return def_val if val is None else val

def flatten(t):
    return t.reshape(t.shape[0], -1)

def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn

def get_module_device(module):
    return next(module.parameters()).device

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

def l2norm(t):
    return F.normalize(t, p = 2, dim = -1)

# simclr loss fn

def contrastive_loss(queries, keys, temperature = 0.1):
    b, device = queries.shape[0], queries.device
    logits = queries @ keys.t()
    logits = logits - logits.max(dim=-1, keepdim=True).values
    logits /= temperature
    return F.cross_entropy(logits, torch.arange(b, device=device))

def nt_xent_loss(queries, keys, temperature = 0.1):
    b, device = queries.shape[0], queries.device

    n = b * 2
    projs = torch.cat((queries, keys))
    logits = projs @ projs.t()

    mask = torch.eye(n, device=device).bool()
    logits = logits[~mask].reshape(n, n - 1)
    logits /= temperature

    labels = torch.cat(((torch.arange(b, device = device) + b - 1), torch.arange(b, device=device)), dim=0)
    loss = F.cross_entropy(logits, labels, reduction = 'sum')
    loss /= n
    return loss

# loss fn

def loss_fn(x, y):
    x = l2norm(x)
    y = l2norm(y)
    return 2 - 2 * (x * y).sum(dim=-1)

# MLP class for projector and predictor

def MLP(dim, projection_size, hidden_size = None):
    hidden_size = default(hidden_size, dim)

    return nn.Sequential(
        nn.Linear(dim, hidden_size),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, projection_size)
    )

def SimSiamMLP(dim, projection_size, hidden_size = 4096):
    hidden_size = default(hidden_size, projection_size * 2)

    return nn.Sequential(
        nn.Linear(dim, hidden_size, bias = False),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size, bias = False),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, projection_size, bias = False),
        nn.BatchNorm1d(projection_size, affine = False)
    )

# a wrapper class for the base neural network
# will manage the interception of the hidden layer output
# and pipe it into the projecter and predictor nets

class NetWrapper(nn.Module):
    def __init__(self, net, projection_size, projection_hidden_size = 4096, layer = -2):
        super().__init__()
        self.net = net
        self.layer = layer

        self.projector = None
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size

        self.hidden = {}
        self.hook_registered = False

    def _find_layer(self):
        if type(self.layer) == str:
            modules = dict([*self.net.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*self.net.children()]
            return children[self.layer]
        return None

    def _hook(self, _, input, output):
        device = input[0].device
        self.hidden[device] = flatten(output)

    def _register_hook(self):
        layer = self._find_layer()
        assert layer is not None, f'hidden layer ({self.layer}) not found'
        handle = layer.register_forward_hook(self._hook)
        self.hook_registered = True

    @singleton('projector')
    def _get_projector(self, hidden):
        _, dim = hidden.shape
        projector = SimSiamMLP(dim, self.projection_size, self.projection_hidden_size)
        return projector.to(hidden)

    def get_representation(self, x, x_surf):
        if self.layer == -1:
            return self.net(x, x_surf)

        if not self.hook_registered:
            self._register_hook()

        self.hidden.clear()
        _ = self.net(x, x_surf)
        hidden = self.hidden[x.device]
        self.hidden.clear()

        assert hidden is not None, f'hidden layer {self.layer} never emitted an output'
        return hidden

    def forward(self, x, x_surf, return_projection = True):
        representation, representation_surf = self.get_representation(x, x_surf)

        if not return_projection:
            return representation

        flattened_representation = rearrange(representation, '... d -> (...) d')
        flattened_representation_surf = rearrange(representation_surf, '... d -> (...) d')

        flattened_representation = torch.cat([
            flattened_representation, flattened_representation_surf
        ], dim = 0)
        
        projector = self._get_projector(flattened_representation)
        projection = projector(flattened_representation)
        return projection, representation

# main class

class SimSiam(nn.Module):
    def __init__(
        self,
        net,
        image_height = 640, 
        image_width = 1280,
        channels = 4,
        surface_channels = 7,
        frames = 15, 
        hidden_layer = -2,
        projection_size = 256,
        projection_hidden_size = 4096,
        augment_fn = None,
        augment_fn2 = None,
        device = "cpu"
    ):
        super().__init__()
        self.net = net

        # default SimCLR augmentation

        self.augment1 = get_default_aug(image_height, image_width, channels) 
        self.augment2 = get_default_aug(image_height, image_width, surface_channels)

        self.online_encoder = NetWrapper(net, projection_size, projection_hidden_size, layer=hidden_layer)
        self.online_predictor = MLP(projection_size, projection_size, projection_hidden_size)

        # get device of network and make wrapper same device
        self.to(device)

        # send a mock image tensor to instantiate singleton parameters
        self.forward(
            torch.randn(2, channels, frames, image_height, image_width, device=device), 
            torch.randn(2, surface_channels, image_height, image_width, device=device)
        )

    def forward(self, x, x_surf):
        assert not (self.training and x.shape[0] == 1), 'you must have greater than 1 sample when training, due to the batchnorm in the projection layer'

        # Augment x 
        # Reshape (b, c, f, h, w) --> (b, c * f, h, w)
        x_shape = x.shape
        x = x.view(x_shape[0], -1, x_shape[3], x_shape[4])
        # Reshape (b, c * f, h, w) --> (b, c, f, h, w)
        x1, x2 = self.augment1(x).view(*x_shape), self.augment1(x).view(*x_shape)
        # Augment x_surf
        x1_surf, x2_surf = self.augment2(x_surf), self.augment2(x_surf)

        online_proj_one, _ = self.online_encoder(x1, x1_surf)
        online_proj_two, _ = self.online_encoder(x2, x2_surf)

        online_pred_one = self.online_predictor(online_proj_one)
        online_pred_two = self.online_predictor(online_proj_two)

        with torch.no_grad():
            target_encoder = self.online_encoder
            target_proj_one, _ = target_encoder(x1, x1_surf)
            target_proj_two, _ = target_encoder(x2, x2_surf)
            target_proj_one.detach_()
            target_proj_two.detach_()

        loss_one = loss_fn(online_pred_one, target_proj_two)
        loss_two = loss_fn(online_pred_two, target_proj_one)

        loss = loss_one + loss_two
        return loss.mean()