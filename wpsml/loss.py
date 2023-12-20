import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import xarray as xr
import numpy as np


def load_loss(loss_type, reduction='mean'):
    if loss_type == "mse":
        return torch.nn.MSELoss(reduction=reduction)
    if loss_type == "msle":
        return MSLELoss(reduction=reduction)
    if loss_type == "mae":
        return torch.nn.L1Loss(reduction=reduction)
    if loss_type == "huber":
        return torch.nn.SmoothL1Loss(reduction=reduction)
    if loss_type == "logcosh":
        return LogCoshLoss(reduction=reduction)
    if loss_type == "xtanh":
        return XTanhLoss(reduction=reduction)
    if loss_type == "xsigmoid":
        return XSigmoidLoss(reduction=reduction)
    

class LogCoshLoss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(torch.log(torch.cosh(ey_t + 1e-12))) if self.reduction == 'mean' else torch.log(torch.cosh(ey_t + 1e-12))

class XTanhLoss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t + 1e-12
        return torch.mean(ey_t * torch.tanh(ey_t)) if self.reduction == 'mean' else ey_t * torch.tanh(ey_t)

class XSigmoidLoss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t + 1e-12
        return torch.mean(2 * ey_t / (1 + torch.exp(-ey_t)) - ey_t) if self.reduction == 'mean' else 2 * ey_t / (1 + torch.exp(-ey_t)) - ey_t

class MSLELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(MSLELoss, self).__init__()
        self.reduction = reduction

    def forward(self, prediction, target):
        log_prediction = torch.log(prediction.abs() + 1)  # Adding 1 to avoid logarithm of zero
        log_target = torch.log(target.abs() + 1)
        loss = F.mse_loss(log_prediction, log_target, reduction=self.reduction)
        return loss



class SpectralLoss3D(nn.Module):
    def __init__(self, wavenum_init=20):
        super(SpectralLoss3D, self).__init__()
        self.wavenum_init = wavenum_init

    def forward(self, output, target, fft_dim = 4):
        device, dtype = output.device, output.dtype
        output = output.float()
        target = target.float()

        # Take FFT over the 'lon' dimension 
        out_fft = torch.fft.rfft(output, dim=fft_dim)  
        target_fft = torch.fft.rfft(target, dim=fft_dim)
        
        # Take absolute value 
        out_fft_abs = torch.abs(out_fft)
        target_fft_abs = torch.abs(target_fft)

        # Average over spatial dims
        out_fft_mean = torch.mean(out_fft_abs, dim=(fft_dim-1, fft_dim-2)) 
        target_fft_mean = torch.mean(target_fft_abs, dim=(fft_dim-1, fft_dim-2))

        # Calculate loss2
        loss2 = torch.mean(torch.abs(out_fft_mean[:, 0, self.wavenum_init:] - target_fft_mean[:, 0, self.wavenum_init:]))

        # Calculate loss3
        loss3 = torch.mean(torch.abs(out_fft_mean[:, 1, self.wavenum_init:] - target_fft_mean[:, 1, self.wavenum_init:]))
        
        # Compute total loss 
        loss = 0.5 * loss2 + 0.5 * loss3
        
        return loss.to(device=device, dtype=dtype)


class SpectralLoss2D(nn.Module):
    def __init__(self, wavenum_init=20, reduction='none'):
        super(SpectralLoss2D, self).__init__()
        self.wavenum_init = wavenum_init
        self.reduction = reduction

    def forward(self, output, target, fft_dim = 3):
        device, dtype = output.device, output.dtype
        output = output.float()
        target = target.float()

        # Take FFT over the 'lon' dimension 
        out_fft = torch.fft.rfft(output, dim=fft_dim)  
        target_fft = torch.fft.rfft(target, dim=fft_dim)
        
        # Take absolute value 
        out_fft_abs = torch.abs(out_fft)
        target_fft_abs = torch.abs(target_fft)

        # Average over spatial dims
        out_fft_mean = torch.mean(out_fft_abs, dim=(fft_dim-1)) 
        target_fft_mean = torch.mean(target_fft_abs, dim=(fft_dim-1))

        # Calculate loss2
        loss2 = torch.abs(out_fft_mean[:, 0, self.wavenum_init:] - target_fft_mean[:, 0, self.wavenum_init:])
        if self.reduction != 'none':
            loss2 = torch.mean(loss2)

        # Calculate loss3
        loss3 = torch.abs(out_fft_mean[:, 1, self.wavenum_init:] - target_fft_mean[:, 1, self.wavenum_init:])
        if self.reduction != 'none':
            loss3 = torch.mean(loss3)
        
        # Compute total loss 
        loss = 0.5 * loss2 + 0.5 * loss3
        
        return loss.to(device=device, dtype=dtype)

    
def latititude_weights(conf):
    
    cos_lat = xr.open_dataset(conf["loss"]["latitude_weights"])["coslat"].values
    
    H, W = cos_lat.shape

    # Compute the sum of cos(lat) over all rows
    cos_lat_sum = cos_lat.sum(axis=0)

    # Compute the latitude-weighting factor for each row
    L = cos_lat / cos_lat_sum
    L = L / L.sum()
    
    min_val = np.min(L) // 2
    max_val = np.max(L)
    normalized_L = (L - min_val) / (max_val - min_val)
    
    #L = torch.from_numpy(L).float().unsqueeze(0)
    #L = L / L.sum()
    
    return torch.from_numpy(normalized_L).float()


def variable_weights(conf, channels, surface_channels, frames):
    # Load weights for U, V, T, Q
    weights_UVTQ = torch.tensor([
        conf["loss"]["variable_weights"]["U"],
        conf["loss"]["variable_weights"]["V"],
        conf["loss"]["variable_weights"]["T"],
        conf["loss"]["variable_weights"]["Q"]
    ]).view(1, channels * frames, 1, 1)

    # Load weights for SP, t2m, V500, U500, T500, Z500, Q500
    weights_sfc = torch.tensor([
        conf["loss"]["variable_weights"]["SP"],
        conf["loss"]["variable_weights"]["t2m"],
        conf["loss"]["variable_weights"]["V500"],
        conf["loss"]["variable_weights"]["U500"],
        conf["loss"]["variable_weights"]["T500"],
        conf["loss"]["variable_weights"]["Z500"],
        conf["loss"]["variable_weights"]["Q500"]
    ]).view(1, surface_channels, 1, 1)
    
    # Combine all weights along the color channel
    variable_weights = torch.cat([weights_UVTQ, weights_sfc], dim=1)
    
    return variable_weights


class TotalLoss2D(nn.Module):
    def __init__(self, conf, validation = False):
        
        super(TotalLoss2D, self).__init__()
        
        self.conf = conf
        self.training_loss = conf["loss"]["training_loss"]
        
        channels = conf["model"]["channels"]
        surface_channels = conf["model"]["surface_channels"]
        frames = conf["model"]["frames"]
        self.var_weights = variable_weights(conf, channels, surface_channels, frames) if conf["loss"]["use_variable_weights"] else None
        self.lat_weights = latititude_weights(conf) if conf["loss"]["use_latitude_weights"] else None
        
        self.use_spectral_loss = conf["loss"]["use_spectral_loss"]
        if self.use_spectral_loss:
            self.spectral_lambda_reg = conf["loss"]["spectral_lambda_reg"]
            self.spectral_loss_surface = SpectralLoss2D(
                wavenum_init=conf["loss"]["spectral_wavenum_init"], 
                reduction='none'
            )
        
        self.validation = validation
        if self.validation:
            self.loss_fn = nn.L1Loss(reduction='none')
        else:
            self.loss_fn = load_loss(self.training_loss, reduction='none') #nn.MSELoss(reduction='none')

    def forward(self, target, pred):
        loss = self.loss_fn(target, pred)
        
        if self.lat_weights is not None:
            loss *= self.lat_weights.to(target.device)
        if not self.validation and self.var_weights is not None:
            loss *= self.var_weights.to(target.device)

        loss = loss.mean()
            
        if not self.validation and self.use_spectral_loss:
            # Add the spectral loss to the overall loss
            loss += self.spectral_lambda_reg * self.spectral_loss_surface(target, pred).mean()
            
        return loss
