import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SpectralLoss(nn.Module):
    def __init__(self, wavenum_init=20):
        super(SpectralLoss, self).__init__()
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


class SpectralLossSurface(nn.Module):
    def __init__(self, wavenum_init=20):
        super(SpectralLossSurface, self).__init__()
        self.wavenum_init = wavenum_init

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
        loss2 = torch.mean(torch.abs(out_fft_mean[:, 0, self.wavenum_init:] - target_fft_mean[:, 0, self.wavenum_init:]))

        # Calculate loss3
        loss3 = torch.mean(torch.abs(out_fft_mean[:, 1, self.wavenum_init:] - target_fft_mean[:, 1, self.wavenum_init:]))
        
        # Compute total loss 
        loss = 0.5 * loss2 + 0.5 * loss3
        
        return loss.to(device=device, dtype=dtype)

    
class MSLELoss(nn.Module):
    def __init__(self):
        super(MSLELoss, self).__init__()

    def forward(self, prediction, target):
        log_prediction = torch.log(prediction.abs() + 1)  # Adding 1 to avoid logarithm of zero
        log_target = torch.log(target.abs() + 1)
        loss = F.mse_loss(log_prediction, log_target)
        return loss
    
    
class WeightedMSELoss(nn.Module):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='none')  # We will handle reduction ourselves

    def forward(self, input, target, weights):
        # Calculate MSE
        loss = self.mse_loss(input, target)
        
        # Apply weights
        loss = loss * weights.view(1, target.size(1), target.size(2), 1, 1)
        
        # Reduce loss; this can also be modified to suit your needs
        loss = loss.mean()
        
        return loss
    
    
class WeightedMSELossSurface(nn.Module):
    def __init__(self):
        super(WeightedMSELossSurface, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='none')  # We will handle reduction ourselves

    def forward(self, input, target, weights):
        # Calculate MSE
        loss = self.mse_loss(input, target)
        
        # Apply weights
        loss = loss * weights.view(1, target.size(1), 1, 1)
        
        # Reduce loss; this can also be modified to suit your needs
        loss = loss.mean()
        
        return loss
    
class WMSESLoss(torch.nn.Module):
    
    def __init__(self, conf, **kwargs):
        
        super().__init__()
        
        # Load weights for U, V, T, Q
        self.weights_UVTQ = torch.tensor([
            conf["weights"]["U"],
            conf["weights"]["V"],
            conf["weights"]["T"],
            conf["weights"]["Q"]
        ]).view(1, conf['model']['channels'], conf['model']['frames'], 1, 1)

        # Load weights for SP, t2m, V500, U500, T500, Z500, Q500
        self.weights_sfc = torch.tensor([
            conf["weights"]["SP"],
            conf["weights"]["t2m"],
            conf["weights"]["V500"],
            conf["weights"]["U500"],
            conf["weights"]["T500"],
            conf["weights"]["Z500"],
            conf["weights"]["Q500"]
        ]).view(1, conf['model']['surface_channels'], 1, 1)

        # reconstruction loss
        if conf['model']['l2_recon_loss']:
            self.recon_loss = nn.MSELoss(reduction='none')
            self.recon_loss_surf = nn.MSELoss(reduction='none')
        else:
            self.recon_loss = nn.L1Loss(reduction='none')
            self.recon_loss_surf = nn.L1Loss(reduction='none')

        # spectral loss
        self.use_spectral_loss = conf['model']['use_spectral_loss']
        self.spectral_lambda_reg = conf['model']['spectral_lambda_reg'] if self.use_spectral_loss else 1.0
        if self.use_spectral_loss:
            self.spectral_loss = SpectralLoss(wavenum_init=conf['model']['spectral_wavenum_init'])
            self.spectral_loss_surface = SpectralLossSurface(wavenum_init=conf['model']['spectral_wavenum_init'])

        # atmosphere weights
        self.use_weights = conf['model']['use_weights'] if 'use_weights' in conf['model'] else False
    
    def forward(self, x, x_surf, y, y_surf):
        # Compute the reconstruction loss
        if self.use_weights:
            recon_loss = (self.recon_loss(x, y) * self.weights_UVTQ).mean()
            recon_loss_surf = (self.recon_loss_surf(x_surf, y_surf) * self.weights_sfc).mean()

            # Compute the spectral loss if applicable
            if self.use_spectral_loss:
                spectral_loss = (self.spectral_loss(x * self.weights_UVTQ, y * self.weights_UVTQ)).mean() * self.spectral_lambda_reg
                spectral_loss_surf = (self.spectral_loss_surface(x_surf * self.weights_sfc, y_surf * self.weights_sfc)).mean() * self.spectral_lambda_reg
        else:
            recon_loss = self.recon_loss(x, y).mean()
            recon_loss_surf = self.recon_loss_surf(x_surf, y_surf).mean()

            # Compute the spectral loss if applicable
            if self.use_spectral_loss:
                spectral_loss = self.spectral_loss(x, y).mean() * self.spectral_lambda_reg
                spectral_loss_surf = self.spectral_loss_surface(x_surf, y_surf).mean() * self.spectral_lambda_reg
            else:
                spectral_loss = 0
                spectral_loss_surf = 0

        # Compute the total loss
        total_loss = recon_loss + recon_loss_surf + spectral_loss + spectral_loss_surf

        return total_loss

