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
