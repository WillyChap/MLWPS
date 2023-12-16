import segmentation_models_pytorch as smp
import torchvision.models as models
import torch.nn.functional as F
import torchvision
import torch.nn as nn
import torch
import numpy as np
import logging
import copy
import os

from wpsml.loss import SpectralLoss, SpectralLossSurface, WeightedMSELoss, WeightedMSELossSurface

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


logger = logging.getLogger(__name__)


supported_models = {
    "unet": smp.Unet,
    "unet++": smp.UnetPlusPlus,
    "manet": smp.MAnet,
    "linknet": smp.Linknet,
    "fpn": smp.FPN,
    "pspnet": smp.PSPNet,
    "pan": smp.PAN,
    "deeplabv3": smp.DeepLabV3,
    "deeplabv3+": smp.DeepLabV3Plus
}

supported_encoders = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x4d', 'resnext101_32x8d', 'resnext101_32x16d', 'resnext101_32x32d', 'resnext101_32x48d', 'dpn68', 'dpn68b', 'dpn92', 'dpn98', 'dpn107', 'dpn131', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'senet154', 'se_resnet50', 'se_resnet101', 'se_resnet152', 'se_resnext50_32x4d', 'se_resnext101_32x4d', 'densenet121', 'densenet169', 'densenet201', 'densenet161', 'inceptionresnetv2', 'inceptionv4', 'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3', 'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7', 'mobilenet_v2', 'xception', 'timm-efficientnet-b0', 'timm-efficientnet-b1', 'timm-efficientnet-b2', 'timm-efficientnet-b3', 'timm-efficientnet-b4', 'timm-efficientnet-b5', 'timm-efficientnet-b6', 'timm-efficientnet-b7', 'timm-efficientnet-b8', 'timm-efficientnet-l2', 'timm-tf_efficientnet_lite0', 'timm-tf_efficientnet_lite1', 'timm-tf_efficientnet_lite2', 'timm-tf_efficientnet_lite3', 'timm-tf_efficientnet_lite4',
                      'timm-resnest14d', 'timm-resnest26d', 'timm-resnest50d', 'timm-resnest101e', 'timm-resnest200e', 'timm-resnest269e', 'timm-resnest50d_4s2x40d', 'timm-resnest50d_1s4x24d', 'timm-res2net50_26w_4s', 'timm-res2net101_26w_4s', 'timm-res2net50_26w_6s', 'timm-res2net50_26w_8s', 'timm-res2net50_48w_2s', 'timm-res2net50_14w_8s', 'timm-res2next50', 'timm-regnetx_002', 'timm-regnetx_004', 'timm-regnetx_006', 'timm-regnetx_008', 'timm-regnetx_016', 'timm-regnetx_032', 'timm-regnetx_040', 'timm-regnetx_064', 'timm-regnetx_080', 'timm-regnetx_120', 'timm-regnetx_160', 'timm-regnetx_320', 'timm-regnety_002', 'timm-regnety_004', 'timm-regnety_006', 'timm-regnety_008', 'timm-regnety_016', 'timm-regnety_032', 'timm-regnety_040', 'timm-regnety_064', 'timm-regnety_080', 'timm-regnety_120', 'timm-regnety_160', 'timm-regnety_320', 'timm-skresnet18', 'timm-skresnet34', 'timm-skresnext50_32x4d', 'timm-mobilenetv3_large_075', 'timm-mobilenetv3_large_100', 'timm-mobilenetv3_large_minimal_100', 'timm-mobilenetv3_small_075', 'timm-mobilenetv3_small_100', 'timm-mobilenetv3_small_minimal_100', 'timm-gernet_s', 'timm-gernet_m', 'timm-gernet_l']


def load_model(model_conf):
    model_conf = copy.deepcopy(model_conf)
    name = model_conf.pop("name")
    if name in supported_models:
        logger.info(f"Loading model {name} with settings {model_conf}")
        return supported_models[name](**model_conf)
    else:
        raise OSError(
            f"Model name {name} not recognized. Please choose from {supported_models.keys()}")


class SegmentationModel(torch.nn.Module):
    
    def __init__(self, conf):
        
        super(SegmentationModel, self).__init__()
                
        self.num_atmos_vars = conf["model"]["channels"]
        self.num_levels = conf["model"]["frames"]
        self.num_single_layer = conf["model"]["surface_channels"]
        use_codebook = False
        
        in_out_channels = int(self.num_atmos_vars*self.num_levels + self.num_single_layer)
        
        dim = 128
        vq_codebook_size = conf['model']['vq_codebook_size']
        vq_decay = conf['model']['vq_decay']
        vq_commitment_weight = conf['model']['vq_commitment_weight']
        
        
        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            decoder_attention_type="scse",
            in_channels=in_out_channels,
            classes=in_out_channels,
        )
        
        # codebook
        self.use_codebook = conf['model']['use_codebook']
        if  self.use_codebook:
            self.vq = VectorQuantize(
                dim = dim,
                codebook_size = vq_codebook_size,     # codebook size
                decay = vq_decay,             # the exponential moving average decay, lower means the dictionary will change faster
                commitment_weight = vq_commitment_weight  # the weight on the commitment loss
            )

    def forward(self, x, x_surface, reshape = True):
        x = self.model(self.concat_and_reshape(x, x_surface))
        return self.split_and_reshape(x) if reshape else x

    def concat_and_reshape(self, x1, x2):
        x1 = x1.view(x1.shape[0], -1, x1.shape[3], x1.shape[4])
        x_concat = torch.cat((x1, x2), dim=1)
        return x_concat

    def split_and_reshape(self, tensor):
        tensor1 = tensor[:, :int(self.num_atmos_vars*self.num_levels), :, :]
        tensor2 = tensor[:, -self.num_single_layer:, :, :]
        tensor1 = tensor1.view(tensor1.shape[0], self.num_atmos_vars, self.num_levels, tensor1.shape[2], tensor1.shape[3])
        return tensor1, tensor2
    
    
class SegmentationRK4(SegmentationModel):
        
    def forward(self, img, img_surface, scaler_y=None):
        k1, k1_surf = self.split_and_reshape(self.model(self.concat_and_reshape(img, img_surface)))
        k2, k2_surf = self.split_and_reshape(self.model(self.concat_and_reshape(img + 0.5 * k1, img_surface + 0.5 * k1_surf)))
        k3, k3_surf = self.split_and_reshape(self.model(self.concat_and_reshape(img + 0.5 * k2, img_surface + 0.5 * k2_surf)))
        k4, k4_surf = self.split_and_reshape(self.model(self.concat_and_reshape(img + k3, img_surface + k3_surf)))

        pred = (k1 + 2 * k2 + 2 * k3 + k4) / 6
        pred_surf = (k1_surf + 2 * k2_surf + 2 * k3_surf + k4_surf) / 6
        
        if scaler_y is not None:
        
            unscaled_tend, unscaled_tend_surf = scaler_y.inverse_transform(
                pred,
                pred_surf
            )

            # Step

            fmap = img + unscaled_tend
            fmap_surface = img_surface + unscaled_tend_surf

            return fmap, fmap_surface
        
        return pred, pred_surf


class UNetEncoderDecoder(torch.nn.Module):
    def __init__(self, conf, **kwargs):
        super().__init__()
        
        self.channels = conf['model']['channels']
        
        self.model = SegmentationModel(conf)
        
        # integration type
        self.rk4_integration = conf['model']['rk4_integration']

        # reconstruction loss
        if conf['model']['l2_recon_loss']:
            self.recon_loss = nn.MSELoss(reduction='none')
            self.recon_loss_surf = nn.MSELoss(reduction='none')
        else:
            self.recon_loss = F.l1_loss
            self.recon_loss_surf = F.l1_loss

        # ssl -- makes more sense to move this to the model class above which contains all layers
        self.visual_ssl = None
        self.visual_ssl_weight = conf['model']['visual_ssl_weight']
        if conf['model']['use_visual_ssl']:
            ssl_type = partial(SimSiam, 
                           channels = conf['model']['channels'], 
                           surface_channels = conf['model']['surface_channels'], 
                           device = next(self.enc_dec.parameters()).device)
            
            self.visual_ssl = ssl_type(
                self.enc_dec.encode,
                image_height = conf['model']['image_height'],
                image_width = conf['model']['image_width'],
                hidden_layer = -1
            )

        # perceptual loss -- possibly the same here
        self.use_vgg = conf['model']['use_vgg']
        if self.use_vgg:
            self.vgg = torchvision.models.vgg16(pretrained = True)
            self.vgg.features[0] = torch.nn.Conv2d(conf['model']['channels'], 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.vgg.classifier = nn.Sequential(*self.vgg.classifier[:-2])
            
            # Freeze the weights of the pre-trained layers
            for param in self.vgg.parameters():
                param.requires_grad = False

        # spectral loss
        self.use_spectral_loss = conf['model']['use_spectral_loss']
        self.spectral_lambda_reg = conf['model']['spectral_lambda_reg'] if self.use_spectral_loss else 1.0
        if self.use_spectral_loss:
            self.spectral_loss = SpectralLoss(wavenum_init=conf['model']['spectral_wavenum_init'])
            self.spectral_loss_surface = SpectralLossSurface(wavenum_init=conf['model']['spectral_wavenum_init'])
    
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
        # see https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/base/model.py

        if self.rk4_integration:
            
            # RK4 steps for encoded result
                        
            k1, k1_surf = self.model(img, img_surface)
            k2, k2_surf = self.model(img + 0.5 * k1, img_surface + 0.5 * k1_surf)
            k3, k3_surf = self.model(img + 0.5 * k2, img_surface + 0.5 * k2_surf)
            k4, k4_surf = self.model(img + k3, img_surface + k3_surf)
            
            # should be z-scores of tendencies
            
            pred = (k1 + 2 * k2 + 2 * k3 + k4) / 6
            pred_surf = (k1_surf + 2 * k2_surf + 2 * k3_surf + k4_surf) / 6
            
            # Undo the z-score output (tendency) of the model
            
            unscaled_tend, unscaled_tend_surf = tendency_scaler.inverse_transform(
                pred,
                pred_surf
            )
            
            # Step
            
            fmap = img + unscaled_tend
            fmap_surface = img_surface + unscaled_tend_surf

        else:            
            fmap, fmap_surface = self.model(img, img_surface)
            pred, pred_surf = fmap, fmap_surface
            
        if not return_loss:
            return fmap, fmap_surface
        
        # Convert true tendencies to z-scores
        if self.rk4_integration:
            true, true_surf = tendency_scaler.transform(y_img-img, y_img_surface-img_surface)
        else:
            true, true_surf = y_img, y_img_surface
            
        # reconstruction loss
        
        if atmosphere_weights is not None and surface_weights is not None:
            recon_loss = (self.recon_loss(true, pred) * atmosphere_weights).mean()
            recon_loss_surface = (self.recon_loss_surf(true_surf, pred_surf) * surface_weights).mean()
        
        else:
            recon_loss = self.recon_loss(true, pred).mean()
            recon_loss_surface = self.recon_loss_surf(true_surf, pred_surf).mean()

        # fourier spectral loss
        
        spec_loss = 0.0
        if self.use_spectral_loss:
            spec_loss_1 = self.spectral_loss(true, pred)
            spec_loss_2 = self.spectral_loss_surface(true_surf, pred_surf)
            spec_loss = 0.5 * (spec_loss_1 + spec_loss_2)
        
        # Add terms
        
        loss = self.spectral_lambda_reg * (recon_loss + recon_loss_surface) + (1 - self.spectral_lambda_reg) * spec_loss
        
        if return_recons:
            return fmap, fmap_surface, loss

        return loss
