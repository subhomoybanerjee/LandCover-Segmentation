import torch
import torch.nn as nn
from segmentation_models_pytorch import UnetPlusPlus,DeepLabV3Plus,FPN,PAN,MAnet,PSPNet,Linknet,Unet


class Custom(nn.Module):
    def __init__(self, encoder_name, in_channels, classes,encoder_weights):
        super(Custom, self).__init__()
        self.input_layer = nn.Conv2d(in_channels=in_channels, out_channels=3, kernel_size=1)

        self.unet = UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=classes
        )
    def forward(self, x):
        x = self.input_layer(x)
        return self.unet(x)
    
def initialize_model(encoder_name, in_channels, num_classes, encoder_weights, device):
    model = UnetPlusPlus(
        encoder_name=encoder_name,
        in_channels=in_channels,
        encoder_weights=encoder_weights,
        classes=num_classes,
        encoder_depth=5,
        decoder_channels=[256, 128, 64, 32, 16],
        decoder_use_batchnorm=True,
        decoder_attention_type='scse'
    ).to(device)
    return model
