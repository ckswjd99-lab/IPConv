import os, sys
import torch, json
import numpy as np

from .main import build_model_main
from .util.slconfig import SLConfig

import os


# model_config_path = "config/DINO/DINO_4scale_swin.py"
# model_checkpoint_path = "checkpoint0029_4scale_swin.pth"

model_config_path = os.path.join(os.path.dirname(__file__), "config", "DINO", "DINO_4scale_swin.py")
model_checkpoint_path = os.path.join(os.path.dirname(__file__), "checkpoint0029_4scale_swin.pth")


def build_dino_4scale_swin(device='cuda'):
    """
    Build the DINO model with 4-scale Swin backbone.
    """
    config = SLConfig.fromfile(model_config_path)
    config.device = device
    model, criterion, postprocessors = build_model_main(config)
    
    # Load the pre-trained weights
    checkpoint = torch.load(model_checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=True)
    
    return model, criterion, postprocessors