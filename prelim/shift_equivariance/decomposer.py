import torchvision
import torch
from torch import nn, Tensor

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union


class FasterRCNN_Decom(nn.Module):
    def __init__(self, detector):
        super(FasterRCNN_Decom, self).__init__()
        self.detector = detector

        self.transform = detector.transform
        self.backbone = detector.backbone
        self.rpn = detector.rpn
        self.roi_heads = detector.roi_heads

    def forward(self, images, get_features=False):
        # print("[Images]")
        # for img in images:
        #     print(img.shape)

        targets = None

        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            )
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)

        print("[Images]")
        for img in images.tensors:
            print(img.shape)

        raw_features = self.backbone.body(images.tensors)
        features = self.backbone.fpn(raw_features)

        # print()
        # print("[Features]")
        # for key, value in features.items():
        #     print(f"{key}: {value.shape}")
            

        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)  # type: ignore[operator]

        if get_features:
            return detections, raw_features, features
        else:
            return detections