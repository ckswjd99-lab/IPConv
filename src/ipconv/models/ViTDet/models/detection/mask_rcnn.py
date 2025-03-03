from collections import OrderedDict
from typing import Any, Callable, Optional

from torch import nn
from torchvision.ops import MultiScaleRoIAlign

from torchvision.ops import misc as misc_nn_ops
from torchvision.transforms._presets import ObjectDetection
from torchvision.models._api import register_model, Weights, WeightsEnum
from torchvision.models._meta import _COCO_CATEGORIES
from torchvision.models._utils import _ovewrite_value_param, handle_legacy_interface
from torchvision.models.resnet import resnet50, ResNet50_Weights
# from torchvision.models.vision_transformer import ViT_B_16_Weights, vit_b_16
from torchvision.models.detection._utils import overwrite_eps
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor, _validate_trainable_layers
from torchvision.models.detection.faster_rcnn import _default_anchorgen, FasterRCNN, FastRCNNConvFCHead, RPNHead

from torchvision.models.detection import MaskRCNN
from ..detection.backbone_utils import _vit_sfpn_extractor
from ..vision_transformer import vit_b_16, ViT_B_16_Weights


__all__ = [
    "MaskRCNN",
    "MaskRCNN_ResNet50_FPN_Weights",
    "MaskRCNN_ResNet50_FPN_V2_Weights",
    "maskrcnn_resnet50_fpn",
    "maskrcnn_resnet50_fpn_v2",
    "maskrcnn_vit_b_16_sfpn"
]

@register_model()
def maskrcnn_vit_b_16_sfpn(
    *,
    weights: Optional[WeightsEnum] = None,  # TODO: Change this to MaskRCNN_ViT_B_16_SFPN_V1_Weights
    progress: bool = True,
    num_classes: Optional[int] = None,
    # weights_backbone: Optional[ViT_B_16_Weights] = ViT_B_16_Weights.IMAGENET1K_V1,
    weights_backbone: Optional[ViT_B_16_Weights] = None,
    **kwargs: Any,
) -> MaskRCNN:
    """Mask R-CNN model with a ViT-B/16 backbone from the `ViT <https://arxiv.org/abs/2010.11929>`_ paper.

    .. betastatus:: detection module

    The input to the model is expected to be a list of tensors, each of shape ``[C, H, W]``, one for each
    image, and should be in ``0-1`` range. Different images can have different sizes.

    The behavior of the model changes depending if it is in training or evaluation mode.

    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:

        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (``Int64Tensor[N]``): the class label for each ground-truth box
        - masks (``UInt8Tensor[N, H, W]``): the segmentation binary masks for each instance

    The model returns a ``Dict[Tensor]`` during training, containing the classification and regression
    losses for both the RPN and the R-CNN, and the mask loss.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a ``List[Dict[Tensor]]``, one for each input image. The fields of the ``Dict`` are as
    follows, where ``N`` is the number of detected instances:

        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (``Int64Tensor[N]``): the predicted labels for each instance
        - scores (``Tensor[N]``): the scores or each instance
        - masks (``UInt8Tensor[N, 1, H, W]``): the predicted masks for each instance, in ``0-1`` range. In order to
          obtain the final segmentation masks, the soft masks can be thresholded, generally
          with a value of 0.5 (``mask >= 0.5``)

    For more details on the output and on how to plot the masks, you may refer to :ref:`instance_seg_output`.

    Mask R-CNN is exportable to ONNX for a fixed batch size with inputs images of fixed size.

    Example::

        >>> model = torchvision.models.detection.maskrcnn_vit_b_16_sfpn(weights=MaskRCNN_ViT_B_16_SFPN_Weights.DEFAULT)
        >>> model.eval()
        >>> x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        >>> predictions = model(x)
        >>>
        >>> # optionally, if you want to export the model to ONNX:
        >>> torch.onnx.export(model, x, "mask_rcnn.onnx", opset_version = 11)

    Args:
        weights (:class:`~torchvision.models.detection.MaskRCNN_ViT_B_16_SFPN_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.detection.MaskRCNN_ViT_B_16_SFPN_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        num_classes (int, optional): number of output classes of the model (including the background)
        weights_backbone (:class:`~torchvision.models.ViT_B_16_Weights`, optional): The
            pretrained weights for the backbone.
        **kwargs: parameters passed to the ``torchvision.models.detection.mask_rcnn.MaskRCNN``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/detection/mask_rcnn.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.detection.MaskRCNN_ViT_B_16_SFPN_V1_Weights
        :members:
    """
    # TODO: Add ViTDet weights.
    # weights = MaskRCNN_ViT_B_16_SFPN_V1_Weights.verify(weights)
    # weights_backbone = ViT_B_16_Weights.verify(weights_backbone)

    if weights is not None:
        weights_backbone = None
        num_classes = _ovewrite_value_param(num_classes, len(weights.meta["categories"]))
    elif num_classes is None:
        num_classes = 91

    backbone = vit_b_16(weights=weights_backbone, progress=progress, include_head=False, image_size=1024)
    backbone = _vit_sfpn_extractor(backbone)
    model = MaskRCNN(backbone, num_classes=num_classes, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model
