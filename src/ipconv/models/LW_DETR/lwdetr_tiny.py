from .lwdetr import build_lwdetr_tiny
from .util.misc import NestedTensor, nested_tensor_from_tensor_list
from .backbone.vit import get_abs_pos

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, List
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F


class LWDETR_Tiny_Contexted(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        self.model, self.criterion, self.postprocessors = build_lwdetr_tiny()
        self.model.eval()
        self.model = self.model.to(self.device)

        # COCO label list (same as in DINO for consistency)
        self.COCO_LABELS_LIST = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
            'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        np.random.seed(42)
        self.COCO_COLORS_ARRAY = np.random.randint(256, size=(91, 3)) / 255
        self.COCO_LABELS_MAP = {k: v for v, k in enumerate(self.COCO_LABELS_LIST)}

        self.output = None

    def preprocess_image(self, image_ndarray: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        image_tensor = torch.tensor(image_ndarray, dtype=torch.uint8).permute(2, 0, 1).to(self.device)
        image_tensor = (image_tensor / 255.0).float()  # Normalize to [0, 1] range
        image_tensor = image_tensor.permute(1, 2, 0)
        # print(image_tensor.shape)
        image_tensor = (image_tensor - torch.tensor([0.485, 0.456, 0.406], device=self.device)) / \
                       torch.tensor([0.229, 0.224, 0.225], device=self.device)
        image = image_tensor.permute(2, 0, 1)  # Convert to (C, H, W) format


        # image = Image.fromarray(image_ndarray).convert("RGB")
        # normalize = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # ])
        # image = normalize(image)
        # image.permute
        orig_image_size = torch.tensor([image.shape[1], image.shape[2]], dtype=torch.float)
        return image, orig_image_size

    @torch.no_grad()
    def forward_contexted(
        self,
        image_ndarray: np.ndarray,  # (1024, 1024, 3) uint8
        anchor_features: Dict[str, torch.Tensor] = {},
        dirtiness_map: torch.Tensor = torch.ones(1, 256, 256, 1, device="cuda"),
        only_backbone: bool = False
    ) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Dict[str, torch.Tensor]]:
        """
        image_ndarray: np.ndarray, shape (H, W, C), uint8
        anchor_features: dict, optional, for context features (not used in default LWDETR)
        only_backbone: bool, if True, only run backbone (not used in default LWDETR)
        Returns: (boxes, labels, scores), {}
        """
        image, orig_image_size = self.preprocess_image(image_ndarray)
        image = image.to(self.device)
        orig_image_size = orig_image_size.to(self.device)

        samples = nested_tensor_from_tensor_list([image])
        orig_image_sizes = torch.stack([orig_image_size])

        model = self.model
        # outputs = self.model(samples)

        ## BACKBONE INFERENCE ##
        joiner = model.backbone
        backbone = joiner[0]
        position_embedding = joiner[1]
        
        # ViT backbone
        vit_backbone = backbone.encoder

        # feats = vit_backbone(samples.tensors)

        x = samples.tensors
        x = vit_backbone.patch_embed(x)

        if vit_backbone.pos_embed is not None:
            if vit_backbone._export:
                x = x + vit_backbone.pos_embed_export
            else:
                x = x + get_abs_pos(
                    vit_backbone.pos_embed, vit_backbone.pretrain_use_cls_token, (x.shape[1], x.shape[2])
                )

        B, H, W, C = x.shape

        assert (H % 4 == 0) and (W % 4 == 0)
        h, w = H // 4, W // 4
        num_tokens = h * w

        x = x.reshape(B, 4, h, 4, w, C).permute(
            0, 1, 3, 2, 4, 5).reshape(B * 16, h * w, C)
        feats = []
        
        total_tokens = x.shape[0] * x.shape[1]
        num_select = int(total_tokens * 0.3)
        dmap_index = torch.randperm(total_tokens, device=x.device)[:num_select]
        
        for idx, blk in enumerate(vit_backbone.blocks):
            ## BLOCK ATTN INFERENCE ##
            mask=None
            B_blk, HW_blk, C_blk = x.shape
            shortcut = x
            x = blk.norm1(x)

            if not blk.window:
                x = x.reshape(B_blk // 16, 16 * HW_blk, C_blk)
                if mask is not None:
                    mask = mask.reshape(B_blk // 16, 16 * HW_blk)

            # if blk.use_cae:
            x = blk.gamma_1 * blk.attn.forward_dmap(x, dmap_index, mask)
            # else:
            #     x = blk.attn(x, mask)

            if not blk.window:
                x = x.reshape(B_blk, HW_blk, C_blk)
                if mask is not None:
                    mask = mask.reshape(B_blk, HW_blk)
            
            x = shortcut + blk.drop_path(x)

            ## BLOCK MLP INFERENCE ##

            # if blk.use_cae:
            x_cached = x
            x = blk.norm2(x)
            x = blk.gamma_2 * blk.mlp(x)
            x = blk.drop_path(x)
            x = x + x_cached
            # else:
            #     x_cached = x
            #     x = blk.norm2(x)
            #     x = blk.mlp(x)
            #     x = blk.drop_path(x)
            #     x = x + x_cached
            
            # feature extraction
            if vit_backbone._out_features[idx]:
                feats.append(x.reshape(B, 4, 4, h, w, C).permute(
                    0, 5, 1, 3, 2, 4).reshape(B, C, H, W))
                
        
        # Rest of the backbone
        feats = backbone.projector(feats)
        
        features = []
        for feat in feats:
            m = samples.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=feat.shape[-2:]).to(torch.bool)[0]
            features.append(NestedTensor(feat, mask))
        
        poss = []
        for x_ in features:
            poss.append(position_embedding(x_, align_dim_orders=False).to(x_.tensors.dtype))


        ## DETECTION HEAD INFERENCE ##
        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(src)
            masks.append(mask)
            assert mask is not None

        # only use one group in inference
        refpoint_embed_weight = model.refpoint_embed.weight[:model.num_queries]
        query_feat_weight = model.query_feat.weight[:model.num_queries]

        hs, ref_unsigmoid, hs_enc, ref_enc = model.transformer(
            srcs, masks, poss, refpoint_embed_weight, query_feat_weight)

        if model.bbox_reparam:
            outputs_coord_delta = model.bbox_embed(hs)
            outputs_coord_cxcy = outputs_coord_delta[..., :2] * ref_unsigmoid[..., 2:] + ref_unsigmoid[..., :2]
            outputs_coord_wh = outputs_coord_delta[..., 2:].exp() * ref_unsigmoid[..., 2:]
            outputs_coord = torch.concat(
                [outputs_coord_cxcy, outputs_coord_wh], dim=-1
            )
        else:
            outputs_coord = (model.bbox_embed(hs) + ref_unsigmoid).sigmoid()

        outputs_class = model.class_embed(hs)

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        # if model.aux_loss:
        #     out['aux_outputs'] = model._set_aux_loss(outputs_class, outputs_coord)

        if model.two_stage:
            hs_enc_list = hs_enc.split(model.num_queries, dim=1)
            cls_enc = []
            group_detr = model.group_detr if model.training else 1
            for g_idx in range(group_detr):
                cls_enc_gidx = model.transformer.enc_out_class_embed[g_idx](hs_enc_list[g_idx])
                cls_enc.append(cls_enc_gidx)
            cls_enc = torch.cat(cls_enc, dim=1)
            out['enc_outputs'] = {'pred_logits': cls_enc, 'pred_boxes': ref_enc}

        outputs = out

        ## POST-PROCESSING ##

        predictions = self.postprocessors['bbox'](outputs, orig_image_sizes)

        boxes = predictions[0]['boxes'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()

        return (boxes, labels, scores), {}