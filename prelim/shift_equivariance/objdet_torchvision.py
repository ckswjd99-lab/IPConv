import cv2
import torch
import torchvision
import numpy as np
from PIL import Image
import torchvision.transforms as T

from decomposer import FasterRCNN_Decom
from utils import visualize_detection_result


def demo(img_path, threshold):
    """
    demo faster rcnn
    :param img_path: image path (default - soccer.png)
    :param threshold: the threshold of object detection score (default - 0.9)
    :return: None
    """

    # 1. load image
    img_pil = Image.open(img_path).convert('RGB')
    transform = T.Compose([T.ToTensor()])
    img = transform(img_pil)
    batch_img = [img.cuda()]

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    ).cuda()

    model = FasterRCNN_Decom(model)

    model.eval()
    pred, feat = model(batch_img, get_features=True)

    shift = 16
    img_shifted = img[:, shift:, shift:]
    batch_img_shifted = [img_shifted.cuda()]


    # 2. remove first batch
    pred_dict = pred[0]
    '''
    pred_dict 
    {'boxes' : tensor,
     'labels' : tensor,
     'scores' : tensor}
    '''

    # 3. get pred boxes and labels, scores
    pred_boxes = pred_dict['boxes']    # [N, 1]
    pred_labels = pred_dict['labels']  # [N]
    pred_scores = pred_dict['scores']  # [N]

    # 4. Get pred according to threshold
    indices = pred_scores >= threshold
    pred_boxes = pred_boxes[indices]
    pred_labels = pred_labels[indices]
    pred_scores = pred_scores[indices]

    # 5. visualize
    visualize_detection_result(img_pil, pred_boxes, pred_labels, pred_scores)


if __name__ == '__main__':
    # demo
    demo('/data/DAVIS/JPEGImages/480p/bear/00000.jpg', threshold=0.9)