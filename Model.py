import torchvision
import torchvision.transforms as transformss
import torch.nn as nn
import numpy as np
import torch
import torch.utils.data 
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from Res2Net.res2net import res2net50

backbone = res2net50(pretrained=True)
faster_rcnn_fe_extractor = nn.Sequential(*list(backbone.children())[:-2])

faster_rcnn_fe_extractor.out_channels = 2048

anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                aspect_ratios=((0.5, 1.0, 2.0),))


roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                output_size=7,
                                                sampling_ratio=2)

mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                    output_size=14,
                                                    sampling_ratio=2)
# put the pieces together inside a MaskRCNN model
model = MaskRCNN(faster_rcnn_fe_extractor,
                num_classes=2,
                rpn_anchor_generator=anchor_generator,
                box_roi_pool=roi_pooler,
                mask_roi_pool=mask_roi_pooler)