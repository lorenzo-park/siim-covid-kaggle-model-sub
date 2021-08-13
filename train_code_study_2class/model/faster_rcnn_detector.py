from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn

import gc
import torch


class FasterRCNNDetector(torch.nn.Module):
    def __init__(self, num_classes, trainable_backbone_layers, pretrained=False, **kwargs):
        super(FasterRCNNDetector, self).__init__()
        # load pre-trained model incl. head
        self.model = fasterrcnn_resnet50_fpn(pretrained=pretrained, pretrained_backbone=pretrained, trainable_backbone_layers=trainable_backbone_layers)

        # get number of input features for the classifier custom head
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features

        # replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    def forward(self, images, targets=None):
        return self.model(images, targets)


def get_model(num_classes, trainable_backbone_layers, checkpoint_path=None, pretrained=False):
    model = FasterRCNNDetector(pretrained=pretrained, num_classes=num_classes, trainable_backbone_layers=trainable_backbone_layers)

    # Load the trained weights
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])

        del checkpoint
        gc.collect()

    return model