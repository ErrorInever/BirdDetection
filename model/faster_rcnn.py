from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn

backbones_names = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']


def get_pretrained_faster_rcnn(num_classes):
    """return nn.Module"""
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    # get number of input channels for the classifier
    in_channels = model.roi_heads.box_predictor.cls_score.in_features
    # replace box predictor and add class 'background'
    model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=in_channels, num_classes=num_classes + 1)
    return model
