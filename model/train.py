import math
import sys
from lib.vision.references.detection import utils


def train(model, optimizer, data_loader, device, epoch, print_freq):
    """
    :param model: nn.Module
    :param optimizer: nn.Module
    :param data_loader:
    :param device:
    :param epoch:
    :param print_freq:
    :return:
    """
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        # turn all data to device
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # forward through model, model return dict of losses
        # loss_box_reg, loss_classifier, loss_objectness, loss_rpn_box_reg
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # set all gradient by zero
        optimizer.zero_grad()
        # backward though Net
        losses.backward()
        # gradient step
        optimizer.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # TODO checkpoints and tensorboard
