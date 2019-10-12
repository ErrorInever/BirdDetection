import math
import sys
import os
from model.utils import save_checkpoint
from lib.vision.references.detection import utils
from tensorboardX import SummaryWriter


def train_one_epoch(model, optimizer, scheduler, data_loader, device, epoch, output_dir,
                    tensorboard=False, print_freq=10):
    """
    defining one epoch of train

    :param model: (nn.Module): instance of model
    :param optimizer: (nn.Module): instance of optimizer
    :param scheduler: object scheduler
    :param data_loader: object dataloader
    :param device: str, faster-rcnn works only GPU
    :param epoch: int, number of epoch
    :param output_dir: directory where to save state of model and log files
    :param tensorboard: if true: save to output_dir log files after each epoch
    :param print_freq: int, after how many iteration print statistic
    """
    if tensorboard:
        logger = SummaryWriter(output_dir)

    # set model to train mode
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    # through all data
    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        # images and targets to GPU
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # forward through model, model return a dictionary of losses
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
        # backward through model
        losses.backward()
        # make gradient step
        optimizer.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # create checkpoint after each epoch
    save_name = os.path.join(output_dir, 'faster_rcnn_{}.pth'.format(epoch))
    save_checkpoint({
        'start_epoch': epoch + 1,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'losses': loss_value
    }, save_name)
    print('save model: {}'.format(save_name))

    # save metric after each epoch
    if tensorboard:
        logger.add_scalars('train/losses', loss_dict)
        logger.add_scalar('train/loss_value', losses)
        logger.close()
    print('metric save {}'.format(output_dir))
