import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def set_seed(val):
    """freeze random sequences"""
    random.seed(val)
    np.random.seed(val)
    torch.manual_seed(val)
    torch.cuda.manual_seed(val)
    torch.backends.cudnn.deterministic = True


def show_data(img, target):
    """ show image and bounding boxes"""
    img = np.array(img)
    num_obj = len(target['boxes'])
    patches = []
    for i in range(num_obj):
        box = target['boxes'][i]
        xy = box[0], box[1]
        width = box[2] - box[0]
        height = box[3] - box[1]
        patches.append(Rectangle(xy, width, height, fill=False, color='r', linewidth=2))
    plt.imshow(img)
    for rec in patches:
        plt.gca().add_patch(rec)
    plt.pause(0.001)


def show_batch(img_tensor, target):
    """ show batch """
    std = [0.229, 0.224, 0.225]
    mean = [0.485, 0.456, 0.406]

    image = img_tensor.permute(1, 2, 0).numpy()
    image = std * image + mean
    # TODO


def collate_fn(batch):
    """ create a list of dictionary from batch """
    return tuple(zip(*batch))


def test_collate_fn(batch):
    return tuple(*batch)
