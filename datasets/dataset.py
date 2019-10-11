import os
import pandas as pd
import torch
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms


class Bird(Dataset):

    def __init__(self, data_dir, train=False):
        """
        :param data_dir: directory of dataset
        :param train: if True: create train dataset else create test dataset
        """
        self.data_dir = data_dir
        self.train = train

        if self.train:
            self.img_dir = os.path.join(data_dir, 'train/Bird')
            self.label_dir = os.path.join(data_dir, 'train/Bird/Label')
        else:
            self.img_dir = os.path.join(data_dir, 'validation/Bird')
            self.label_dir = os.path.join(data_dir, 'validation/Bird/Label')
        # get list names of images
        self.img_name = [name.split('.')[0] for name in os.listdir(self.img_dir) if name.endswith('.jpg')]

    def __getitem__(self, idx):
        img_pil = Image.open(os.path.join(self.img_dir, self.img_name[idx] + '.jpg')).convert('RGB')
        # read targets, coordinates: x1, y1, x2, y2
        targets = pd.read_csv(os.path.join(self.label_dir, self.img_name[idx] + '.txt'), sep=" ", header=None,
                              names=['class', 'left', 'top', 'right', 'bottom'])

        targets_boxes = targets[['left', 'top', 'right', 'bottom']]
        # number of object on image
        num_objs = len(targets_boxes)
        # get bounding box for each object on image
        list_boxes = []
        for i in range(num_objs):
            list_boxes.append(targets_boxes.iloc[i].tolist())
        # convert boxes to tensor
        boxes = torch.as_tensor(list_boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        # img transform
        img = self.img_transform(img_pil)
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels

        if self.train:
            # convert image to numpy array
            img = np.array(img_pil)

            # define augmentation
            # applies the given augmenter in 50% of all cases
            sometimes = lambda aug: iaa.Sometimes(0.5, aug)
            sequence = iaa.Sequential([
                iaa.Resize(size=224),
                iaa.Fliplr(p=0.5),
                sometimes(iaa.Snowflakes()),
                sometimes(iaa.Salt(p=0.10)),
                sometimes(iaa.Multiply((0.25, 0.50)))
            ])
            # boxes transform
            bbs = self._get_list_bbs(list_boxes)
            boxes = ia.BoundingBoxesOnImage(bounding_boxes=bbs, shape=img.shape)
            boxes = sequence.augment_bounding_boxes(boxes)
            boxes = boxes.to_xyxy_array(dtype=np.float32)
            boxes = torch.from_numpy(boxes)
            target['boxes'] = boxes
            # image transform
            transform = self.img_transform
            img = sequence.augment_image(img)
            img = transform(img)

        return img, target

    def __len__(self):
        return len(self.img_name)

    def _get_list_bbs(self, boxes):
        """ get list of objects bbs """
        bbs = []
        for bbox in boxes:
            bbs.append(ia.BoundingBox(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3]))
        return bbs

    @property
    def img_transform(self):
        """transform for image"""
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])])
        return transform
