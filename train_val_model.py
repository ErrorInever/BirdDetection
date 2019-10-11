import torch
import os
import time
import datetime
from torch.utils.data import DataLoader
from datasets.dataset import Bird
from utils.functions import collate_fn
from utils.config import cfg
from model.faster_rcnn import get_pretrained_faster_rcnn
from model.train import train_one_epoch
from lib.vision.references.detection.engine import evaluate

if __name__ == '__main__':
    # faster rcnn train only GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device != 'cuda':
        raise Exception('cuda not available')
    data_dir = str(input('data dir'))
    output_dir = 'output'
    # define datasets
    print('loading data')
    train_dataset = Bird(data_dir, train=True)
    test_dataset = Bird(data_dir, train=False)

    train_dataloader = DataLoader(train_dataset, batch_size=10,
                                  shuffle=True, num_workers=5, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=10,
                                 shuffle=True, num_workers=5, collate_fn=collate_fn)

    # define model
    print('creating model')
    model = get_pretrained_faster_rcnn(1)
    # construct optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=1.0e-3, momentum=0.9, weight_decay=1e-4)
    # define scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)

    num_epochs = 40

    model.to(device)

    print('start training ...')
    start_time = time.time()
    for epoch in range(num_epochs):
        # train for one epoch
        train_one_epoch(model, optimizer, train_dataloader, device, epoch, output_dir, tensorboard=True, print_freq=2)
        # update learning rate
        scheduler.step()
        # evaluate after every epoch
        evaluate(model, test_dataloader, device)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
