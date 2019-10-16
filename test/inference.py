import cv2
import os
import torch
from datetime import datetime
from test.images import Images
from torch.utils.data import DataLoader
from test.utils import test_collate_fn, frame_to_tensor
from termcolor import colored
from tqdm import tqdm


def suppression_threshold(detects, threshhold=0.7):
    """
    delete boxes < threshhold of scores
    :param detects: list of dictionary of predicts
    :param threshhold: float
    :return: list of dictionary where each item: {boxes: [N,4], scores: [N]}
    """
    if len(detects) == 1:
        t_slice = len(detects[0]['scores'][detects[0]['scores'] > threshhold])
        return {'boxes': detects[0]['boxes'][: t_slice],
                'scores': detects[0]['scores'][: t_slice]}

    samples = []

    # FIXME: doesn't work if length detects == 1
    for detect in detects:
        t_slice = len(detect['scores'][detect['scores'] > threshhold])
        sample = {
            'boxes': detect['boxes'][: t_slice],
            'scores': detect['scores'][: t_slice]
        }
        samples.append(sample)
    return samples


def draw_box(img, detect):
    """draw boxes around objects
    :param detect: targets
    :return image with drawing bounding box
    """
    img = img.permute(1, 2, 0).cpu().numpy().copy()
    img = img * 255
    boxes = detect['boxes']
    scores = detect['scores'].cpu().detach().numpy()

    for i, box in enumerate(boxes):
        score = scores[i] * 100
        score = round(score, 1)
        # rectangle around object
        p1 = tuple(box[:2])
        p2 = tuple(box[2:])
        cv2.rectangle(img, p1, p2, (255, 0, 0), 2)
        # rectangle around text
        # NOTE: possible need to correct transform of boxes
        text_size = cv2.getTextSize('bird {}%'.format(score), cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0]
        p3 = (p1[0], p1[1] - text_size[1] - 4)
        p4 = (p1[0] + text_size[0] + 4, p1[1])
        cv2.rectangle(img, p3, p4, (255, 0, 0), -1)
        cv2.putText(img, 'bird {}%'.format(score), p1, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
    return img


def detect_on_images(model, device, img_dir, outdir):
    """
    detecting images sequences
    :param model: nn.Module
    :param device: current device
    :param img_dir: path to images directory
    :param outdir: path to output directory
    """
    print(colored('INFO: loading images', 'yellow'))

    img_dataset = Images(img_dir)
    img_dataloader = DataLoader(img_dataset, batch_size=10, shuffle=False, num_workers=4, collate_fn=test_collate_fn)

    print(colored('INFO: detecting...', 'yellow'))
    start_time = datetime.now()
    for images in tqdm(img_dataloader):
        # turn all images to device
        images = list(image.to(device) for image in images)

        with torch.no_grad():
            detects = model(images)
            detects = suppression_threshold(detects, threshhold=0.7)

        img_rect = []
        for i, detect in enumerate(detects):
            img_rect.append(draw_box(images[i], detect))

        for i, img in enumerate(img_rect):
            save_path = os.path.join(outdir, 'detection_{}.png'.format(i))
            cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    end_time = datetime.now()
    print(colored('INFO: detection {} images finished in {}s'.format(len(img_dataset),
                                                                     (end_time - start_time).total_seconds()), 'green'))


def detect_on_video(model, device, video_dir, outdir, threshhold):
    cap = cv2.VideoCapture(video_dir)
    save_path = os.path.join(outdir, 'detection_{}.avi'.format(datetime.today().strftime('%Y-%m-%d')))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    read_frames = 0
    print(colored('INFO: detecting...', 'yellow'))
    start_time = datetime.now()
    while cap.isOpened():
        flag, frame = cap.read()
        read_frames += 1
        if flag:
            frame = frame_to_tensor(frame, device)
            detection = model(frame)
            detection = suppression_threshold(detection, threshhold)
            if len(detection) != 0:
                frame = draw_box(frame, detection)

            out.write(frame)
            if read_frames % 30 == 0:
                print('Number of frames processed {}'.format(read_frames), flush=True)

            if cv2.waitKey(1) % 0xFF == ord('q'):
                break
        else:
            break
    print('Detection finished in {}'.format(start_time - datetime.now()))
    print('Total frames:', read_frames)
    cap.release()
    out.release()
    print('Detected video saved to ' + outdir)
