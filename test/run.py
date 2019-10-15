import torch
import argparse
import sys
from termcolor import colored
from model import faster_rcnn
from test.inference import detect_on_images


def parse_args():
    parser = argparse.ArgumentParser(description='Faster-RCNN test')
    parser.add_argument('--model', dest='model',
                        help='path to model weights',
                        default=None, type=str)
    parser.add_argument('--images', dest='images',
                        help='path to directory where images stored',
                        default=None, type=str)
    parser.add_argument('--video', dest='video',
                        help='path to directory where video stored',
                        default=None, type=str)
    parser.add_argument('--outdir', dest='outdir',
                        help='directory to save results, default save to /output',
                        default='output', type=str)
    parser.add_argument('--use_gpu', dest='use_gpu',
                        help='whether use GPU, if the GPU is unavailable then the CPU will be used',
                        action='store_true')

    parser.print_help()
    args = parser.parse_args()
    return args


def choice():
    yes = {'yes', 'y', 'ye', ''}
    no = {'no', 'n'}
    print("Do you want to continue?")
    choice = input().lower()
    if choice in no:
        sys.exit()
    elif choice in yes:
        print(colored('Using CPU', 'green', attrs=['blink']))
        return
    else:
        sys.stdout.write("Please respond with 'yes' or 'no'")


if __name__ == '__main__':
    args = parse_args()
    print(colored('Called with args:', 'yellow', attrs=['underline']))
    print(args.__dict__)

    if args.model is None:
        raise RuntimeError('path to model not specified')
    elif args.images is args.video:
        raise RuntimeError('path to images and videos not specified')
    elif args.outdir == 'output':
        print(colored('INFO: output directory not specified, all data will be saved to /output',
                      'yellow'))

    if torch.cuda.is_available() and not args.use_gpu:
        print(colored("WARNING: You have a GPU device, so you should probably run with --use_gpu", 'red'))
        choice()
    else:
        print(colored('GPU unavailable', 'red'))
        choice()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(colored('INFO: creating model', 'yellow'))
    model = faster_rcnn.get_pretrained_faster_rcnn(num_classes=1)
    print(colored('INFO: loading checkpoint', 'yellow'))

    state = torch.load(args.model)

    if device == 'cpu':
        model.load_state_dict(state['state_dict'], map_location=torch.device('cpu'))
    else:
        model.load_state_dict(state['state_dict'])

    print(colored('INFO: checkpoint loading successful', 'green'))
    print(colored('INFO: model => Faster-RCNN: train epoch {} | loss {}% [device {}]'.format(
        state['start_epoch'] - 1, state['losses'], device), 'green'))

    model.to(device)
    model.eval()

    if args.images is not None:
        detect_on_images(model, device, args.images, args.outdir)
    elif args.video is not None:
        pass






