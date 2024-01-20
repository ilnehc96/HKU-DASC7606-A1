import argparse
import collections
import os
import numpy as np

from tqdm import tqdm
import torch
import torch.optim as optim
from torchvision import transforms

from retinanet import model
from retinanet.dataloader import CocoDataset, collater, Resizer, \
    AspectRatioBasedSampler, Augmenter, Normalizer
from retinanet.eval import Evaluation
    
from torch.utils.data import DataLoader

from pycocotools.cocoeval import COCOeval
import json



def main(args=None):
    parser = argparse.ArgumentParser(description='Simple test script to test the trained RetinaNet network.')
    parser.add_argument('--coco_path', help='Path to COCO directory', default='./data')
    parser.add_argument('--checkpoint_path', help='Path of checkpoint', default='./output/model_final.pt')
    parser.add_argument('--depth', help='ResNet depth, must be one of 18, 34, 50, 101, 152', type=int, default=101)
    parser.add_argument('--threshold', help='Threshold, between 0 and 1', type=float, default=0.05)
    parser.add_argument('--set_name', help='evaluation data: val or test', type=str, default='test')

    parser = parser.parse_args(args)

    if not os.path.isfile(parser.checkpoint_path):
        raise ValueError('Must provide a valid --checkpoint_path to find the saved checkpoint.')

    if parser.coco_path is None:
        raise ValueError('Must provide --coco_path when training on COCO.')

    dataset_test = CocoDataset(parser.coco_path, set_name=parser.set_name,
                                transform=transforms.Compose([Normalizer(), Resizer()]))

    # Create the model
    if parser.depth == 50:
        retinanet = model.resnet50(num_classes=dataset_test.num_classes(), pretrained=False)
    elif parser.depth == 101:
        retinanet = model.resnet101(num_classes=dataset_test.num_classes(), pretrained=False)
    else:
        raise ValueError('Unsupported model depth in source code, must be one of 18, 34, 50, 101, 152')

    # Load the model
    retinanet = torch.load(os.path.join(parser.checkpoint_path))

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    retinanet.eval()
    retinanet.training = False

    print('Num test images: {}'.format(len(dataset_test)))

    eval = Evaluation()
    eval.evaluate(dataset_test, retinanet, threshold=parser.threshold)

if __name__ == '__main__':
    main()
