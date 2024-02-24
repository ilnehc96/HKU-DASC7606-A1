import argparse
import collections
import os
import numpy as np
from tqdm import tqdm

from retinanet.dataloader import CocoDataset

from pycocotools.cocoeval import COCOeval

def main(args=None):
    parser = argparse.ArgumentParser(description='Simple test script to get final metrics.')
    parser.add_argument('--coco_path', help='Path to COCO directory', default='./data_test')

    parser = parser.parse_args(args)

    if parser.coco_path is None:
        raise ValueError('Must provide --coco_path.')

    dataset_test = CocoDataset(parser.coco_path, set_name='test')

    # load results in COCO evaluation tool
    coco_true = dataset_test.coco
    coco_pred = coco_true.loadRes('test_bbox_results.json')

    # run COCO evaluation
    coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
    # coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


if __name__ == '__main__':
    main()
