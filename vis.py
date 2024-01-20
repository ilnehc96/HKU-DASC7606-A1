import os
import sys
import argparse
import random

import numpy as np
from skimage import io
import matplotlib
import matplotlib.pyplot as plt

from torchvision import transforms
from retinanet.dataloader import CocoDataset, Resizer, Normalizer

def draw(imgroot, img, coco_true, coco_pred=None):
    I = io.imread(os.path.join(imgroot, img['file_name']))
    plt.plot()
    plt.axis('off')
    plt.title(img['file_name'],fontsize=8,color='blue')
    plt.imshow(I, aspect='equal')
    annids = coco_true.getAnnIds(imgIds=img['id'])
    anns = coco_true.loadAnns(annids)
    coco_true.showAnns(anns, draw_bbox=True)
    if coco_pred is not None:
        annids = coco_pred.getAnnIds(imgIds=img['id'])
        anns = coco_pred.loadAnns(annids)
        coco_pred.showAnns(anns, draw_bbox=True)

def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    parser.add_argument('--coco_path', help='Path to COCO directory', default='./data')
    parser.add_argument('--pred_file', help='Path to prediction file', default='./val_bbox_results.json')
    parser.add_argument('--vis_pred', action="store_true", help='Visualize prediction results')

    parser = parser.parse_args(args)
    vis_pred = parser.vis_pred

    imgroot = os.path.join(parser.coco_path, 'images', 'val')

    dataset_val = CocoDataset(parser.coco_path, set_name='val',
                                transform=transforms.Compose([Normalizer(), Resizer()]))
    coco_true = dataset_val.coco
    coco_pred = coco_true.loadRes(parser.pred_file) if vis_pred else None

    cats = coco_true.loadCats(coco_true.getCatIds())
    cat_names = [cat['name'] for cat in cats]
    print(cat_names)
    catids = coco_true.getCatIds(catNms=random.randint(0,len(cat_names)-1))
    imgids = coco_true.getImgIds(catIds=catids)

    # You can set to a certain id to visualize same images for comparison
    img = coco_true.loadImgs(imgids[np.random.randint(0, len(imgids))])[0]

    plt.figure()
    draw(imgroot, img, coco_true, coco_pred)
    plt.savefig('vis.png')
    plt.show()

if __name__ == '__main__':
    main()
