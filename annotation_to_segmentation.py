"""
Converts raw human annotations to binary segmentations and encodes
segmentations using RLE formats using the pycocotools Mask API. The final
output is stored in a json file.
"""
from argparse import ArgumentParser
import json
import numpy as np
import os
from PIL import Image, ImageDraw
from pycocotools import mask
import sys
from tqdm import tqdm

from eval_constants import LOCALIZATION_TASKS
from utils import encode_segmentation

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_mask(polygons, img_dims):
    """
    Creates a binary mask (of the original matrix size) given a list of polygon
	annotations format.

    Args:
        polygons (list): [[[x11,y11],[x12,y12],...[x1n,y1n]],...]

    Returns:
        mask (np.array): binary mask, 1 where the pixel is predicted to be the,
						 pathology, 0 otherwise
    """
    poly = Image.new('1', (img_dims[1], img_dims[0]))
    for polygon in polygons:
        coords = [(point[0], point[1]) for point in polygon]
        ImageDraw.Draw(poly).polygon(coords,  outline=1, fill=1)

    binary_mask = np.array(poly, dtype="int")
    return binary_mask


def ann_to_mask(input_path, output_path):
    """
    Args:
        input_path (string): json file path with raw human annotations
        output_path (string): json file path for saving encoded segmentations
    """
    print(f"Reading annotations from {input_path}...")
    with open(input_path) as f:
        ann = json.load(f)

    print(f"Creating and encoding segmentations...")
    results = {}
    for img_id in tqdm(ann.keys()):
        if img_id not in results:
            results[img_id] = {}
        for task in LOCALIZATION_TASKS:
            # create segmentation
            polygons = ann[img_id][task] if task in ann[img_id] else []
            img_dims = ann[img_id]['img_size']
            segm_map = create_mask(polygons, img_dims)

            # encode to coco mask
            encoded_map = encode_segmentation(segm_map)
            results[img_id][task] = encoded_map

    assert len(results.keys()) == len(ann.keys())

    # save segmentations to json file
    print(f"Segmentation masks saved to {output_path}")
    with open(output_path, "w") as outfile:
        json.dump(results, outfile)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--ann_path', type=str,
                        help='json file path with raw human annotations')
    parser.add_argument('--output_path', type=str,
                        default='./human_segmentations.json',
                        help='json file path for saving encoded segmentations')
    args = parser.parse_args()

    ann_to_mask(args.ann_path, args.output_path)
