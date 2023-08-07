import json
import os
import argparse
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
from pycocotools import mask as mask_util

def convert_segmentation_to_mask(segmentation, width, height):
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)

    if isinstance(segmentation, list):
        for segment in segmentation:
            # if len(segment) >= 6:
            #     draw.polygon(segment, outline=1, fill=1)
            draw.polygon(segment, outline=1, fill=1)
        
    elif isinstance(segmentation, dict) and 'counts' in segmentation and 'size' in segmentation:
        rle = mask_util.frPyObjects(segmentation, height, width)
        mask = mask_util.decode(rle)
        
    return np.array(mask)

def main(args):
    with open(args.input_json, 'r') as f:
        coco_data = json.load(f)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    categories = {category['id']: category['name'] for category in coco_data['categories']}
    
    for image_info in tqdm(coco_data['images']):
        image_id = image_info['id']
        file_name = image_info['file_name']
        image_width = image_info['width']
        image_height = image_info['height']

        combined_mask = np.zeros((image_height, image_width), dtype=np.uint8)

        for annotation in coco_data['annotations']:
            if annotation['image_id'] == image_id:
                segmentation = annotation['segmentation']
                mask = convert_segmentation_to_mask(segmentation, image_width, image_height)
                combined_mask = np.maximum(combined_mask, mask)

        image_mask_pil = Image.fromarray(combined_mask)
        mask_file_name = os.path.splitext(file_name)[0] + '_combined_mask.png'
        mask_path = os.path.join(output_dir, mask_file_name)
        image_mask_pil.save(mask_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert COCO segmentation to masks")
    parser.add_argument("--input_json", required=True, help="Path to COCO JSON file")
    parser.add_argument("--output_dir", required=True, help="Directory to save output masks")
    args = parser.parse_args()

    main(args)




# How to run
# python cocoTObitmaskNew.py --input_json instances_val2017.json --output_dir masks_output

# Please adjust the code as needed and ensure that your COCO JSON file (instances_val2017.json) 
# and output directory (masks_output) paths are correctly specified.

# The output mask filenames will have "_combined_mask.png" appended to the original image filenames.