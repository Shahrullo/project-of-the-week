import json
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from utils import get_binary_mask, get_bbox, convert_to_coco_rle, binary_mask_to_rle


parser = argparse.ArgumentParser(description='Convert iMaterialist csv format to COCO json format')

parser.add_argument('--csv_path', type=str, help='Path to the CSV file')
parser.add_argument('--output_json_path', type=str, help='Path to the output filtered CSV file')

args = parser.parse_args()


with open('label_descriptions.json') as f:
    label_desc = json.load(f)
    
cat_df = pd.DataFrame(label_desc['categories']).set_index('id').sort_index()
attr_df = pd.DataFrame(label_desc['attributes']).set_index('id').sort_index()

# Loading Dataset

# train_df = pd.read_csv('train.csv')
train_df = pd.read_csv(args.csv_path)
train_df['AttributesIds'] = train_df['AttributesIds'].fillna('')
print(len(train_df))
train_df.head()

def get_resize_image_info(image_width: int, image_height: int, new_image_size: int):

    if image_width > image_height:
        scale = image_width / new_image_size
        new_width = new_image_size
        new_height = int(image_height / scale)
    else:
        scale = image_height / new_image_size
        new_height = new_image_size
        new_width = int(image_width / scale)

    return new_width, new_height, scale

def generate_coco_annotations(df: pd.DataFrame, image_size: int = None):

    imgs_df = df.groupby(['ImageId'], as_index=False).agg({
        'Height': 'first',
        'Width': 'first',
    })
    imgs_df.columns = ['ImageId', 'Height', 'Width']

    attributes_map = {attr_id: i for i, attr_id in enumerate(attr_df.index)}

    info = {
        'num_attributes': len(attributes_map),
    }

    categories = []
    for category_id, row in cat_df.iterrows():
        categories.append({
            'id': category_id + 1,
            'name': row['name'],
            'supercategory': row['supercategory'],
        })

    images = []
    image_ids = {}
    for image_id, row in imgs_df.iterrows():
        width = row['Width']
        height = row['Height']
        if image_size is not None:
            width, height, _ = get_resize_image_info(width, height, image_size)

        images.append({
            'id': image_id + 1,
            'width': width,
            'height': height,
            'file_name': row['ImageId'] + '.jpg',
        })
        image_ids[row['ImageId']] = image_id + 1

    annotations = []
    for i, (annotation_id, row) in tqdm(enumerate(df.iterrows())):
        mask = get_binary_mask(row['EncodedPixels'], row['Height'], row['Width'])
        bbox = get_bbox(mask)

        if image_size is not None:
            new_width, new_height, scale = get_resize_image_info(row['Width'], row['Height'], image_size)

            # resize box
            bbox = (np.array(bbox, dtype=np.float32) / scale).tolist()

            # resize and encode mask
            pil_image = Image.fromarray(mask.astype(np.uint8))
            pil_image = pil_image.resize((new_width, new_height), Image.NEAREST)
            mask = np.asarray(pil_image)
            rle = binary_mask_to_rle(mask)
        else:
            rle = convert_to_coco_rle([int(p) for p in row['EncodedPixels'].split()], row['Height'], row['Width'])

        annotations.append({
            'id': annotation_id + 1,
            'image_id': image_ids[row['ImageId']],
            'category_id': int(row['ClassId']) + 1,
            'segmentation': rle,
            'area': int(mask.sum()),
            'bbox': bbox,
            'iscrowd': 0,
            'attrobute_ids': [attributes_map[int(attr_id)]
                              for attr_id in row['AttributesIds'].split(',')] if row['AttributesIds'] != '' else [],
        })

        if i % 1000 == 0:
            print(i)

    return {
        'info': info,
        'images': images,
        'categories': categories,
        'annotations': annotations,
    }

if __name__ == "__main__":

    train_annotations = generate_coco_annotations(train_df)
    # with open('train_coco.json', 'w') as f:
    with open(args.output_json_path, 'w') as f:
        json.dump(train_annotations, f)