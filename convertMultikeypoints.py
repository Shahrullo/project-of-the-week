import argparse
import json
import uuid
from tqdm import tqdm

def convert_to_correct_base_format(coco_json_path):
    with open(coco_json_path, 'r') as coco_file:
        coco_data = json.load(coco_file)

    base_data = {
        "bboxes": [],
        "image": {
            "rotation": 0,  # Assuming rotation is 0
            "image_width": coco_data['images'][0]['width'],
            "image_height": coco_data['images'][0]['height']
        }
    }

    for annotation in tqdm(coco_data['annotations'], desc='Converting', unit='annotations'):
        keypoints = annotation['keypoints']
        num_keypoints = len(keypoints) // 3

        bbox_x = keypoints[0:num_keypoints*3:3]
        bbox_y = keypoints[1:num_keypoints*3:3]

        min_x, max_x = min(bbox_x), max(bbox_x)
        min_y, max_y = min(bbox_y), max(bbox_y)

        box = [[min_x, min_y], [max_x, max_y]]

        # pk = annotation['id']
        pk = uuid.uuid4()
        _class = 1  # Assuming class 1 is for 'person'
        boxtype = "lines"

        base_data["bboxes"].append({
            "box": box,
            "pk": str(pk),
            "class": _class,
            "boxtype": boxtype
        })

    return base_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert COCO JSON file to correct base JSON format.")
    parser.add_argument("--coco_json", help="Path to the input COCO JSON file.")
    args = parser.parse_args()

    # output_json = args.coco_json.replace(".json", "_correct_base_format.json")
    base_data = convert_to_correct_base_format(args.coco_json)

    output_json = "base_format.json"
    with open(output_json, 'w') as output_file:
        json.dump(base_data, output_file, indent=4)
