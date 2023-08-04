import os
import json
import uuid
import argparse
import numpy as np
from tqdm import tqdm

body_connections = np.array(
    [
        {"box": [[1, 2], [2, 3], [3, 5], [5, 7]], "class": 0},
        {"box": [[1, 3]], "class": 0},
        {"box": [[2, 4], [4, 6]], "class": 0},
        {"box": [[6, 7], [7, 9]], "class": 1},
        {"box": [[6, 8], [8, 10]], "class": 1},
        {"box": [[9, 11]], "class": 1},
        {"box": [[16, 14], [14, 12]], "class": 2},
        {"box": [[17, 15], [15, 13]], "class": 2},
        {"box": [[6, 12], [12, 13]], "class": 3},
        {"box": [[7, 13]], "class": 3},
    ]
)

def postprocess_pose_keypoints(keypoints, body_connections):
    result = []
    for keypoint in keypoints:
        for connections in body_connections:
            boxid = connections["class"]
            group_line = []
            for connection in connections["box"]:
                keypoint_a = keypoint[(connection[0] - 1) * 3 : connection[0] * 3]
                keypoint_b = keypoint[(connection[1] - 1) * 3 : connection[1] * 3]
                x_start, y_start, visibility_start = (
                    keypoint_a[0],
                    keypoint_a[1],
                    keypoint_a[2],
                )
                x_end, y_end, visibility_end = (
                    keypoint_b[0],
                    keypoint_b[1],
                    keypoint_b[2],
                )
                if visibility_start > 0.5 and visibility_end > 0.5:
                    group_line.append([round(x_start), round(y_start)])
                    group_line.append([round(x_end), round(y_end)])

            if len(group_line):
                result.append(
                    {
                        "box": group_line,
                        "pk": str(uuid.uuid4()),
                        "class": boxid,
                        "boxtype": "lines",
                    }
                )

    return result


def process_samples(input_file, output_folder):
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    annotations = data['annotations']
    
    for annotation in tqdm(annotations, desc='Processing samples', unit='sample'):
        keypoints = annotation['keypoints']
        processed_keypoints = postprocess_pose_keypoints(keypoints, body_connections)
        
        output_filename = f"{annotation['image_id']}.json"
        output_path = os.path.join(output_folder, output_filename)
        
        with open(output_path, 'w') as f:
            json.dump(processed_keypoints, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process COCO keypoints dataset")
    parser.add_argument("input_file", help="Path to the input JSON file")
    parser.add_argument("output_folder", help="Path to the output folder")
    args = parser.parse_args()

    body_connections = [
        # ... (same as the provided structure)
    ]

    process_samples(args.input_file, args.output_folder)
