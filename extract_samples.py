import argparse
import json
import random
from tqdm import tqdm

def get_random_samples(input_file, num_samples):
    with open(input_file, 'r') as f:
        data = json.load(f)

    images = data['images']
    sample_images = random.sample(images, min(num_samples, len(images)))

    image_ids = {image['id'] for image in sample_images}
    sample_annotations = [ann for ann in data['annotations'] if ann['image_id'] in image_ids]

    sample_data = {
        'info': data['info'],
        'licenses': data['licenses'],
        'images': sample_images,
        'annotations': sample_annotations
    }

    return sample_data

def main():
    parser = argparse.ArgumentParser(description='Extract random samples from COCO person keypoints JSON file')
    parser.add_argument('--input', type=str, required=True, help='Input COCO JSON file path')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file path')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to extract')
    args = parser.parse_args()

    sample_data = get_random_samples(args.input, args.num_samples)

    with open(args.output, 'w') as f:
        json.dump(sample_data, f, indent=4)

if __name__ == '__main__':
    main()

