import cv2
import json
from PIL import Image
from tqdm import tqdm
from copy import deepcopy
from pycocotools import mask


def rle2polygon(segmentation):
    m = mask.decode(segmentation)
    m[m > 0] = 255
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    polygons = []
    for contour in contours:
        epsilon = 0.001 * cv2.arcLength(contour, True)
        contour_approx = cv2.approxPolyDP(contour, epsilon, True)
        polygon = contour_approx.flatten().tolist()
        polygons.append(polygon)
    return polygons

# Load COCO JSON
with open('trainPart_coco.json', 'r') as json_file:
    coco_data = json.load(json_file)

# Create a new COCO JSON structure
new_coco_data = deepcopy(coco_data)
new_coco_data['images'] = []
new_coco_data['annotations'] = []

for image_data in tqdm(coco_data['images']):
    image_id = image_data['id']
    image_filename = image_data['file_name']
    image_width = image_data['width']
    image_height = image_data['height']

    max_x = float('-inf')
    max_y = float('-inf')
    min_x = float('inf')
    min_y = float('inf')

    # Find the widest and highest points within all bounding boxes
    for annotation in coco_data['annotations']:
        if annotation['image_id'] == image_id:
            bbox_x, bbox_y, bbox_width, bbox_height = annotation['bbox']
            max_x = max(max_x, bbox_x + bbox_width)
            max_y = max(max_y, bbox_y + bbox_height)
            min_x = min(min_x, bbox_x)
            min_y = min(min_y, bbox_y)

    # Calculate new bounding box dimensions
    new_bbox_x = max(0, min_x)
    new_bbox_y = max(0, min_y)
    new_bbox_width = min(image_width, max_x - new_bbox_x)
    new_bbox_height = min(image_height, max_y - new_bbox_y)

    # Calculate new bounding box coordinates as per requirement
    new_bbox_width = min(image_width, int(new_bbox_width * 1.2))
    new_bbox_height = min(image_height, int(new_bbox_height * 1.5))

    # Crop and save image
    img = Image.open(f'trainPart/{image_filename}')

    # Check for out-of-bounds cropping
    new_bbox_x = min(new_bbox_x, image_width - new_bbox_width)
    new_bbox_y = min(new_bbox_y, image_height - new_bbox_height)

    # Update the image data with new width and height
    new_image_data = deepcopy(image_data)
    new_image_data['width'] = new_bbox_width
    new_image_data['height'] = new_bbox_height
    new_coco_data['images'].append(new_image_data)

    for annotation in coco_data['annotations']:
        if annotation['image_id'] == image_id:

            # Adjust bounding box cordinates
            bbox_x, bbox_y, bbox_width, bbox_height = annotation['bbox']
            bbox_x -= new_bbox_x
            bbox_y -= new_bbox_y
            annotation['bbox'] = [bbox_x, bbox_y, bbox_width, bbox_height]

            # # Adjust RLE encoded segmentation
            # if isinstance(annotation['segmentation'], dict): # Check if segmentation is RLE
            #     binary_mask = mask.decode(annotation['segmentation'])
            #     cropped_binary_mask = binary_mask[new_bbox_y:new_bbox_y+new_bbox_height, new_bbox_x:new_bbox_x+new_bbox_width]
            #     rle = mask.encode(np.asfortranarray(cropped_binary_mask))
            #     annotation['annotation'] = rle
            # else: # Handle polygons
            
            # Convert RLE to polygons if it's RLE encoded
            if isinstance(annotation['segmentation'], dict): # Check is segmentation is RLE
                annotation['segmentation'] = rle2polygon(mask.frPyObjects(annotation['segmentation'], image_width, image_height))
                # binary_mask = mask.decode(annotation['segmentation'])
                # polygons = mask.toBbox(binary_mask)
                # annotation['segmentation'] = polygons

            # Adjust polygon points

            new_segmentations = []
            for segment in annotation['segmentation']:
                new_segment = []
                for i in range(0, len(segment), 2): # Iterate over pairs of coordinates
                    x = segment[i] - new_bbox_x
                    y = segment[i+1] - new_bbox_y

                    # Optional: Clamp the coordinates to the cropped area
                    x = max(0, min(new_bbox_width, x))
                    y = max(0, min(new_bbox_height, y))

                    new_segment.extend([x, y])

                new_segmentations.append(new_segment)

            annotation['segmentation'] = new_segmentations

            new_coco_data['annotations'].append(annotation)

# Save the new COCO JSON
with open('trainPartCropped_coco.json', 'w') as json_file:
    json.dump(new_coco_data, json_file)