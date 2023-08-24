import os
import numpy as np
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


def visualize_yolo_dataset(image_folder, label_folder):
    image_paths = sorted([os.path.join(image_folder, filename) for filename in os.listdir(image_folder) if filename.endswith(('.jpg', '.png', '.jpeg'))])
    label_paths = sorted([os.path.join(label_folder, filename) for filename in os.listdir(label_folder) if filename.endswith('.txt')])
    image_paths = image_paths[:5]
    label_paths = label_paths[:5]

    for image_path, label_path in tqdm(zip(image_paths, label_paths), total=len(image_paths), desc='Visualization'):
        print(f'Image_path: {image_path}')
        image = Image.open(image_path)
        img_width, img_height = image.size

        with open(label_path, 'r') as label_file:
            lines = label_file.readlines()

        for line in lines:
            parts = line.strip().split()
            
            class_id, *points = map(float, parts)
            class_id = int(class_id)
            x_min, y_min, x_max, y_max = min(points[0::2]), min(points[1::2]), max(points[0::2]), max(points[1::2])           
            # Unnormalize the bounding box coordinates
            x_min = int(x_min * img_width)
            y_min = int(y_min * img_height)
            x_max = int(x_max * img_width)
            y_max = int(y_max * img_height)
            print(f'x_min: {x_min}\n, y_min: {y_min}\n, x_max: {x_max}\n, y_max: {y_max}')

            draw = ImageDraw.Draw(image)
            color = tuple(np.random.randint(0, 255, size=(3,), dtype=np.uint8))
            draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=2)

            label_text = f'Class {class_id}'
            font = ImageFont.load_default() 
            draw.text((x_min, y_min - 15), label_text, font=font, fill=color)

        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.axis('off')
        plt.show()
        
# Provide the paths to your images and labels folders
image_folder_path = 'trainPart'
label_folder_path = 'trainPartTxtMainLabels'

visualize_yolo_dataset(image_folder_path, label_folder_path)
