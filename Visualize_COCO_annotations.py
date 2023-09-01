import os
import json
import random
import base64
import IPython
import requests
import numpy as np
from io import BytesIO
from math import trunc
from PIL import Image
from PIL import ImageDraw

# Load the dataset json
class CocoDataset():
    def __init__(self, annotation_path, image_dir):
        self.annotation_path = annotation_path
        self.image_dir = image_dir
        self.colors = ['blue', 'purple', 'red', 'green', 'orange', 'salmon', 'pink', 'gold',
                        'orchid', 'slateblue', 'limegreen', 'seagreen', 'darkgreen', 'olive',
                        'teal', 'aquamarine', 'steelblue', 'powderblue', 'dodgerblue', 'navy',
                        'magenta', 'sienna', 'maroon']
        
        json_file = open(self.annotation_path)
        self.coco = json.load(json_file)
        json_file.close()

        self.process_categories()
        self.process_images()
        self.process_segmentations()

    def display_info(self):
        print("Dataset Info")
        print("============")
        for key, item in self.info.items():
            print(f' {key}: {item}')

        requirements = [['description', str],
                        ['url', str],
                        ['version', str],
                        ['year', int],
                        ['contributor', str],
                        ['date_created', str]]
        for req, req_type in requirements:
            if req not in self.info:
                print(f'ERROR: {req} is missing')
            elif type(self.info[req]) != req_type:
                print(f'ERROR: {req} should be type {str(req_type)}')
        print('')

    def display_licenses(self):
        print('Licenses:')
        print('=========')

        requirements = [['id', int],
                        ['url', str],
                        ['name', str]]
                        
        for license in self.licenses:
            for key, item in license.items():
                print(f' {key}: {item}')
            for req, req_type in requirements:
                if req not in license:
                    print(f'ERROR: {req} is missing')
                elif type(license[req]) != req_type:
                    print(f'ERROR: {req} should be type {str(req_type)}')
            print('')
        print('')

    def display_categories(self):
        print('Categories:')
        print('=========')
        for sc_key, sc_val in self.super_categories.items():
            print('  super_category: {}'.format(sc_key))
            for cat_id in sc_val:
                print(f'    id {cat_id}: {self.categories[cat_id]["name"]}')
            print('')

    def display_image(self, image_id, show_polys=True, show_bbox=True, show_labels=True, show_crowds=True, use_url=False):
        pass


    def process_info(self):
        self.info = self.coco['info']

    def process_licenses(self):
        self.licenses = self.coco['licenses']

    def process_categories(self):
        self.categories = {}
        self.super_categories = {}
        for category in self.coco['categories']:
            cat_id = category['id']
            super_category = category['supercategory']

            # Add category to the categories dict
            if cat_id not in self.categories:
                self.categories[cat_id] = category
            else:
                print(f'ERROR: Skipping duplicate id: {category}')

            # Add category to super_categories dict
            if super_category not in self.super_categories:
                self.super_categories[super_category] = {cat_id} # Create a new set with the category id
            else:
                self.super_categories[super_category] |= {cat_id} # Add category id to the set

    def process_images(self):
        self.images = {}
        for image in self.coco['images']:
            image_id = image['id']
            if image_id in self.images:
                print(f"Error. Skipping duplicate in image id: {image}")
            else:
                self.images[image_id] = image

    def process_segmentations(self):
        self.segmentations = {}
        for segmentation in self.coco['segmentations']:
            image_id = segmentation['image_id']
            if image_id not in self.segmentations:
                self.segmentations[image_id] = []
            self.segmentations[image_id].append(segmentation)


annotation_path = 'trainPart_coco.json'
image_dir = 'trainPart'

coco_dataset = CocoDataset(annotation_path, image_dir)


html = coco_dataset.display_image(use_url=False)
IPython.display.HTML(html)
 