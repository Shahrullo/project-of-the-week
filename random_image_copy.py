import os
import shutil
import random
import argparse
from tqdm import tqdm

def copy_random_images(input_folder, output_folder, num_images):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get a list of all image files in the input folder
    image_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]

    # Randomly select 'num_images' from the list of image files
    selected_files = random.sample(image_files, num_images)

    # Copy the selected files to the output folder
    for file_name in tqdm(selected_files, desc='Copying images', unit='image'):
        src_path = os.path.join(input_folder, file_name)
        dst_path = os.path.join(output_folder, file_name)
        shutil.copy(src_path, dst_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Randomly copy a specified number of images from an input folder to an output folder.')
    parser.add_argument('--input_folder', type=str, help='Path to the input folder containing images')
    parser.add_argument('--output_folder', type=str, help='Path to the output folder for copied images')
    parser.add_argument('--num_images', type=int, help='Number of images to randomly select and copy')

    args = parser.parse_args()

    copy_random_images(args.input_folder, args.output_folder, args.num_images)
