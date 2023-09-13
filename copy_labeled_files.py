import os
import shutil
import argparse
from tqdm import tqdm

def copy_files(input_folder, output_folder, label_file):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Read the list of filenames from label.txt
    with open(label_file, 'r') as label_file:
        filenames = set(line.strip() for line in label_file)

    # Iterate through files in the 'labels' folder
    for root, _, files in tqdm(os.walk(input_folder), desc="Processing Files"):
        for filename in files:
            file_path = os.path.join(root, filename)

            # Check if the filename exists in label.txt
            if filename in filenames:
                # Copy the file to the output folder
                shutil.copy(file_path, os.path.join(output_folder, filename))

if __name__ == '__main__':
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Copy labeled files from a folder to a new folder.')

    # Add arguments for input folder, output folder, and label file
    parser.add_argument('--input_folder', type=str, help='Path to the input folder (containing .txt files)')
    parser.add_argument('--output_folder', type=str, help='Path to the output folder (where labeled files will be copied)')
    parser.add_argument('--label_file', type=str, help='Path to the label.txt file containing filenames to match')

    # Parse command-line arguments
    args = parser.parse_args()

    # Check if the input folder exists
    if not os.path.exists(args.input_folder):
        print(f"Error: Input folder '{args.input_folder}' does not exist.")
        exit(1)

    # Check if the label file exists
    if not os.path.exists(args.label_file):
        print(f"Error: Label file '{args.label_file}' does not exist.")
        exit(1)

    # Copy labeled files to the output folder
    copy_files(args.input_folder, args.output_folder, args.label_file)

    print(f"Labeled files copied to '{args.output_folder}'.")

