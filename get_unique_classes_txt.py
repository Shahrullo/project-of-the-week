import glob
import argparse
from tqdm import tqdm

def get_unique_classes(label_files):

    unique_classes = set()

    for file_path in tqdm(label_files, desc="txt files"):
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                # extract the class label from the first value of each line
                class_label = line.split()[0]
                unique_classes.add(int(class_label))

    return unique_classes

def main(directory_path):
    # Get a list of all label file paths in the specified directory
    label_files = glob.glob(directory_path + '*.txt')

    # Get the unique class labels from the label files
    unique_classes = get_unique_classes(label_files)

    # Print the unique class labels
    for class_label in sorted(unique_classes):
        print(class_label)


if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Extract unique class labels from txt files")

    # Add an argument for the directory path
    parser.add_argument("--directory", type=str, help="Path for the txt files")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function with the provided directory path
    main(args.directory)