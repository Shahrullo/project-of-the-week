import os
import argparse
import pandas as pd
from tqdm import tqdm

def get_image_filenames(folder_path):
    image_filenames = []
    for filename in tqdm(os.listdir(folder_path), desc='Image filenames'):
        if filename.endswith('.jpg'):  # Adjust the file extension accordingly
            image_filenames.append(filename.split('.')[0])
    return image_filenames

# Without progress bar
# def filter_csv_by_image_ids(csv_path, image_filenames):
#     df = pd.read_csv(csv_path)
#     filtered_df = df[df['ImageId'].isin(image_filenames)]
#     return filtered_df

# with progress bar
def filter_csv_by_image_ids(csv_path, image_filenames):
    df = pd.read_csv(csv_path)
    filtered_rows = []
    total_rows = len(df)
    with tqdm(total=total_rows, desc='Filtering CSV') as pbar:
        for _, row in df.iterrows():
            if row['ImageId'] in image_filenames:
                filtered_rows.append(row)
            pbar.update(1)
    filtered_df = pd.DataFrame(filtered_rows, columns=df.columns)
    return filtered_df

def write_filtered_csv(filtered_df, output_csv_path):
    filtered_df.to_csv(output_csv_path, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Filter a CSV file based on image filenames in a specified folder.')
    parser.add_argument('folder_path', type=str, help='Path to the folder containing the images')
    parser.add_argument('csv_path', type=str, help='Path to the CSV file')
    parser.add_argument('output_csv_path', type=str, help='Path to the output filtered CSV file')

    args = parser.parse_args()

    image_filenames = get_image_filenames(args.folder_path)
    filtered_df = filter_csv_by_image_ids(args.csv_path, image_filenames)
    write_filtered_csv(filtered_df, args.output_csv_path)
