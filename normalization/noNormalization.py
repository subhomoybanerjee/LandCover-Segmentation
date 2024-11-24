import numpy as np
import os
import sys
sys.path.append('../')
from data import read_image
from sklearn.preprocessing import StandardScaler
import pickle
import argparse
import warnings


def noNormalize(image_dir,BANDS):
    image_data = read_image(image_dir, BANDS)
    return image_data/10000


def main():
    parser = argparse.ArgumentParser(description="Normalize images with specified bands and cloud handling.")
    parser.add_argument('--bands', type=int, nargs='+', required=True, help="List of bands to include, e.g., 1 2 3 4")
    parser.add_argument('--output', type=str, required=True, help="Name of the output pickle file")

    args = parser.parse_args()
    BANDS = args.bands
    output_file = args.output

    path = '../datasets'
    all_img_dir = os.path.join(path, "allPatches/images/")

    temp = os.listdir(all_img_dir)
    dic = {}

    c=0
    for i in temp:
        file = os.path.join(all_img_dir,i)
        dic[i] = noNormalize(file, BANDS)
        c+=1
        print(f'Processing file {c + 1}/{len(temp)}: {file}')

    save = os.path.join(path,'normalized','noNormalized')
    if not os.path.exists(save):
        os.makedirs(save)

    with open(os.path.join(save, f'{output_file}.pkl'), 'wb') as file:
        pickle.dump(dic, file)

    print(f'\nNormalized patches created and saved to {os.path.join(save, output_file)}.pkl')


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()