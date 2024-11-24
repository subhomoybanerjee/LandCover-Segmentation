import numpy as np
import os
import sys
sys.path.append('../')
from data import read_image
from sklearn.preprocessing import StandardScaler
import pickle
import argparse
import warnings


def localNormalize(image_dir, scale, BANDS,cloud):


    image_data = read_image(image_dir, BANDS)
    height, width, bands = image_data.shape
    normalized_data = np.zeros_like(image_data, dtype=np.float32)

    mask = (image_data[:, :, 0] > 0) & (image_data[:, :, 1] > 0) & (image_data[:, :, 2] > 0)

    for band in range(len(BANDS)):
        band_data = image_data[:, :, band].flatten()  # orginal image data, not normalized contains clouds
        if cloud=='True':
            accumulated_pixels = band_data.reshape(-1, 1)
        elif cloud=='False':
            accumulated_pixels = band_data[mask.flatten()].reshape(-1, 1)

        scaler = scale()
        scaler.fit(accumulated_pixels)
        band_data_transformed = scaler.transform(band_data.reshape(-1, 1))


        band_data_transformed = band_data_transformed.flatten()
        normalized_data[:, :, band] = band_data_transformed.reshape(height,width)


    return normalized_data


def main():
    parser = argparse.ArgumentParser(description="Normalize images with specified bands and cloud handling.")
    parser.add_argument('--bands', type=int, nargs='+', required=True, help="List of bands to include, e.g., 1 2 3 4")
    parser.add_argument('--clouds', type=str, choices=['True', 'False'], required=True,
                        help="Cloud filtering: True or False")
    parser.add_argument('--output', type=str, required=True, help="Name of the output pickle file")

    args = parser.parse_args()
    BANDS = args.bands
    cloud = args.clouds
    output_file = args.output

    path = '../datasets/allPatches'
    all_img_dir = os.path.join(path, "images/")

    dir = all_img_dir
    temp = os.listdir(dir)
    dic = {}
    for i in range(len(temp)):
        file = os.path.join(dir, temp[i])
        dic[temp[i]] = localNormalize(file, StandardScaler, BANDS, cloud)
        print(f'Processing file {i + 1}/{len(temp)}: {file}')

    save = os.path.join('../datasets/normalized', "l1Normalized")
    if not os.path.exists(save):
        os.makedirs(save)

    with open(os.path.join(save, f'{output_file}.pkl'), 'wb') as file:
        pickle.dump(dic, file)

    print(f'\nNormalized patches created and saved to {os.path.join(save, output_file)}.pkl')


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()