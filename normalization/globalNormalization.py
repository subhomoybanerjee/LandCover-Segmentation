import numpy as np
import os
import sys
from sklearn.preprocessing import StandardScaler
import pickle
import argparse
import rasterio
import numpy as np
sys.path.append('../')
from data import read_image

def fit_global_scaler(image_dirs, scale, BANDS,cloud):

    accumulated_pixels = []
    for _ in range(len(BANDS)):
        accumulated_pixels.append([])
    c=0
    for image_dir in image_dirs:
        for filename in image_dir:
            filepath = filename
            image_data = read_image(filepath, BANDS)
            mask = (image_data[:, :, 0] > 0) & (image_data[:, :, 1] > 0) & (image_data[:, :, 2] > 0)
            for band in range(len(BANDS)):
                band_data = image_data[:, :, band].flatten()
                if cloud == 'True':
                    band_data = band_data.reshape(-1, 1)
                elif cloud == 'False':
                    band_data = band_data[mask.flatten()].reshape(-1, 1)
                accumulated_pixels[band].extend(band_data)
            c+=1
            print(c,end=',')

    scalers = []

    print(np.array(accumulated_pixels).shape)
    for band in range(len(BANDS)):
        scaler = scale()
        non_zero_pixels = np.array(accumulated_pixels[band]).reshape(-1, 1)
        print(f'fitting band:{band+1}')
        scaler.fit(non_zero_pixels)
        scalers.append(scaler)

    return scalers

def globalNormalize(image_data, scalers):
    height, width, bands = image_data.shape
    normalized_data = np.zeros_like(image_data, dtype=np.float32)

    for band in range(bands):
        band_data = image_data[:, :, band].flatten()
        band_data_transformed = scalers[band].transform(band_data.reshape(-1, 1))
        band_data_transformed = band_data_transformed.flatten()
        normalized_data[:, :, band] = band_data_transformed.reshape(height, width)

    return normalized_data

def gnSaver(image_dirs,scaler,bands,cloud):
    for i in range(len(image_dirs)):
        images=[]
        for j in os.listdir(image_dirs[i]):
            image=os.path.join(image_dirs[i],j)
            images.append(image)
        image_dirs[i]=images
    scalers = fit_global_scaler(image_dirs, scaler,bands,cloud)
    return scalers

def main():
    parser = argparse.ArgumentParser(description="Normalize image data with specified bands, cloud filtering, and output file.")
    parser.add_argument("--bands", nargs="+", type=int, default=list(range(1, 21)), help="List of band indices to use.")
    parser.add_argument("--clouds", type=str, default=False, help="Include cloud pixels (True or False).")
    parser.add_argument("--output", type=str, required=True, help="Output file path for the pickled data.")
    args = parser.parse_args()

    path = '../datasets'
    train_img_dir = os.path.join(path, 'split/train/images/')
    all_img_dir = os.path.join(path, "allPatches/images/")

    fit_image_dirs = [train_img_dir]

    print('Fitting scaler with selected bands and cloud filter settings.')
    scalers = gnSaver(fit_image_dirs, StandardScaler, args.bands, args.clouds)
    print('Scalers calculated.')

    transform_img_dir = all_img_dir
    temp = os.listdir(transform_img_dir)

    dic = {}
    for i, file_name in enumerate(temp):
        file = os.path.join(transform_img_dir, file_name)
        rfile = read_image(file, args.bands)
        dic[file_name] = globalNormalize(rfile, scalers)
        print(f'Processing file {i + 1}/{len(temp)}: {file}')

    save = os.path.join(path, "normalized/gNormalized")
    if not os.path.exists(save):
        os.makedirs(save)

    with open(os.path.join(save,f'{args.output}.pkl'), 'wb') as file:
        pickle.dump(dic, file)

    print(f'\nNormalized patches created and saved to {os.path.join(save,args.output)}.pkl')

if __name__ == "__main__":
    main()