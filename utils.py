import os
import cv2 as cv
import numpy as np

def process_img(src):
    img = src
    return img

def load_data(data_path):
    data = []
    img_list = os.listdir(data_path)
    for img_idx, img_path in enumerate(img_list):
        src = cv.imread(data_path+img_path) # ndarray, (540, 960, 3)
        img = process_img(src)
        data.append(img)
    data = np.array(data)
    return data


if __name__ == "__main__":
    load_data('sample_data/')

