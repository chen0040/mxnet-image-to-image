import numpy as np
import mxnet as mx
from mxnet import nd
import matplotlib.pyplot as plt
import os
from PIL import Image


def transform(data, target_wd=64, target_ht=64):
    # resize to target_wd * target_ht
    data = mx.image.imresize(data, target_wd, target_ht)
    # transpose from (target_wd, target_ht, 3)
    # to (3, target_wd, target_ht)
    data = nd.transpose(data, (2, 0, 1))
    # normalize to [-1, 1]
    data = data.astype(np.float32) / 127.5 - 1
    # if image is greyscale, repeat 3 times to get RGB image.
    if data.shape[0] == 1:
        data = nd.tile(data, (3, 1, 1))
    return data


def load_image(img, target_wd=64, target_ht=64):
    img_arr = mx.image.imread(img)
    img_arr = transform(img_arr, target_wd, target_ht)
    return img_arr


def load_images(data_dir_path, extension='.jpg', target_wd=64, target_ht=64):
    img_list = []
    for path, _, fnames in os.walk(data_dir_path):
        for fname in fnames:
            if not fname.endswith(extension):
                continue
            img = os.path.join(path, fname)
            img_arr = load_image(img, target_wd, target_ht)
            img_list.append(img_arr)
    return img_list


def save_image(img_data, save_to_file):
    Image.fromarray(img_data).save(save_to_file)


def visualize(img_arr):
    plt.imshow(((img_arr.asnumpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8))
    plt.axis('off')
    plt.show()


