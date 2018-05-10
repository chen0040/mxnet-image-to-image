import numpy as np
import mxnet as mx
from mxnet import nd, image
from mxnet.gluon.model_zoo import vision as models
import matplotlib.pyplot as plt
import os
from PIL import Image

rgb_mean = nd.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
rgb_std = nd.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))


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


def transform_image(img_path, image_width, image_height):
    x = image.imread(img_path)
    x = image.imresize(x, image_width, image_height)
    x = transform(x)
    return x


def inverted_transform(img):
    return ((img.as_in_context(mx.cpu()) * rgb_std + rgb_mean) * 255).transpose((1, 2, 0))


def load_vgg16_image(img_path):
    x = image.imread(img_path)
    x = image.imresize(x, 224, 224)
    return x


class Vgg16FeatureExtractor(object):

    def __init__(self, model_ctx=mx.cpu()):
        self.model_ctx = model_ctx
        self.image_net = models.vgg16(pretrained=True)
        self.image_net.collect_params().reset_ctx(ctx=model_ctx)

    def extract_image_features(self, image_path):
        img = load_vgg16_image(image_path)
        img = transform(img, target_ht=224, target_wd=224).expand_dims(axis=0)
        img = img.as_in_context(self.model_ctx)
        return self.image_net(img)
