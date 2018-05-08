import os
from matplotlib import pyplot as plt
import sys


def patch_path(path):
    return os.path.join(os.path.dirname(__file__), path)


def main():
    sys.path.append(patch_path('..'))
    data_dir_path = patch_path('data/lfw_dataset')

    from mxnet_img_to_img.library.image_utils import load_images, visualize
    from mxnet_img_to_img.data.lfw_data_set import download_lfw_dataset_if_not_exists

    download_lfw_dataset_if_not_exists(data_dir_path)
    img_list = load_images(data_dir_path)
    for i in range(4):
        plt.subplot(1, 4, i + 1)
        visualize(img_list[i + 10])
    plt.show()


if __name__ == '__main__':
    main()