import os
import sys
import mxnet as mx


def patch_path(path):
    return os.path.join(os.path.dirname(__file__), path)


def main():
    sys.path.append(patch_path('..'))

    output_dir_path = patch_path('models')

    from mxnet_img_to_img.library.dcgan import DCGan
    from mxnet_img_to_img.data.facades_data_set import load_image_pairs

    img_pairs = load_image_pairs(patch_path('data/facades'))
    gan = DCGan(model_ctx=mx.gpu(0), data_ctx=mx.gpu(0))
    gan.random_input_size = 50
    gan.img_width = 64  # default value is 256, too large for my graphics card memory
    gan.img_height = 64  # default value is 256, too large for my graphics card memory

    gan.fit(image_pairs=img_pairs, model_dir_path=output_dir_path)


if __name__ == '__main__':
    main()
