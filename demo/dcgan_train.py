import os
import sys
import mxnet as mx
import logging


LOAD_EXISTING_MODEL = False


def patch_path(path):
    return os.path.join(os.path.dirname(__file__), path)


def main():
    sys.path.append(patch_path('..'))

    output_dir_path = patch_path('models')

    logging.basicConfig(level=logging.DEBUG)

    from mxnet_img_to_img.library.dcgan import DCGan
    from mxnet_img_to_img.data.facades_data_set import load_image_pairs

    ctx = mx.cpu()
    img_pairs = load_image_pairs(patch_path('data/facades'))
    gan = DCGan(model_ctx=ctx)
    gan.random_input_size = 24

    if LOAD_EXISTING_MODEL:
        gan.load_model(output_dir_path)

    gan.fit(image_pairs=img_pairs, model_dir_path=output_dir_path, epochs=100)


if __name__ == '__main__':
    main()
