from mxnet import nd
import numpy as np


class ImagePool():

    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []
            self.source_image_feats = []

    def query(self, image_text_pairs):
        if self.pool_size == 0:
            return image_text_pairs
        ret_images = []
        ret_source_image_feats = []
        images, source_image_feats = image_text_pairs

        for i in range(images.shape[0]):
            image = nd.expand_dims(images[i], axis=0)
            source_image_feat = nd.expand_dims(source_image_feats[i], axis=0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                self.source_image_feats.append(source_image_feat)
                ret_images.append(image)
                ret_source_image_feats.append(source_image_feat)
            else:
                p = nd.random_normal(0, 1, shape=(1,)).asscalar()
                if p < 0.5:
                    random_index = nd.random_uniform(0, self.pool_size - 1, shape=(1,)).astype(np.uint8).asscalar()
                    tmp_img = self.images[random_index].copy()
                    tmp_source_image_feat = self.source_image_feats[random_index].copy()
                    self.images[random_index] = image
                    self.source_image_feats[random_index] = source_image_feat
                    ret_images.append(tmp_img)
                    ret_source_image_feats.append(tmp_source_image_feat)
                else:
                    ret_images.append(image)
                    ret_source_image_feats.append(source_image_feat)
        ret_images = nd.concat(*ret_images, dim=0)
        ret_source_image_feats = nd.concat(*ret_source_image_feats, dim=0)
        return [ret_images, ret_source_image_feats]
