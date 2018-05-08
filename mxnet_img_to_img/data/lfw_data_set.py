from __future__ import print_function
import os
import tarfile
from mxnet.gluon import utils


def download_lfw_dataset_if_not_exists(data_dir_path='/tmp/lfw_dataset'):
    lfw_url = 'http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz'
    if not os.path.exists(data_dir_path):
        os.makedirs(data_dir_path)
        data_file = utils.download(lfw_url)
        with tarfile.open(data_file) as tar:
            tar.extractall(path=data_dir_path)



