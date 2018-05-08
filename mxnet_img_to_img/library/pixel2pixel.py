import logging
import mxnet as mx
from mxnet import gluon, autograd, nd
from mxnet.gluon.nn import Conv2D, Conv2DTranspose, LeakyReLU, Activation, BatchNorm, HybridBlock, Dropout, \
    HybridSequential
import numpy as np
import datetime
import time
import os

from mxnet_img_to_img.library.image_utils import load_image, save_image


class UnetSkipUnit(HybridBlock):

    def __init__(self, inner_channels, outer_channels, inner_blocks=None, inner_most=False, outer_most=False,
                 use_dropout=False, use_bias=False):
        super(UnetSkipUnit, self).__init__()

        with self.name_scope():
            self.outer_most = outer_most
            en_conv = Conv2D(channels=inner_channels, in_channels=outer_channels, kernel_size=4, strides=2, padding=1,
                             use_bias=use_bias)
            en_relu = LeakyReLU(0.2)
            en_norm = BatchNorm(momentum=.1, in_channels=inner_channels)
            de_relu = Activation('relu')
            de_norm = BatchNorm(momentum=.1, in_channels=outer_channels)

            if inner_most:
                de_conv = Conv2DTranspose(channels=outer_channels, in_channels=inner_channels, kernel_size=4, strides=2,
                                          padding=1, use_bias=use_bias)
                encoder = [en_relu, en_conv]
                decoder = [de_relu, de_conv]
                model = encoder + decoder
            elif outer_most:
                de_conv = Conv2DTranspose(channels=outer_channels, in_channels=inner_channels * 2, kernel_size=4,
                                          strides=2, padding=1, use_bias=use_bias)
                encoder = [en_conv]
                decoder = [de_relu, de_conv, Activation('tanh')]
                model = encoder + [inner_blocks] + decoder
            else:
                de_conv = Conv2DTranspose(channels=outer_channels, in_channels=inner_channels * 2, kernel_size=4,
                                          strides=2, padding=1, use_bias=use_bias)
                encoder = [en_relu, en_conv, en_norm]
                decoder = [de_relu, de_conv, de_norm]
                model = encoder + [inner_blocks] + decoder

            if use_dropout:
                model += [Dropout(0.5)]

            self.model = HybridSequential()
            with self.model.name_scope():
                for block in model:
                    self.model.add(block)

    def hybrid_forward(self, F, x, *args, **kwargs):
        if self.outer_most:
            return self.model(x)
        else:
            return F.concat(self.model(x), x, dim=1)


class UnetGenerator(HybridBlock):
    def __init__(self, in_channels, num_downs, ngf=64, use_dropout=False):
        super(UnetGenerator, self).__init__()

        with self.name_scope():
            unet = UnetSkipUnit(ngf*8, ngf*8, inner_most=True)
            for _ in range(num_downs - 5):
                unet = UnetSkipUnit(ngf * 8, ngf * 8, unet, use_dropout=use_dropout)
            unet = UnetSkipUnit(ngf * 8, ngf * 4, unet)
            unet = UnetSkipUnit(ngf * 4, ngf * 2, unet)
            unet = UnetSkipUnit(ngf * 2, in_channels, unet, outer_most=True)

            self.model = unet

    def hybrid_forward(self, F, x, *args, **kwargs):
        return self.model(x)


class Discriminator(HybridBlock):

    def __init__(self, in_channels, ndf=64, n_layers=3, use_sigmoid=False, use_bias=False):
        super(Discriminator, self).__init__()

        with self.name_scope():
            self.model = HybridSequential()
            kernel_size = 4
            padding = int((np.ceil(kernel_size - 1) / 2))
            self.model.add(Conv2D(channels=ndf, in_channels=in_channels, kernel_size=kernel_size,
                                  strides=2, padding=padding))
            self.model.add(LeakyReLU(.2))
            nf_mult = 1
            for n in range(1, n_layers):
                nf_mult_prev = nf_mult
                nf_mult = min(2 ** n, 8)
                self.model.add(Conv2D(channels=ndf * nf_mult, in_channels=ndf * nf_mult_prev, kernel_size=kernel_size,
                                      padding=padding, strides=2))
                self.model.add(BatchNorm(momentum=.1, in_channels=ndf * nf_mult))
                self.model.add(LeakyReLU(.2))

            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n_layers, 8)
            self.model.add(Conv2D(channels=ndf * nf_mult, in_channels=ndf * nf_mult_prev, kernel_size=kernel_size,
                                  padding=padding, strides=1))
            self.model.add(BatchNorm(momentum=.1, in_channels=ndf * nf_mult))
            self.model.add(LeakyReLU(.2))
            self.model.add(Conv2D(channels=1, in_channels=ndf * nf_mult, kernel_size=kernel_size, padding=padding,
                                  strides=1))

            if use_sigmoid:
                self.model.add(Activation('sigmoid'))

    def hybrid_forward(self, F, x, *args, **kwargs):
        return self.model(x)


class ImagePool():

    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        ret_images = []
        for i in range(images.shape[0]):
            image = nd.expand_dims(images[i], axis=0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                ret_images.append(image)
            else:
                p = nd.random_normal(0, 1, shape=(1, )).asscalar()
                if p < 0.5:
                    random_index = nd.random_uniform(0, self.pool_size-1, shape=(1, )).astype(np.uint8).asscalar()
                    tmp = self.images[random_index].copy()
                    self.images[random_index] = image
                    ret_images.append(tmp)
                else:
                    ret_images.append(image)
        ret_images = nd.concat(*ret_images, dim=0)
        return ret_images


class Pixel2PixelGan:

    model_name = 'pixel-2-pixel-gan'

    def __init__(self, model_ctx=mx.cpu(), data_ctx=mx.cpu()):
        self.netG = None
        self.netD = None
        self.img_width = 256
        self.img_height = 256
        self.pool_size = 50
        self.num_down_sampling = 8
        self.model_ctx = model_ctx
        self.data_ctx = data_ctx

    @staticmethod
    def param_init(param, ctx):
        if param.name.find('conv') != -1:
            if param.name.find('weight') != -1:
                param.initialize(init=mx.init.Normal(0.02), ctx=ctx)
            else:
                param.initialize(init=mx.init.Zero(), ctx=ctx)
        elif param.name.find('batchnorm') != -1:
            param.initialize(init=mx.init.Zero(), ctx=ctx)
            # Initialize gamma from normal distribution with mean 1 and std 0.02
            if param.name.find('gamma') != -1:
                param.set_data(nd.random_normal(1, 0.02, param.data().shape))

    @staticmethod
    def network_init(net, ctx):
        for param in net.collect_params().values():
            Pixel2PixelGan.param_init(param, ctx)

    @staticmethod
    def facc(label, pred):
        pred = pred.ravel()
        label = label.ravel()
        return ((pred > 0.5) == label).mean()

    @staticmethod
    def get_config_file_path(model_dir_path):
        return os.path.join(model_dir_path, Pixel2PixelGan.model_name + '-config.npy')

    @staticmethod
    def get_params_file_path(model_dir_path, net_name):
        return os.path.join(model_dir_path, Pixel2PixelGan.model_name + '-' + net_name + '.params')

    def load_model(self, model_dir_path):
        config = np.load(self.get_config_file_path(model_dir_path)).item()
        self.img_width = config['image_width']
        self.img_height = config['image_height']
        self.pool_size = config['pool_size']
        self.num_down_sampling = config['num_down_sampling']

        self.netG = UnetGenerator(in_channels=3, num_downs=self.num_down_sampling)
        self.netD = Discriminator(in_channels=6)

        self.netG.load_params(self.get_params_file_path(model_dir_path, 'netG'), ctx=self.model_ctx)
        self.netD.load_params(self.get_params_file_path(model_dir_path, 'netD'), ctx=self.model_ctx)

    def checkpoint(self, model_dir_path):
        self.netG.save_params(self.get_params_file_path(model_dir_path, 'netG'))
        self.netD.save_params(self.get_params_file_path(model_dir_path, 'netD'))

    def fit(self, image_pairs, model_dir_path, lr=0.0002, beta1=0.5, lambda1=100, epochs=100, batch_size=10):

        config = dict()
        config['image_width'] = self.img_width
        config['image_height'] = self.img_height
        config['pool_size'] = self.pool_size
        config['num_down_sampling'] = self.num_down_sampling
        np.save(self.get_config_file_path(model_dir_path), config)

        img_in_list = []
        img_out_list = []
        for source_img_path, target_img_path in image_pairs:
            source_img = load_image(source_img_path, self.img_width, self.img_height)
            target_img = load_image(target_img_path, self.img_width, self.img_height)
            source_img = nd.expand_dims(source_img, axis=0)
            target_img = nd.expand_dims(target_img, axis=0)
            img_in_list.append(source_img)
            img_out_list.append(target_img)
        train_data = mx.io.NDArrayIter(data=[nd.concat(*img_in_list, dim=0), nd.concat(*img_out_list, dim=0)],
                                       batch_size=batch_size)

        ctx = self.model_ctx

        # Pixel2Pixel networks
        self.netG = UnetGenerator(in_channels=3, num_downs=self.num_down_sampling)
        self.netD = Discriminator(in_channels=6)

        # Initialize parameters
        Pixel2PixelGan.network_init(self.netG, ctx)
        Pixel2PixelGan.network_init(self.netD, ctx)

        # trainer for the generator and discrminator
        trainerG = gluon.Trainer(self.netG.collect_params(), 'adam', {'learning_rate': lr, 'beta1': beta1})
        trainerD = gluon.Trainer(self.netD.collect_params(), 'adam', {'learning_rate': lr, 'beta1': beta1})

        GAN_loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
        L1_loss = gluon.loss.L1Loss()

        image_pool = ImagePool(self.pool_size)
        metric = mx.metric.CustomMetric(self.facc)

        logging.basicConfig(level=logging.DEBUG)

        for epoch in range(epochs):
            tic = time.time()
            btic = time.time()
            train_data.reset()
            iter = 0
            fake_out = []

            for batch in train_data:
                ############################
                # (1) Update D network: maximize log(D(x, y)) + log(1 - D(x, G(x, z)))
                ###########################
                real_in = batch.data[0].as_in_context(ctx)
                real_out = batch.data[1].as_in_context(ctx)

                fake_out = self.netG(real_in)
                fake_concat = image_pool.query(nd.concat(real_in, fake_out, dim=1))
                with autograd.record():
                    # Train with fake image
                    output = self.netD(fake_concat)
                    fake_label = nd.zeros(shape=output.shape, ctx=ctx)
                    errD_fake = GAN_loss(output, fake_label)
                    metric.update([fake_label, ], [output, ])

                    # Train with real image
                    output = self.netD(nd.concat(real_in, real_out, dim=1))
                    real_label = nd.ones(shape=output.shape, ctx=ctx)
                    errD_real = GAN_loss(output, real_label)
                    metric.update([real_label, ], [output, ])
                    errD = (errD_real + errD_fake) * 0.5
                    errD.backward()

                trainerD.step(batch.data[0].shape[0])

                ############################
                # (2) Update G network: maximize log(D(x, G(x, z))) - lambda1 * L1(y, G(x, z))
                ###########################
                with autograd.record():
                    fake_out = self.netG(real_in)
                    fake_concat = nd.concat(real_in, fake_out, dim=1)
                    output = self.netD(fake_concat)
                    real_label = nd.ones(shape=output.shape, ctx=ctx)
                    errG = GAN_loss(output, real_label) + L1_loss(real_out, fake_out) * lambda1
                    errG.backward()

                trainerG.step(batch.data[0].shape[0])

                # Print log infomation every ten batches
                if iter % 10 == 0:
                    name, acc = metric.get()
                    logging.info('speed: {} samples/s'.format(batch_size / (time.time() - btic)))
                    logging.info(
                        'discriminator loss = %f, generator loss = %f, binary training acc = %f at iter %d epoch %d'
                        % (nd.mean(errD).asscalar(),
                           nd.mean(errG).asscalar(), acc, iter, epoch))
                iter = iter + 1
                btic = time.time()

            name, acc = metric.get()
            metric.reset()
            logging.info('\nbinary training acc at epoch %d: %s=%f' % (epoch, name, acc))
            logging.info('time: %f' % (time.time() - tic))

            self.checkpoint(model_dir_path)

            # Visualize one generated image for each epoch
            fake_img = fake_out[0]
            fake_img = ((fake_img.asnumpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8)
            save_image(fake_img,
                       os.path.join(model_dir_path, Pixel2PixelGan.model_name + '-training-') + str(epoch) + '.png')

    def generate(self, source_image):
        source_image = nd.expand_dims(source_image, axis=0)
        source_image = source_image.as_in_context(self.model_ctx)
        return self.netG(source_image)[0]