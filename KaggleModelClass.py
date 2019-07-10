from glob import glob
from os.path import join

import numpy as np
from natsort import natsorted
from sklearn.model_selection import train_test_split

from Datagen import PngDataGenerator

from Losses import dice_coef_loss
import keras
from Models import BlockModel2D

class PneumothoraxModel:
    def __init__(self, *args, image_path='/data/Kaggle/train-png',
                 mask_path='/data/Kaggle/train-mask',
                 test_path='/data/Kaggle/test-png',
                 dims=(512, 512),
                 batch_size=8,
                 val_split=.2,
                 epochs=1,
                 optimizer='Adam',
                 loss='dice',
                 multi_process=True):
        self.image_path = image_path
        self.mask_path = mask_path
        self.test_path = test_path
        self.dims = dims
        self.n_channels = 1
        self.batch_size = batch_size
        self.val_split = val_split
        self.epochs = epochs
        self.multi_process = multi_process
        self.rng = np.random.RandomState()
        self.optimzer_str = optimizer
        self.loss_str = loss
        self.init_functions()

    def init_functions(self):
        self.init_aug_params()
        self.init_file_list()
        self.init_optimization()
        self.init_model()
        self.validation_split()
        self.setup_datagen()

    def init_aug_params(self):
        self.train_aug_params = {'batch_size': self.batch_size,
                                 'dim': self.dims,
                                 'n_channels': 1,
                                 'shuffle': True,
                                 'rotation_range': 10,
                                 'width_shift_range': 0.1,
                                 'height_shift_range': 0.1,
                                 'brightness_range': None,
                                 'shear_range': 0.,
                                 'zoom_range': 0.1,
                                 'channel_shift_range': 0.,
                                 'fill_mode': 'constant',
                                 'cval': 0.,
                                 'horizontal_flip': True,
                                 'vertical_flip': False,
                                 'rescale': None,
                                 'preprocessing_function': None,
                                 'interpolation_order': 1}
        self.val_aug_params = {'batch_size': batch_size,
                               'dim': dims,
                               'n_channels': 1,
                               'shuffle': True,
                               'rotation_range': 0,
                               'width_shift_range': 0.,
                               'height_shift_range': 0.,
                               'brightness_range': None,
                               'shear_range': 0.,
                               'zoom_range': 0.,
                               'channel_shift_range': 0.,
                               'fill_mode': 'constant',
                               'cval': 0.,
                               'horizontal_flip': False,
                               'vertical_flip': False,
                               'rescale': None,
                               'preprocessing_function': None,
                               'interpolation_order': 1}

    def init_file_list(self):
        # get file list
        self.img_files = natsorted(glob(join(self.image_path, '*.png')))
        self.mask_files = natsorted(glob(join(self.mask_path, '*.png')))
        self.test_files = natsorted(glob(join(self.test_path, '*.png')))
        try:
            assert len(self.img_files) == len(self.mask_files)
        except AssertionError:
            print("WARNING:")
            print('Number of image files and list files do not match')

    def init_optimization(self,lr=1e-4):
        if self.optimzer_str.lower() == 'adam':
            self.optimizer = keras.optimizers.Adam(lr=lr)
        else:
            raise('Unknown optimizer')
        if self.loss_str.lower() == 'dice':
            self.loss = dice_coef_loss
        else:
            raise('Unknown loss function')

    def init_model(self):
        self.model = self.get_model()
        self.model.compile(self.optimizer, loss=self.loss)

    def validation_split(self):
        trainX, valX, trainY, valY = train_test_split(
            self.img_files, self.mask_files, test_size=self.val_split, random_state=self.rng, shuffle=True)
        train_dict = dict([(f, mf) for f, mf in zip(trainX, trainY)])
        val_dict = dict([(f, mf) for f, mf in zip(valX, valY)])
        self.trainX = trainX
        self.valX = valX
        self.trainY = train_dict
        self.valY = valY

    def setup_datagen(self):
        self.train_gen = PngDataGenerator(self.trainX,
                                          self.trainY,
                                          **self.train_aug_params)
        self.val_gen = PngDataGenerator(self.valX,
                                        self.valY,
                                        **self.val_aug_params)

    def get_model(self, filt_num=16, numBlocks=4):
        return BlockModel2D(input_shape=self.dims+(self.n_channels,),
                            filt_num=filt_num, numBlocks=numBlocks)
    def init_callbacks(self,filename='Pneumothorax_model_weights.h5',use_checkpoint=True,use_plateau=True):
        callbacks = []
        if use_checkpoint:
            callbacks.append(ModelCheckpoint(filename, monitor='val_loss',
                           verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1))
        if use_plateau:
            callbacks.append(ReduceLROnPlateau(monitor='val_loss',factor=.5,patience=3,verbose=1))
        self.callbacks = callbacks
    def train(self,epochs=None):
        if epochs is None:
            epochs = self.epochs
        print('Beginning training...')
        self.history = self.model.fit_generator(generator=self.train_gen,
                            epochs=epochs, use_multiprocessing=self.multi_process,
                            workers=8, verbose=1, callbacks=self.callbacks,
                            validation_data=self.val_gen)
