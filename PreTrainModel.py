import os
from glob import glob
# Setup data
from os.path import join

import GPUtil
import numpy as np
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from natsort import natsorted
from sklearn.model_selection import train_test_split

from Datagen import PngClassDataGenerator, PngDataGenerator
from Losses import dice_coef_loss
from Models import BlockModel2D, BlockModel_Classifier, ConvertEncoderToCED

rng = np.random.RandomState(seed=1)

try:
    if not 'DEVICE_ID' in locals():
        DEVICE_ID = GPUtil.getFirstAvailable()[0]
        print('Using GPU', DEVICE_ID)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)
except Exception as e:
    raise('No GPU available')

pre_train_datapath = '/data/Kaggle/nih-chest-dataset/images_resampled_sorted_into_categories/Pneumothorax/'
pre_train_negative_datapath = '/data/Kaggle/nih-chest-dataset/images_resampled_sorted_into_categories/No_Finding/'

train_datapath = '/data/Kaggle/train-png'
train_mask_path = '/data/Kaggle/train-mask'

pretrain_weights_filepath = 'Pretrain_class_weights.h5'
weight_filepath = 'Kaggle_Weight_wpretrain_{epoch:02d}-{val_loss:.4f}.h5'

# parameters
im_dims = (512, 512)
n_channels = 1
batch_size = 4
val_split = .2
epochs = 5
multi_process = True
train_params = {'batch_size': batch_size,
                'dim': im_dims,
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

val_params = {'batch_size': batch_size,
              'dim': im_dims,
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

# Get list of files
positive_img_files = natsorted(glob(join(pre_train_datapath, '*.png')))
print('Found {} positive files'.format(len(positive_img_files)))
negative_img_files = natsorted(
    glob(join(pre_train_negative_datapath, '*.png')))
print('Found {} negative files'.format(len(negative_img_files)))
# make labels
pos_labels = [1.]*len(positive_img_files)
neg_labels = [0.]*len(negative_img_files)
# combine
pretrain_img_files = positive_img_files + negative_img_files
pretrain_labels = pos_labels + neg_labels

# Split into test/validation sets
pre_trainX, pre_valX, pre_trainY, pre_valY = train_test_split(
    pretrain_img_files, pretrain_labels, test_size=val_split, random_state=rng, shuffle=True)

pre_train_dict = dict([(f, mf) for f, mf in zip(pre_trainX, pre_trainY)])
pre_val_dict = dict([(f, mf) for f, mf in zip(pre_valX, pre_valY)])

# Setup datagens
pre_train_gen = PngClassDataGenerator(pre_trainX,
                                      pre_train_dict,
                                      **train_params)
pre_val_gen = PngClassDataGenerator(pre_valX,
                                    pre_val_dict,
                                    **val_params)


# Create model
pre_model = BlockModel_Classifier(input_shape=im_dims+(n_channels,),
                                  filt_num=16, numBlocks=4)

# Compile model
pre_model.compile(Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Create callbacks
cb_check = ModelCheckpoint(pretrain_weights_filepath, monitor='val_loss',
                           verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
cb_plateau = ReduceLROnPlateau(
    monitor='val_loss', factor=.5, patience=3, verbose=1)


# Train model
pre_history = pre_model.fit_generator(generator=pre_train_gen,
                                      epochs=epochs, use_multiprocessing=multi_process,
                                      workers=8, verbose=1, callbacks=[cb_check, cb_plateau],
                                      validation_data=pre_val_gen)
