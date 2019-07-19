import os
from glob import glob
from os.path import join

os.environ['HDF5_USE_FILE_LOCKING'] = 'false'

import GPUtil
import numpy as np
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from natsort import natsorted
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

from Datagen import PngClassDataGenerator, PngDataGenerator
from Losses import dice_coef_loss
from Models import BlockModel2D, BlockModel_Classifier, ConvertEncoderToCED
from Parameters import get_train_params,get_val_params

rng = np.random.RandomState(seed=1)

try:
    if not 'DEVICE_ID' in locals():
        DEVICE_ID = GPUtil.getFirstAvailable()[0]
        print('Using GPU', DEVICE_ID)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)
except Exception as e:
    raise('No GPU available')


# ~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~ SETUP~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~

# Setup data
pre_train_datapath = '/data/Kaggle/nih-chest-dataset/images_resampled_sorted_into_categories/Pneumothorax/'
pre_train_negative_datapath = '/data/Kaggle/nih-chest-dataset/images_resampled_sorted_into_categories/No_Finding/'

pos_img_path = '/data/Kaggle/pos-norm-png'
pos_mask_path = '/data/Kaggle/pos-mask-png'

train_datapath = '/data/Kaggle/train-png'
train_mask_path = '/data/Kaggle/train-mask'

pretrain_weights_filepath = 'Pretrain_class_weights.h5'
weight_filepath = 'Kaggle_Weight_wpretrain_{epoch:02d}-{val_loss:.4f}.h5'

# pre-train parameters
pre_im_dims = (512, 512)
pre_n_channels = 1
pre_batch_size = 8
pre_val_split = .15
pre_epochs = 2
pre_multi_process = False

# train parameters
im_dims = (512, 512)
n_channels = 1
batch_size = 4
learnRate = 1e-4
val_split = .2
epochs = [2, 2]  # epochs before and after unfreezing weights
multi_process = True

# model parameters
filt_nums = 16
num_blocks = 4

# datagen params
pre_train_params = get_train_params(pre_batch_size,pre_im_dims,pre_n_channels)
pre_val_params = get_val_params(pre_batch_size,pre_im_dims,pre_n_channels)
train_params = get_train_params(batch_size,im_dims,n_channels)
val_params = get_val_params(batch_size,im_dims,n_channels)
full_train_params = get_train_params(2,(1024,1024),1)
full_val_params = get_val_params(2,(1024,1024),1)

# ~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~Pre-training~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~

print('---------------------------------')
print('---- Setting up pre-training ----')
print('---------------------------------')

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

# get class weights for  balancing
class_weights = class_weight.compute_class_weight(
    'balanced', np.unique(pretrain_labels), pretrain_labels)
class_weight_dict = dict(enumerate(class_weights))

# Split into test/validation sets
pre_trainX, pre_valX, pre_trainY, pre_valY = train_test_split(
    pretrain_img_files, pretrain_labels, test_size=val_split, random_state=rng, shuffle=True)

pre_train_dict = dict([(f, mf) for f, mf in zip(pre_trainX, pre_trainY)])
pre_val_dict = dict([(f, mf) for f, mf in zip(pre_valX, pre_valY)])

# Setup datagens
pre_train_gen = PngClassDataGenerator(pre_trainX,
                                      pre_train_dict,
                                      **pre_train_params)
pre_val_gen = PngClassDataGenerator(pre_valX,
                                    pre_val_dict,
                                    **pre_val_params)


# Create model
pre_model = BlockModel_Classifier(input_shape=pre_im_dims+(pre_n_channels,),
                                  filt_num=filt_nums, numBlocks=num_blocks)

# Compile model
pre_model.compile(Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Create callbacks
cb_check = ModelCheckpoint(pretrain_weights_filepath, monitor='val_loss',
                           verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
cb_plateau = ReduceLROnPlateau(
    monitor='val_loss', factor=.5, patience=3, verbose=1)

print('---------------------------------')
print('----- Starting pre-training -----')
print('---------------------------------')

# Train model
pre_history = pre_model.fit_generator(generator=pre_train_gen,
                                      epochs=pre_epochs, use_multiprocessing=pre_multi_process,
                                      workers=8, verbose=1, callbacks=[cb_check, cb_plateau],
                                      class_weight=class_weight_dict,
                                      validation_data=pre_val_gen)

# Load best weights
pre_model.load_weights(pretrain_weights_filepath)

# Calculate confusion matrix
print('Calculating classification confusion matrix...')
pre_val_gen.shuffle = False
preds = pre_model.predict_generator(pre_val_gen,verbose=1)
labels = [pre_val_gen.labels[f] for f in pre_val_gen.list_IDs]
y_pred = np.rint(preds)
totalNum = len(y_pred)
y_true = np.rint(labels)[:totalNum]
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

print('----------------------')
print('Classification Results')
print('----------------------')
print('True positives: {}'.format(tp))
print('True negatives: {}'.format(tn))
print('False positives: {}'.format(fp))
print('False negatives: {}'.format(fn))
print('% Positive: {:.02f}'.format(100*(tp+fp)/totalNum))
print('% Negative: {:.02f}'.format(100*(tn+fn)/totalNum))
print('% Accuracy: {:.02f}'.format(100*(tp+tn)/totalNum))
print('-----------------------')

# ~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~ Training ~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~

print('Setting up 512-training')

# convert to segmentation model
model = ConvertEncoderToCED(pre_model)

# create segmentation datagens
# using positive images only
train_img_files = natsorted(glob(join(pos_img_path, '*.png')))
train_mask_files = natsorted(glob(join(pos_mask_path, '*.png')))


# Split into test/validation sets
trainX, valX, trainY, valY = train_test_split(
    train_img_files, train_mask_files, test_size=val_split, random_state=rng, shuffle=True)

train_dict = dict([(f, mf) for f, mf in zip(trainX, trainY)])
val_dict = dict([(f, mf) for f, mf in zip(valX, valY)])

# Setup datagens
train_gen = PngDataGenerator(trainX,
                             train_dict,
                             **train_params)
val_gen = PngDataGenerator(valX,
                           val_dict,
                           **val_params)


# Create callbacks
cb_check = ModelCheckpoint(weight_filepath, monitor='val_loss',
                           verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
cb_plateau = ReduceLROnPlateau(
    monitor='val_loss', factor=.5, patience=3, verbose=1)

# Compile model
model.compile(Adam(lr=learnRate), loss=dice_coef_loss)

print('---------------------------------')
print('----- Starting 512-training -----')
print('---------------------------------')

history = model.fit_generator(generator=train_gen,
                              epochs=epochs[0], use_multiprocessing=multi_process,
                              workers=8, verbose=1, callbacks=[cb_check, cb_plateau],
                              validation_data=val_gen)

# make all layers trainable again
for layer in model.layers:
    layer.trainable = True

# Compile model
model.compile(Adam(lr=learnRate), loss=dice_coef_loss)

print('----------------------------------')
print('--Training with unfrozen weights--')
print('----------------------------------')

history2 = model.fit_generator(generator=train_gen,
                               epochs=epochs[1], use_multiprocessing=multi_process,
                               workers=8, verbose=1, callbacks=[cb_check, cb_plateau],
                               validation_data=val_gen)

print('Setting up 1024 training')

# make full-size model
full_model = BlockModel2D((1024, 1024, n_channels), filt_num=16, numBlocks=4)
h5files = glob('*.h5')
load_file = max(h5files, key=os.path.getctime)
full_model.load_weights(load_file)

# Compile model
full_model.compile(Adam(lr=learnRate), loss=dice_coef_loss)
# Create callbacks
cb_check = ModelCheckpoint(weight_filepath, monitor='val_loss',
                           verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
cb_plateau = ReduceLROnPlateau(
    monitor='val_loss', factor=.5, patience=3, verbose=1)

# Setup full size datagens
train_gen = PngDataGenerator(trainX,
                             train_dict,
                             **full_train_params)
val_gen = PngDataGenerator(valX,
                           val_dict,
                           **full_val_params)

print('---------------------------------')
print('---- Starting 1024-training -----')
print('---------------------------------')

# train full size model
history_full = full_model.fit_generator(generator=train_gen,
                                        epochs=2, use_multiprocessing=multi_process,
                                        workers=8, verbose=1, callbacks=[cb_check, cb_plateau],
                                        validation_data=val_gen)


# Rename best weights
h5files = glob('*.h5')
load_file = max(h5files, key=os.path.getctime)
os.rename(load_file, 'Best_Kaggle_weights_wpretrainfull.h5')

print('Renamed weights file {} to {}'.format(
    load_file, 'Best_Kaggle_weights_wpretrainfull.h5'))