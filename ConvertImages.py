import csv
import glob
import os
import numpy as np

import cv2
import pydicom as dcm
from natsort import index_natsorted, natsorted, order_by_index

from mask_functions_pneumothorax import rle2mask

train_datapath = '/data/Kaggle/train'
train_outpath = '/data/Kaggle/train-png'
train_mask_path = '/data/Kaggle/train-mask'
if not os.path.exists(train_outpath):
    os.mkdir(train_outpath)
csv_file = 'train-rle.csv'

all_files = [f for f in glob.glob(
    train_datapath + '**/**/*.dcm', recursive=True)]


def splitfile(file):
    _, file = os.path.split(file)
    return os.path.splitext(file)[0]


# Convert all dicoms to png
for cur_file in all_files:
    ds = dcm.dcmread(cur_file)
    cur_name = splitfile(cur_file) + '.png'
    cur_outpath = os.path.join(train_outpath, cur_name)
    cv2.imwrite(cur_outpath, ds.pixel_array)

# Read csv file
with open(csv_file, 'r') as f:
    reader = csv.reader(f)
    data = [row for row in reader]
data = data[1:]

# gather file ids
all_file_ids = [d[0] for d in data]
all_rle = [d[1] for d in data]
unq_file_ids = list(set(all_file_ids))

# Generate masks to PNG
for cur_id in unq_file_ids:
    # find matches for current file id
    matches = [i for i,d in enumerate(all_file_ids) if d == cur_id]
    if all_rle[matches[0]] == '-1':
        # no annotation for this image
        continue
    if len(matches) > 1:
        # combine multiple annotations into one mask
        cur_rle = [all_rle[m] for m in matches]
        masks = [rle2mask(r,1024,1024) for r in cur_rle]
        mask = sum(masks)
    else:
        # convert single annotation to mask
        cur_rle = all_rle[matches[0]]
        mask = rle2mask(cur_rle,1024,1024)
    # write mask tp png
    cur_outpath = os.path.join(train_mask_path,cur_id+'.png')
    cv2.imwrite(cur_outpath,mask)

