#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import imutils
import os
import sys
import random
from pysim import config
import glob
import shutil
import tensorflow as tf
from functools import partial
import matplotlib.pyplot as plt


# In[ ]:


print("Original data:", config.ORIG_INPUT_DATASET)
print("INFO: This data is saved as tf.records so no further processing is done on disk.")
print("Folders", config.TRAIN_PATH,config.VAL_PATH,config.TEST_PATH, "will not be created.")
print("INFO: Further data processing is done on the fly.")
filename = os.path.sep.join([config.ORIG_INPUT_DATASET, config.filename])
print("Following file is processed:", config.filename)


# In[ ]:


# Print examples to see how data was encoded
raw_records = tf.data.TFRecordDataset(filename)
for raw_record in raw_records.take(2):
  example = tf.train.Example()
  example.ParseFromString(raw_record.numpy())
  # print(example) 


# In[ ]:


def _tfrecord_parse(example):
    tfrecord_format = {"boundary": tf.io.FixedLenFeature([], tf.string),
                        "sflow": tf.io.FixedLenFeature([], tf.string)}
    example = tf.io.parse_single_example(example, tfrecord_format) 
    
    boundary = tf.io.decode_raw(example["boundary"], tf.uint8)  
    boundary = tf.cast(boundary, tf.float32)
    boundary = tf.reshape(boundary, [128, 256, 1])
    #boundary = tf.image.grayscale_to_rgb(boundary) 

    sflow = tf.io.decode_raw(example["sflow"], tf.float32)
    sflow = tf.cast(sflow, tf.float32)
    sflow = tf.reshape(sflow, [128, 256, 2])
    #sflow = sflow[:, :, 0]
    #sflow = tf.reshape(sflow, [128, 256, 2])
    return boundary, sflow
def load_dataset(filename):
  dataset = tf.data.TFRecordDataset(filename)
  dataset = dataset.map(partial(_tfrecord_parse))
  return dataset

def get_dataset(filename, batch_size):
  dataset = load_dataset(filename)
  dataset = dataset.shuffle(1024)
  dataset = dataset.batch(batch_size)
  return dataset

dataset = get_dataset(filename, config.BATCH_SIZE)
print("INFO: TFRecords file was parsed, and prepared for shuffling/batching!!!")


# In[ ]:


i = 0
for record in dataset:
    i = i + 1
print ("INFO: There are", i * config.BATCH_SIZE, "samples in dataset!!!")    

train_size = int(config.TRAIN_SPLIT * i)
val_size = int(config.VAL_SPLIT * i)
test_size  = i - train_size - val_size
#print(train_size, val_size, test_size)
# Split data in train, val, test on the fly!!!
train_dataset = dataset.take(train_size).repeat(500)
remaining     = dataset.skip(train_size)
val_dataset = dataset.take(val_size).repeat(500)
remanining    = dataset.skip(val_size)
test_dataset  = dataset.take(test_size)

#boundary_batch, sflow_batch = next(iter(train_dataset))



