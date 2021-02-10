#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import os


# In[7]:


# ORIGINAL DIRECTORY --> 
GOOGLE_DRIVE = True
RESTART_MODEL = False
filename = "data.tfrecords"
if (GOOGLE_DRIVE == True):
    ORIG_INPUT_DATASET = "/content/gdrive/My Drive/datasets/cfd_hennigh"
# BASE DIRECTORY --> were modified data will be saved
    BASE_INPUT_DATASET = "/content/gdrive/My Drive/datasets/cfd_hennigh"    
else:
    ORIG_INPUT_DATASET = "/home/imagda/_projects_2020/datasets/cfd"
# BASE DIRECTORY --> were modified data will be saved
    BASE_INPUT_DATASET = "/home/imagda/_projects_2020/datasets/cfd"
    
TRAIN_PATH = os.path.sep.join([BASE_INPUT_DATASET, "train"])
VAL_PATH = os.path.sep.join([BASE_INPUT_DATASET , "val"])
TEST_PATH = os.path.sep.join([BASE_INPUT_DATASET , "test"])

print(TRAIN_PATH, VAL_PATH, TEST_PATH)
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1

BATCH_SIZE = 12

CLR_METHOD = "exp_range" #triangular # triangular2
MIN_LR = 1E-3
MAX_LR = 1E-1
STEP_SIZE = 8

# In[8]:


#get_ipython().system('jupyter nbconvert --to script config.ipynb')


# In[ ]:




