import threading
from itertools import combinations, chain
from json import load, dump
from math import ceil
from os import listdir
from os.path import join, dirname, isfile, abspath, isdir, basename
from random import shuffle

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from utils import read_lines, load_img_arr

DATA_DIR = join(dirname(dirname(__file__)), 'data')
TEST_DIR = join(DATA_DIR, 'test')
TRAIN_DIR = join(DATA_DIR, 'train')
TRAIN_INFO_FILE = join(DATA_DIR, 'train_info_s.csv')
SUBMISSION_INFO_FILE = join(DATA_DIR, 'submission_info.csv')
ORGANIZED_DATA_INFO_FILE = 'organized_data_info_.json'
MODELS_DIR = join(dirname(dirname(__file__)), 'models')
MISC_DIR = join(dirname(dirname(__file__)), 'misc')

dat = pd.read_csv(TRAIN_INFO_FILE)
print(dat.shape)

print(DATA_DIR)
print( join(DATA_DIR, 'test'))
print(TRAIN_INFO_FILE)
print(dirname(__file__))