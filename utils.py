from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers.legacy import SGD
from tensorflow.keras.models import Sequential
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
import numpy as np
import warnings
import logging
import random
import pickle
import json
import nltk
import os