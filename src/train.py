import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.layers import MultiHeadAttention
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Activation
from tensorflow.keras import layers,models
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from model import build_model
import os
import joblib
"""
set virtual environment
backend='tensorlfow'
set random state=42
shuffle=True
"""

