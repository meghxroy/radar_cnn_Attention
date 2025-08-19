import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.layers import MultiHeadAttention
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Activation
from tensorflow.keras import layers, models
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
"""
data_visualization
1.seed.np.keras&tensor
2.squeeze and execution layer
"""
class TransformerBlock(layers.Layer):
  def__init__(self,embed_dim,num_heads,ff_dim_rate=0.2):
    super().__init__()
    self.att=layers.MultiHeadAttention(num_heads,key_dim = embed_dim)
    self.ffn=models.Sequential([
      layers.Dense(ff_dim,activation='relu'),
      layers.Dense(embed_dim),
    ])
    self.norm1=layers.LayerNormalization(epsilon=1e-6)
    self.norm2=layers.LayerNormalization(epsilon=1e-6)
    self.dropout1=layers.Dropout(rate)
    self.dropout2=layers.Dropout(rate)
def call(self,inputs,training=False):
  attn_output=self.att(inputs,inputs)
  attn_output=self.dropout1(attn_output,training=training)
  out1=self.norm(inputs + attn_output)
  ffn_out=self.ffn(out1)
  ffn_out=self.dropout2(ffn_out,training=training)
  return self.norm2(out1 + ffn_out)
def build_model(input_shape):
  inputs=tf.keras.Input(shape=(input_shape))
  x=layers.Conv1D(
      

