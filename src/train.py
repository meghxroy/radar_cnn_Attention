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
input_dir = "GitHub Project CSVs/inputs"
output_dir = "GitHub Project CSVs/outputs"
num_files = 5

# === Helper Function ===
def pad_or_trim_sequences(arr, target_length=315):
    current_length=arr.shape[0]
    if current_length > target_length:
        return arr[:target_length, :]
    elif current_length < target_length:
        pad_width = target_length - current_length
        padding=np.zeros((pad_width, arr.shape[1]))
        return np.vstack([arr, padding])
    else:
        return arr

# === Load All CSVs & Determine Max Columns ===
X_list=[]
Y_list = []
max_columns = 0

for i in range(1, num_files + 1):
    x = pd.read_csv(os.path.join(input_dir, f"input{i}.csv"), header=None).values
    y = pd.read_csv(os.path.join(output_dir, f"output{i}.csv"), header=None).values
   


    if x.ndim == 1:
        x = x[np.newaxis, :]
    if y.ndim == 1:
        y = y[np.newaxis, :]

    if x.shape[0]!= y.shape[0]:
        raise ValueError(f"❌ Row mismatch in input{i}.csv and output{i}.csv: {x.shape[0]} vs {y.shape[0]}")

    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError(f"❌ output{i}.csv must contain only binary values (0 or 1)")

    max_columns = max(max_columns, x.shape[1], y.shape[1])
    X_list.append(x)
    Y_list.append(y)

print(f"📏 Max columns in any file: {max_columns}")

# === Pad/Trim ===
# Ensure arrays are consistent in shape and type
X_prep = [pad_or_trim_sequences(x, 315) for x in X_list]
Y_prep = [pad_or_trim_sequences(y, 315) for y in Y_list]


# === Concatenate and Add Channel Dimension ===
X = np.stack(X_prep, axis=0)
Y = np.stack(Y_prep, axis=0)
 

# === Standardize Inputs ===
X_flat = X.reshape(-1, X.shape[-1])  # Flatten for StandardScaler
scaler = StandardScaler()
X_scaled_flat=scaler.fit_transform(X_flat)
X_scaled=X_scaled_flat.reshape(X.shape)
joblib.dump(scaler,'scaler.pkl')
print("scaler saved as scaler.pkl")
print("X.shape:",X_scaled.shape)
print("Y.shape:",Y.shape)
model = build_model(input_shape=X_scaled.shape[1:])
model.compile(optimizer='adam', 
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), 
    metrics=[tf.keras.metrics.BinaryAccuracy()])
print("Model output shape:", model.output_shape)
early_stop=tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# === Train the Model ===
print("📢 Starting training now...")
history = model.fit(
    X_scaled, Y,
    epochs=50,
    batch_size=256,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
     #type ignore
)
"""
unbiased
"""
print("✅ Training complete.")
model.save("transformer_model.h5")
# === Save Model ===

