#Train the Model Using 70% of the Dataset


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress cuDNN/cuBLAS warnings


import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense


print("[*] Loading TON_IoT dataset...")
pathToDataset = r'./data/TON_IoT/Train_Test_datasets/Train_Test_Network_dataset/train_test_network.csv'
#pathToDataset = r'/home/ahmad/projects/TinyML-Cybersec/data/TON_IoT/Train_Test_datasets/Train_Test_Network_dataset/train_test_network.csv'
#pathToDataset = r'~/projects/TinyML-Cybersecurity/data/TON_IoT/Train_Test_datasets/Train_Test_Network_dataset/train_test_network.csv'
print(pathToDataset)
df = pd.read_csv(pathToDataset)


print("Columns in dataset:", df.columns)


# First version!
"""
# Define the correct feature columns for training
features = ['duration', 'src_bytes', 'dst_bytes', 'src_pkts', 'dst_pkts', 'proto']

# Use 'label' or 'type' instead of 'type_of_attack'
df = df[features + ['label']]  # Assuming 'label' is the attack type column

# Convert categorical columns to numeric values (if necessary)
df['proto'] = df['proto'].astype('category').cat.codes  # Convert categorical features

print("Updated dataset columns:", df.columns)

X = df[features]
y = df['label']
"""


# Imporoved version
# Convert 'type' (multiclass labels) into numbers
le = LabelEncoder()
df['type_encoded'] = le.fit_transform(df['type'])  # Map attack types to numbers

# Update feature selection to use 'type_encoded' instead of 'label'
features = ['duration', 'src_bytes', 'dst_bytes', 'src_pkts', 'dst_pkts', 'proto']
X = df[features]
y = df['type_encoded']  # Multiclass classification instead of binary


# Improved version 2 (more features)
"""
le = LabelEncoder()
df['type_encoded'] = le.fit_transform(df['type'])  # Map attack types to numbers

features = ['duration', 'src_bytes', 'dst_bytes', 'src_pkts', 'dst_pkts', 'proto',
            'conn_state', 'service', 'missed_bytes', 'http_status_code', 'ssl_established']

X = df[features]
y = df['type_encoded']  # Multiclass classification instead of binary
"""

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print("[*] Training TinyML model...")
"""
UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. 
When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
Keras model is using input_shape inside the first Dense layer, which is not recommended in Sequential models.

model = tf.keras.Sequential([
    tf.keras.layers.Dense(12, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
"""

model = Sequential([
    Input(shape=(6,)),  # Define input shape explicitly
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=16)

print("[*] Converting model to TensorFlow Lite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("./models/model_ton_iot.tflite", "wb") as f:
    f.write(tflite_model)

print("TinyML model training and conversion completed!")
