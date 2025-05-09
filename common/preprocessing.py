import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

def load_and_preprocess_data(file_path):
    logging.info("")
    logging.info(f"[*] Loading dataset from: {file_path}")

    # Load dataset
    df = pd.read_csv(file_path)

    # Dataset info
    logging.info("[*] Dataset Columns(%d): %s", df.columns.size, df.columns.tolist())
    logging.info("[*] Sample Rows:\n%s", df.head())

    # Select important features
    selected_columns = ['duration', 'src_bytes', 'dst_bytes', 'src_pkts', 'dst_pkts', 'proto', 'label', 'type']
    df = df[selected_columns]

    logging.info("[*] Selected Columns: %s", df.columns.tolist())
    logging.info("[*] Sample After Selection:\n%s", df.head())

    # Handle categorical data
    df['proto'] = df['proto'].astype('category').cat.codes

    # Encode attack types (target variable)
    le = LabelEncoder()
    df['type_encoded'] = le.fit_transform(df['type'])

    # Normalize features
    scaler = StandardScaler()
    features = ['duration', 'src_bytes', 'dst_bytes', 'src_pkts', 'dst_pkts', 'proto']
    X = scaler.fit_transform(df[features])
    y = df['type_encoded']

    # Save tools
    joblib.dump(scaler, './models/scaler.pkl')
    joblib.dump(le, './models/label_encoder.pkl')

    logging.info("[*] Final Input Shape: %s", X.shape)
    logging.info("[*] Final Labels Shape: %s", y.shape)

    return X, y, features, le.classes_