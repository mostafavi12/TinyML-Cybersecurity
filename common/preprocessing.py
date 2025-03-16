import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

def load_and_preprocess_data(file_path):
    print(f"[*] Loading dataset from: {file_path}")

    # Load dataset
    df = pd.read_csv(file_path)

    # Print dataset info
    print("\n[*] Dataset Columns:", df.columns.tolist())
    print("[*] Sample Rows:\n", df.head())

    # Select important features (update this as needed)
    selected_columns = ['duration', 'src_bytes', 'dst_bytes', 'src_pkts', 'dst_pkts', 'proto', 'label', 'type']
    df = df[selected_columns]

    # Print info after column selection
    print("\n[*] Selected Columns:", df.columns.tolist())
    print("[*] Sample After Selection:\n", df.head())

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

    # Save preprocessing tools for later use
    joblib.dump(scaler, './models/scaler.pkl')
    joblib.dump(le, './models/label_encoder.pkl')

    print("\n[*] Final Input Shape:", X.shape)
    print("[*] Final Labels Shape:", y.shape)

    return X, y, features