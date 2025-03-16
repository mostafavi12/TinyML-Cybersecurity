import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

def load_and_preprocess_data():
    print("[*] Loading TON_IoT dataset...")
    df = pd.read_csv("./data/TON_IoT/Train_Test_datasets/Train_Test_Network_dataset/train_test_network.csv")

    # Define features
    features = ['duration', 'src_bytes', 'dst_bytes', 'src_pkts', 'dst_pkts', 'proto',
                'conn_state', 'service', 'missed_bytes', 'http_status_code', 'ssl_established']

    # Convert categorical features to numeric codes
    for col in ['proto', 'conn_state', 'service', 'ssl_established']:
        df[col] = df[col].astype('category').cat.codes

    # Encode attack types (target variable)
    le = LabelEncoder()
    df['type_encoded'] = le.fit_transform(df['type'])

    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])
    y = df['type_encoded']

    # Save preprocessing tools
    joblib.dump(scaler, "./models/scaler.pkl")
    joblib.dump(le, "./models/label_encoder.pkl")

    return X, y, features
