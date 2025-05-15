# Updated preprocessing.py to compute and compare SHAP values across all features and top N features
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
import joblib
import json
import shap
import matplotlib.pyplot as plt
import os

def load_and_preprocess_data(file_path):
    logging.info("")
    logging.info(f"[*] Loading dataset from: {file_path}")

    # Load dataset
    df = pd.read_csv(file_path)

    # Replace '-' with NaN and fill with -1 across the whole dataset
    df.replace('-', np.nan, inplace=True)
    df.fillna(-1, inplace=True)

    # Convert 'T'/'F' values to 1/0
    for col in df.columns:
        if df[col].isin(['T', 'F']).any():
            df[col] = df[col].replace({'T': 1, 'F': 0})

    # Convert http_trans_depth to numeric if it exists
    if 'http_trans_depth' in df.columns:
        df['http_trans_depth'] = pd.to_numeric(df['http_trans_depth'], errors='coerce').fillna(-1)

    # Encode ssl_version and ssl_cipher as categories
    for col in ['ssl_version', 'ssl_cipher']:
        if col in df.columns:
            df[col] = df[col].astype('category').cat.codes

    # Original plus engineered features
    df['src_port_category'] = df['src_port'].apply(lambda x: 'well_known' if x < 1024 else 'ephemeral')
    df['is_private_ip'] = df['src_ip'].apply(lambda ip: 1 if str(ip).startswith("192.") or str(ip).startswith("10.") else 0)
    df['byte_rate'] = (df['src_bytes'] + df['dst_bytes']) / df['duration'].replace(0, 1e-6)
    df['pkt_rate'] = (df['src_pkts'] + df['dst_pkts']) / df['duration'].replace(0, 1e-6)

    # Encode categorical features
    categorical_cols = ['proto', 'conn_state', 'service', 'src_port_category', 'http_method', 'http_version']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category').cat.codes

    # Drop high-cardinality or unstructured text columns
    drop_cols = [
        'label', 'type', 'type_encoded',
        'src_ip', 'dst_ip', 'dns_query', 'http_uri',
        'http_user_agent', 'ssl_subject', 'ssl_issuer',
        'weird_name', 'weird_addl',
        'http_orig_mime_types', 'http_resp_mime_types'
    ]

    # Encode target
    le = LabelEncoder()
    df['type_encoded'] = le.fit_transform(df['type'])

    # Save encoder
    joblib.dump(le, './models/label_encoder.pkl')

    # Boruta feature selection
    available_features = df.drop(columns=[col for col in drop_cols if col in df.columns])
    feature_cols = available_features.columns.tolist()
    if 'type_encoded' in feature_cols:
        feature_cols.remove('type_encoded')

    # Validate all features are numeric before Boruta
    non_numeric_cols = []
    for col in feature_cols:
        if not np.issubdtype(df[col].dtype, np.number):
            non_numeric_cols.append(col)

    if non_numeric_cols:
        logging.error("[!] Non-numeric columns found before Boruta: %s", non_numeric_cols)
        for col in non_numeric_cols:
            unique_vals = df[col].unique()[:5]
            logging.error("    -> Column '%s' sample values: %s", col, unique_vals)
        raise ValueError("Non-numeric columns detected — fix encoding before Boruta.")

    X_full = available_features[feature_cols].values
    y_full = df['type_encoded'].values

    rf_for_boruta = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5, random_state=42)
    boruta_selector = BorutaPy(estimator=rf_for_boruta, n_estimators='auto', verbose=2, random_state=42)
    boruta_selector.fit(X_full, y_full)

    selected_features = [feature_cols[i] for i, keep in enumerate(boruta_selector.support_) if keep]
    logging.info(f"[*] Boruta Selected Features: {selected_features}")

    # Save selected features
    with open("./models/selected_features.json", "w") as f:
        json.dump(selected_features, f, indent=4)

    # Final dataset
    X_selected = available_features[selected_features]
    scaler = StandardScaler()
    X = scaler.fit_transform(X_selected)
    y = df['type_encoded']

    # Save scaler
    joblib.dump(scaler, './models/scaler.pkl')

    logging.info("[*] Final Input Shape: %s", X.shape)
    logging.info("[*] Final Labels Shape: %s", y.shape)

    # === SHAP Feature Importance on All Features ===
    try:
        logging.info("[*] Computing SHAP values for all features (pre-selection)...")
        os.makedirs("./visualizations", exist_ok=True)
        rf_shap = RandomForestClassifier(n_estimators=20, max_depth=5, random_state=42)
        rf_shap.fit(X_full, y_full)
        explainer = shap.TreeExplainer(rf_shap)
        shap_values = explainer.shap_values(X_full)

        # Summary plot for all features
        plt.figure()
        shap.summary_plot(shap_values, features=X_full, feature_names=feature_cols, show=False)
        plt.tight_layout()
        plt.savefig("./visualizations/shap_summary_all_features.png")
        logging.info("[*] SHAP summary plot saved: shap_summary_all_features.png")

        # Compute mean absolute SHAP values per feature
        mean_shap = np.abs(shap_values).mean(axis=1).mean(axis=0)
        shap_importance = pd.DataFrame({"feature": feature_cols, "importance": mean_shap})
        shap_importance = shap_importance.sort_values(by="importance", ascending=False)

        # Save top N SHAP features plot
        top_n = 10
        top_features = shap_importance.head(top_n)['feature'].tolist()
        top_indices = [feature_cols.index(f) for f in top_features]
        plt.figure()
        shap.summary_plot(np.array(shap_values)[:, :, top_indices], features=X_full[:, top_indices], feature_names=top_features, show=False)
        plt.tight_layout()
        plt.savefig("./visualizations/shap_summary_top_features.png")
        logging.info("[*] SHAP summary for top %d features saved: shap_summary_top_features.png", top_n)

    except Exception as e:
        logging.warning(f"[!] SHAP computation failed: {e}")

    return X, y, selected_features, le.classes_
