# streamlit_final_original.py
"""
Original script adapted for Streamlit:
- Upload your CSV (no Iris fallback, must provide file)
- Uses all features + label (label = 'Type' if exists, else last column)
- Runs SLP (no hidden layer, logistic activation)
- Runs MLP (1 hidden layer with user-chosen neurons, relu activation)
- Shows predictions & accuracy for both directly in web UI
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

st.title("SLP vs MLP — Original Style (with file + hidden layer option)")

# === UI controls ===
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
hidden_neurons = st.slider("Hidden layer size (MLP)", min_value=1, max_value=50, value=10, step=1)
run = st.button("Run")

if not uploaded_file:
    st.stop()

# === Load data ===
df = pd.read_csv(uploaded_file)

# Label = 'Type' if exists, else last column
label_col = "Type" if "Type" in df.columns else df.columns[-1]
feature_cols = [c for c in df.columns if c != label_col]

X = df[feature_cols]
y_raw = df[label_col]

# Encode labels if non-numeric
if y_raw.dtype == object or not np.issubdtype(y_raw.dtype, np.number):
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
else:
    y = y_raw

# === Train & compare when button pressed ===
if run:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # --- SLP (no hidden layer) ---
    slp = MLPClassifier(hidden_layer_sizes=(), activation='logistic',
                        max_iter=1000, random_state=42)
    slp.fit(X_train, y_train)
    y_pred_slp = slp.predict(X_test)
    acc_slp = accuracy_score(y_test, y_pred_slp)

    # --- MLP (1 hidden layer) ---
    mlp = MLPClassifier(hidden_layer_sizes=(hidden_neurons,), activation='relu',
                        max_iter=1000, random_state=42)
    mlp.fit(X_train, y_train)
    y_pred_mlp = mlp.predict(X_test)
    acc_mlp = accuracy_score(y_test, y_pred_mlp)

    # === Show results in web UI ===
    st.subheader("Predictions")
    st.write("SLP (no hidden layer):")
    st.code(np.array2string(y_pred_slp, separator=', '), language='text')
    st.write("MLP (hidden layer = %d neurons):" % hidden_neurons)
    st.code(np.array2string(y_pred_mlp, separator=', '), language='text')

    st.subheader("Accuracy")
    st.write(f"SLP: **{acc_slp:.4f}**")
    st.write(f"MLP ({hidden_neurons} neurons): **{acc_mlp:.4f}**")

    st.subheader("Classification Reports")
    st.write("**SLP**")
    st.text(classification_report(y_test, y_pred_slp, zero_division=0))
    st.write("**MLP**")
    st.text(classification_report(y_test, y_pred_mlp, zero_division=0))
