import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

st.title("Single-Layer Perceptron(SLP) Vs Multi-Layer Perceptron (MLP)")

# === UI controls ===
uploaded_file = st.file_uploader("Upload CSV file(Optional)", type=["csv"])
hidden_neurons = st.slider("Hidden layer size (MLP)", min_value=1, max_value=50, value=10, step=1)
st.text("The higher the value, the more neurons in the hidden layer. means it thinks more but not always better because the complexity of the data is too low, so the result vary a lot.")
run = st.button("Run")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("glass.csv")

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
    )# --- 0.2(20%) 42 columns of data ---

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
    st.text("The Prediction of the 80% of the train data and compared to 20% of the test data(42 Rows)")
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
