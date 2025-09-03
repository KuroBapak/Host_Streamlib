# app_minimal.py
import streamlit as st
import pandas as pd
import numpy as np

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

# models
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# tensorflow / keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

st.set_page_config(page_title="Minimal ML Runner", layout="centered")
st.title("NN,SVM,NV,RFT,KNN,DNN, Runner")

method = st.selectbox("Choose method", ["NN", "SVM", "NV", "RFT", "KNN", "DNN"])

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

run = st.button("Run")

def prepare_from_csv(df):
    # assume last column is target
    X = df.iloc[:, :-1].copy()
    y = df.iloc[:, -1].copy()

    # encode categorical features automatically
    for col in X.select_dtypes(include=['object', 'category']).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    # encode target
    if y.dtype == 'object' or y.dtype.name == 'category' or y.dtype == 'string':
        y = LabelEncoder().fit_transform(y.astype(str))
    else:
        # ensure labels are 0..n-1
        unique = np.unique(y)
        if not np.array_equal(unique, np.arange(len(unique))):
            y = LabelEncoder().fit_transform(y)

    return X.values, y

def print_metrics(y_true, y_pred, probs=None):
    acc = accuracy_score(y_true, y_pred)
    st.write(f"Test Accuracy: {acc:.6f}")
    st.write("Confusion Matrix:")
    st.text(str(confusion_matrix(y_true, y_pred)))
    st.write("Classification Report:")
    st.text(classification_report(y_true, y_pred))
    if probs is not None:
        try:
            roc = roc_auc_score(y_true, probs, multi_class='ovr', average='macro')
            st.write(f"ROC AUC (macro, ovr): {roc:.6f}")
        except Exception as e:
            st.write(f"ROC AUC not available: {e}")

if run:
        if uploaded_file is None:
            st.error("Please upload a CSV file")
        else:
            try:
                df = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")
                st.stop()

            X, y = prepare_from_csv(df)
            # default split from your examples: test_size=0.3 random_state=2021
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2021, stratify=y if len(np.unique(y))>1 else None)

            # NN (simple MLP)
            if method == "NN":
                model = Sequential([
                    Input(shape=(X_train.shape[1],)),
                    Dense(60, activation='relu'),
                    Dense(60, activation='relu'),
                    Dense(len(np.unique(y)), activation='softmax')
                ])
                model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
                st.write("Training MLP (30 epochs):")
                history = model.fit(X_train, y_train, epochs=30, verbose=1)
                loss, acc = model.evaluate(X_test, y_test, verbose=0)
                st.write(f"Test Loss: {loss:.6f}")
                st.write(f"Test Accuracy: {acc:.6f}")
                y_probs = model.predict(X_test)
                y_pred = np.argmax(y_probs, axis=1)
                print_metrics(y_test, y_pred, probs=y_probs)


            # SVM (LinearSVC + SVC poly as in your snippet)
            elif method == "SVM":
                st.write("Training LinearSVC and SVC (poly)...")
                # LinearSVC (no probs)
                lsvc = LinearSVC(C=0.1, random_state=2021, max_iter=5000, loss='squared_hinge')
                lsvc.fit(X_train, y_train)
                y_pred_l = lsvc.predict(X_test)
                st.write("LinearSVC results:")
                print_metrics(y_test, y_pred_l, probs=None)

                # SVC poly with probability
                svc_poly = SVC(C=1, kernel='poly', probability=True, random_state=2021)
                svc_poly.fit(X_train, y_train)
                y_pred_poly = svc_poly.predict(X_test)
                probs_poly = svc_poly.predict_proba(X_test)
                st.write("SVC (poly) results:")
                print_metrics(y_test, y_pred_poly, probs=probs_poly)

            # Naive Bayes
            elif method == "NV":
                st.write("Training GaussianNB...")
                nb = GaussianNB()
                nb.fit(X_train, y_train)
                y_pred = nb.predict(X_test)
                probs = nb.predict_proba(X_test) if hasattr(nb, "predict_proba") else None
                print_metrics(y_test, y_pred, probs=probs)

            # Random Forest
            elif method == "RFT":
                st.write("Training RandomForest (n_estimators=100)...")
                rf = RandomForestClassifier(n_estimators=100, random_state=42)
                rf.fit(X_train, y_train)
                y_pred = rf.predict(X_test)
                probs = rf.predict_proba(X_test) if hasattr(rf, "predict_proba") else None
                print_metrics(y_test, y_pred, probs=probs)

            # KNN (k=3)
            elif method == "KNN":
                st.write("Training KNN (k=3)...")
                knn = KNeighborsClassifier(n_neighbors=3)
                knn.fit(X_train, y_train)
                y_pred = knn.predict(X_test)
                probs = knn.predict_proba(X_test) if hasattr(knn, "predict_proba") else None
                print_metrics(y_test, y_pred, probs=probs)

            # DNN (deep)
            elif method == "DNN":
                st.write("Training Deep Neural Network (up to 200 epochs, with callbacks)...")
                # scale features (your DNN used StandardScaler)
                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_test_s = scaler.transform(X_test)

                num_classes = len(np.unique(y))
                if num_classes > 2:
                    y_train_cat = to_categorical(y_train, num_classes=num_classes)
                    y_test_cat = to_categorical(y_test, num_classes=num_classes)
                else:
                    y_train_cat = y_train
                    y_test_cat = y_test

                model = Sequential([
                    Dense(256, activation='relu', input_shape=(X_train_s.shape[1],)),
                    BatchNormalization(),
                    Dropout(0.3),
                    Dense(128, activation='relu'),
                    BatchNormalization(),
                    Dropout(0.3),
                    Dense(64, activation='relu'),
                    BatchNormalization(),
                    Dropout(0.2),
                    Dense(32, activation='relu'),
                    BatchNormalization(),
                    Dropout(0.2),
                    Dense(16, activation='relu'),
                    BatchNormalization(),
                    Dense(num_classes, activation='softmax')
                ])
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                              loss='categorical_crossentropy' if num_classes>2 else 'binary_crossentropy',
                              metrics=['accuracy'])
                # callbacks like your code
                early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
                reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10)
                history = model.fit(X_train_s, y_train_cat, validation_data=(X_test_s, y_test_cat),
                                    epochs=200, batch_size=16, callbacks=[early_stop, reduce_lr], verbose=1)
                loss, acc = model.evaluate(X_test_s, y_test_cat, verbose=0)
                st.write(f"Test Loss: {loss:.6f}")
                st.write(f"Test Accuracy: {acc:.6f}")
                probs = model.predict(X_test_s)
                preds = np.argmax(probs, axis=1)
                print_metrics(y_test, preds, probs=probs)

            else:
                st.error("Unknown method.")
