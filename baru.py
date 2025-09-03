# app_simple.py
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

st.title("NN, SVM, NV, RFT, KNN, DNN Runner (minimal)")

method = st.selectbox("Choose method", ["NN", "SVM", "NV", "RFT", "KNN", "DNN"])
uploaded_file = st.file_uploader("Upload CSV (required)", type=["csv"])
run = st.button("Run")

def prepare_from_csv(df):
    X = df.iloc[:, :-1].copy()
    y = df.iloc[:, -1].copy()
    for col in X.select_dtypes(include=['object', 'category']).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    if y.dtype == 'object' or y.dtype.name == 'category' or y.dtype == 'string':
        y = LabelEncoder().fit_transform(y.astype(str))
    else:
        unique = np.unique(y)
        if not np.array_equal(unique, np.arange(len(unique))):
            y = LabelEncoder().fit_transform(y)
    return X.values, y

if run:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv("glass.csv")

    X, y = prepare_from_csv(df)
    # same split as your examples
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=2021,
        stratify=y if len(np.unique(y))>1 else None
    )

    # ------------------- NN (MLP) -------------------
    if method == "NN":
        st.text("NEURAL NETWORK - MLP")
        epochs = 30
        model = Sequential([
            Input(shape=(X_train.shape[1],)),
            Dense(60, activation='relu'),
            Dense(60, activation='relu'),
            Dense(len(np.unique(y)), activation='softmax')
        ])
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        # verbose=1 will print epoch progress to server console (terminal)
        model.fit(X_train, y_train, epochs=epochs, verbose=1)
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        st.text(f"Test Loss: {loss:.6f}")
        st.text(f"Test Accuracy: {acc:.6f}")
        # predict and ROC AUC if possible
        probs = model.predict(X_test)
        try:
            roc = roc_auc_score(y_test, probs, multi_class='ovr', average='macro')
            st.text(f"ROC AUC Score: {roc:.6f}")
        except Exception:
            pass

    # ------------------- SVM -------------------
    elif method == "SVM":
        st.text("Support Vector Machine(SVM)")
        # LinearSVC (no predict_proba)
        lsvc = LinearSVC(C=0.1, random_state=2021, max_iter=5000, loss='squared_hinge')
        lsvc.fit(X_train, y_train)
        y_pred_l = lsvc.predict(X_test)
        st.text("Confusion Matrix for Linear Model: ")
        st.text(str(confusion_matrix(y_test, y_pred_l)))
        st.text("")
        st.text("Accuracy for Linear Model: %.2f " % lsvc.score(X_test, y_test))
        st.text("")
        st.text("Classification Report For Linear Model: ")
        st.text(classification_report(y_test, y_pred_l))

        # Non-linear: SVC poly
        svc_poly = SVC(C=1, kernel='poly', probability=True, random_state=2021)
        svc_poly.fit(X_train, y_train)
        y_pred_poly = svc_poly.predict(X_test)
        st.text("")
        st.text("Confusion Matrix for non-Linear Model: ")
        st.text(str(confusion_matrix(y_test, y_pred_poly)))
        st.text("")
        st.text("Accuracy for non-Linear Model: %.2f " % svc_poly.score(X_test, y_test))
        st.text("")
        st.text("Classification Report For non-Linear Model: ")
        st.text(classification_report(y_test, y_pred_poly))

        # ROC AUC (use predict_proba from the SVCs that support it)
        try:
            roc_linear = roc_auc_score(y_test, svc_poly.predict_proba(X_test), multi_class='ovr')
            st.text(f"ROC AUC for non-Linear Model: {roc_linear:.6f}")
        except Exception:
            pass

    # ------------------- Naive Bayes -------------------
    elif method == "NV":
        st.text("NAIVE BAYES")
        nb = GaussianNB()
        nb.fit(X_train, y_train)
        y_pred = nb.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.text(f"Accuracy: {acc:.6f}")
        st.text("Confusion Matrix:")
        st.text(str(confusion_matrix(y_test, y_pred)))
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))

    # ------------------- Random Forest -------------------
    elif method == "RFT":
        st.text("Random Forest Technique")
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.text(f"Accuracy: {acc:.2f}")

    # ------------------- KNN -------------------
    elif method == "KNN":
        st.text("K-NEAREST NEIGHBOUR")
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.text(f"Accuracy: {acc * 100:.2f}%")

    # ------------------- DNN -------------------
    elif method == "DNN":
        st.text("DEEP NEURAL NETWORKS")
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

        early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10)

        # verbose=1 will show epoch progress in server console
        history = model.fit(X_train_s, y_train_cat, validation_data=(X_test_s, y_test_cat),
                            epochs=200, batch_size=16, callbacks=[early_stop, reduce_lr], verbose=1)

        # final evaluation
        loss, acc = model.evaluate(X_test_s, y_test_cat, verbose=0)
        st.text(f"Test Accuracy: {acc:.4f}")

    else:
        st.error("Unknown method.")
