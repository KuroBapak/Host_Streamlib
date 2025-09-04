# app_simple_fixed_glass.py
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

st.title("NN, SVM, NV, RFT, KNN, DNN Runner")

method = st.selectbox("Choose method", [
    "NEURAL NETWORK - MLP(NN)",
    "Support Vector Machine(SVM)",
    "NAIVE BAYES(NV)",
    "Random Forest Technique(RFT)",
    "K-NEAREST NEIGHBOUR(KNN)",
    "DEEP NEURAL NETWORKS(DNN)"
])
run = st.button("Run")


def prepare_from_csv(df):
    # assume last column is label
    X = df.iloc[:, :-1].copy()
    y = df.iloc[:, -1].copy()
    # encode categorical features if any
    for col in X.select_dtypes(include=['object', 'category']).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    # encode target to 0..n-1 integers
    if y.dtype == 'object' or y.dtype.name == 'category' or y.dtype == 'string':
        y = LabelEncoder().fit_transform(y.astype(str))
    else:
        unique = np.unique(y)
        if not np.array_equal(unique, np.arange(len(unique))):
            y = LabelEncoder().fit_transform(y)
    return X.values, y

if run:
    try:
        df = pd.read_csv("glass.csv")
    except Exception as e:
        st.error(f"Failed to read 'glass.csv' in app folder: {e}")
        st.stop()

    X, y = prepare_from_csv(df)

    # INPUT_DIM = 9 (features: RI, Na, Mg, Al, Si, K, Ca, Ba, Fe)
    # OUTPUT_CLASSES = 6 (Type values: 1,2,3,5,6,7 -> encoded to 6 classes)
    INPUT_DIM = 9
    OUTPUT_CLASSES = 6

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2021)

    # ------------------- NN (MLP) -------------------
    if method == "NEURAL NETWORK - MLP(NN)":
        st.text("NEURAL NETWORK - MLP")
        model = Sequential([
            Input(shape=(9,)),
            Dense(60, activation='relu'),
            Dense(60, activation='relu'),
            Dense(6, activation='softmax')
        ])
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=1)
        loss, acc = model.evaluate(X_test, y_test)
        st.text(f"Test Loss: {loss:.6f}")
        st.text(f"Test Accuracy: {acc:.6f}")
        y_probs = model.predict(X_test)
        roc = roc_auc_score(y_test, y_probs, multi_class='ovr', average='macro')
        st.text(f"ROC AUC Score: {roc:.6f}")


        # ------------------- SVM -------------------
    elif method == "Support Vector Machine(SVM)":
        st.text("Support Vector Machine(SVM)")

        model_linear = LinearSVC(C=0.1 ,random_state=2021, max_iter=5000, loss='squared_hinge')
        model_linear2 = SVC(C=1 ,kernel='linear', probability=True, random_state=2021)
        model_non_linear = SVC(C=1 ,kernel='poly', probability=True, random_state=2021)

        # fit models
        model_linear.fit(X_train, y_train)
        model_linear2.fit(X_train, y_train)
        model_non_linear.fit(X_train, y_train)

        # make predictions from the model_linear and model_non_linear
        Y_pred_linear = model_linear.predict(X_test)
        Y_pred_nonlinear = model_non_linear.predict(X_test)

        # confusion matrix
        conf_matrix_linear = confusion_matrix(y_test, Y_pred_linear)
        print("Confusion Matrix for Linear Model: \n", conf_matrix_linear)
        st.text("Confusion Matrix for Linear Model: \n" + str(conf_matrix_linear))
        print()
        st.text("")

        conf_matrix_nonlinear = confusion_matrix(y_test, Y_pred_nonlinear)
        print("Confusion Matrix for non-Linear Model: \n", conf_matrix_nonlinear)
        st.text("Confusion Matrix for non-Linear Model: \n" + str(conf_matrix_nonlinear))
        print()
        st.text("")

        # accuracy
        accuracy_linear = model_linear.score(X_test, y_test)
        print("Accuracy for Linear Model: %.2f \n" % accuracy_linear)
        st.text("Accuracy for Linear Model: %.2f " % accuracy_linear)

        accuracy_nonlinear = model_non_linear.score(X_test, y_test)
        print("Accuracy for non-Linear Model: %.2f \n" % accuracy_nonlinear)
        st.text("Accuracy for non-Linear Model: %.2f " % accuracy_nonlinear)

        # classification report
        print("Classification Report For Linear Model: \n", classification_report(y_test, Y_pred_linear))
        st.text("Classification Report For Linear Model: \n" + classification_report(y_test, Y_pred_linear))
        print("Classification Report For non-Linear Model: \n", classification_report(y_test, Y_pred_nonlinear))
        st.text("Classification Report For non-Linear Model: \n" + classification_report(y_test, Y_pred_nonlinear))

        linear = roc_auc_score(y_test, model_non_linear.predict_proba(X_test), multi_class='ovr')
        non_linear = roc_auc_score(y_test, model_linear2.predict_proba(X_test), multi_class='ovr')
        st.text(f"ROC AUC for Linear Model : {linear:.6f}")
        st.text(f"ROC AUC for non-Linear Model: {non_linear:.6f}")
        print("ROC AUC for Linear Model :", linear)
        print("ROC AUC for non-Linear Model:", non_linear)


    # ------------------- Naive Bayes -------------------
    elif method == "NAIVE BAYES(NV)":
        st.text("NAIVE BAYES")
        nb = GaussianNB()
        nb.fit(X_train, y_train)
        y_pred = nb.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        st.text(f"Accuracy: {acc:.6f}")
        st.text(f"Confusion Matrix:\n {conf_matrix}")
        st.text(f"Classification Report:\n {class_report}")

    # ------------------- Random Forest -------------------
    elif method == "Random Forest Technique(RFT)":
        st.text("Random Forest Technique")
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.text(f"Accuracy: {acc:.2f}")

    # ------------------- KNN -------------------
    elif method == "K-NEAREST NEIGHBOUR(KNN)":
        st.text("K-NEAREST NEIGHBOUR")
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.text(f"Accuracy: {acc * 100:.2f}%")

    # ------------------- DNN -------------------
    elif method == "DEEP NEURAL NETWORKS(DNN)":
        st.text("DEEP NEURAL NETWORKS")
        # y is already label-encoded elsewhere in the app (0..5).
        # create one-hot targets (like your original snippet)
        y_cat = to_categorical(y, num_classes=6)

        # Train-test split using stratify on the integer labels (y)
        # Match your original example's split params: test_size=0.2, random_state=42
        X_train, X_test, y_train_cat, y_test_cat = train_test_split(
            X, y_cat, test_size=0.2, random_state=42, stratify=y
        )

        # Normalize features (fit scaler on training set only)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Build Deep Neural Network (use input_shape=(X_train.shape[1],) as original)
        model = Sequential([
            Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
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

            Dense(6, activation='softmax')  # 6 classes for glass.csv
        ])

        # Compile model (same as your original)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Callbacks (same as original)
        early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10)

        # Train (use validation_split like original)
        history = model.fit(
            X_train, y_train_cat,
            validation_split=0.2,
            epochs=200,
            batch_size=16,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )

        # Final evaluation (same pattern)
        loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
        st.text(f"Test Accuracy: {acc:.4f}")


    else:
        st.error("Unknown method.")
