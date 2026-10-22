
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------- PAGE SETTINGS ----------------

st.set_page_config(
    page_title="Diabetes Health Representation",
    page_icon="🩺",
    layout="wide"
)

st.title("Diabetes Health Representation")
st.write(
    "Diabetes prediction using a Stacked Autoencoder "
    "with Mini-Batch Gradient Descent."
)


# ---------------- AUTOMATIC DATASET LOADING ----------------

@st.cache_data(show_spinner="Loading diabetes dataset...")
def load_dataset():
    dataset = fetch_ucirepo(id=891)

    features = dataset.data.features.copy()
    targets = dataset.data.targets.copy()

    df = pd.concat(
        [
            features.reset_index(drop=True),
            targets.reset_index(drop=True)
        ],
        axis=1
    )

    return df


try:
    df = load_dataset()
except Exception as e:
    st.error("Unable to download the dataset.")
    st.write("Check your internet connection and try again.")
    st.code(str(e))
    st.stop()


# ---------------- DATASET INFORMATION ----------------

st.subheader("1. Dataset Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Records", f"{df.shape[0]:,}")

with col2:
    st.metric("Total Columns", df.shape[1])

with col3:
    st.metric("Missing Values", int(df.isnull().sum().sum()))


# ---------------- TARGET SELECTION ----------------

st.subheader("2. Model Configuration")

if "Diabetes_binary" in df.columns:
    default_target = "Diabetes_binary"
elif "Diabetes_012" in df.columns:
    default_target = "Diabetes_012"
else:
    default_target = df.columns[-1]

target = st.selectbox(
    "Select prediction target",
    list(df.columns),
    index=list(df.columns).index(default_target)
)

st.sidebar.header("Model Settings")

latent_dim = st.sidebar.selectbox(
    "Latent Dimension",
    [8, 16, 32],
    index=1
)

epochs = st.sidebar.slider(
    "Training Epochs",
    min_value=1,
    max_value=30,
    value=5
)

batch_size = st.sidebar.selectbox(
    "Mini-Batch Size",
    [64, 128, 256, 512],
    index=1
)

learning_rate = st.sidebar.select_slider(
    "Learning Rate",
    options=[0.0001, 0.0005, 0.001, 0.005],
    value=0.001
)

test_percentage = st.sidebar.slider(
    "Test Set Percentage",
    min_value=10,
    max_value=30,
    value=20
)

train_button = st.button(
    "Train Stacked Autoencoder",
    type="primary",
    use_container_width=True
)


# ---------------- TRAINING ----------------

if train_button:

    st.subheader("3. Data Preparation")

    X = df.drop(columns=[target]).copy()
    y = pd.to_numeric(df[target], errors="coerce")

    X = X.apply(pd.to_numeric, errors="coerce")

    valid_rows = y.notna()
    X = X.loc[valid_rows].reset_index(drop=True)
    y = y.loc[valid_rows].reset_index(drop=True)

    if y.nunique() < 2:
        st.error("The selected target must contain at least two classes.")
        st.stop()

    # Split data before fitting preprocessing
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_percentage / 100,
            random_state=42,
            stratify=y
        )
    except ValueError as e:
        st.error(f"Unable to split the dataset: {e}")
        st.stop()

    imputer = SimpleImputer(strategy="median")

    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = np.nan_to_num(X_train).astype(np.float32)
    X_test = np.nan_to_num(X_test).astype(np.float32)

    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    st.success("Dataset prepared successfully!")

    st.write(f"Training records: {len(X_train):,}")
    st.write(f"Testing records: {len(X_test):,}")

    # ---------------- STACKED AUTOENCODER ----------------

    st.subheader("4. Stacked Autoencoder Training")

    input_dim = X_train.shape[1]
    actual_latent_dim = min(latent_dim, input_dim)

    class StackedAutoencoder(nn.Module):
        def __init__(self, input_size, latent_size):
            super().__init__()

            self.encoder = nn.Sequential(
                nn.Linear(input_size, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, latent_size)
            )

            self.decoder = nn.Sequential(
                nn.Linear(latent_size, 32),
                nn.ReLU(),
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Linear(64, input_size)
            )

        def forward(self, x):
            encoded = self.encoder(x)
            return self.decoder(encoded)

    torch.manual_seed(42)

    model = StackedAutoencoder(
        input_dim,
        actual_latent_dim
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate
    )

    loss_function = nn.MSELoss()

    train_tensor = torch.tensor(
        X_train,
        dtype=torch.float32
    )

    loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=batch_size,
        shuffle=True
    )

    loss_history = []
    progress = st.progress(0)
    status = st.empty()

    for epoch in range(epochs):
        model.train()

        total_loss = 0.0
        total_samples = 0

        for (batch,) in loader:
            optimizer.zero_grad()

            reconstructed = model(batch)
            loss = loss_function(reconstructed, batch)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch)
            total_samples += len(batch)

        average_loss = total_loss / max(total_samples, 1)
        loss_history.append(average_loss)

        progress.progress((epoch + 1) / epochs)
        status.write(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Reconstruction Loss: {average_loss:.5f}"
        )

    st.success("Autoencoder training completed!")

    # ---------------- RECONSTRUCTION LOSS ----------------

    st.subheader("5. Reconstruction Loss")

    fig, ax = plt.subplots()

    ax.plot(
        range(1, epochs + 1),
        loss_history,
        marker="o"
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Squared Error")
    ax.set_title("Training Reconstruction Loss")
    ax.grid(True)

    st.pyplot(fig)
    plt.close(fig)

    # ---------------- ENCODE DATA ----------------

    model.eval()

    with torch.no_grad():
        encoded_train = model.encoder(
            torch.tensor(X_train, dtype=torch.float32)
        ).numpy()

        encoded_test = model.encoder(
            torch.tensor(X_test, dtype=torch.float32)
        ).numpy()

    # ---------------- CLASSIFICATION ----------------

    st.subheader("6. Classification Results")

    baseline = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )

    encoded_classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )

    with st.spinner("Training classification models..."):
        baseline.fit(X_train, y_train)
        encoded_classifier.fit(encoded_train, y_train)

        baseline_pred = baseline.predict(X_test)
        encoded_pred = encoded_classifier.predict(encoded_test)

    def get_metrics(actual, predicted):
        return {
            "Accuracy": accuracy_score(actual, predicted),
            "Precision (weighted)": precision_score(
                actual, predicted,
                average="weighted",
                zero_division=0
            ),
            "Recall (weighted)": recall_score(
                actual, predicted,
                average="weighted",
                zero_division=0
            ),
            "F1 Score (weighted)": f1_score(
                actual, predicted,
                average="weighted",
                zero_division=0
            )
        }

    baseline_metrics = get_metrics(y_test, baseline_pred)
    encoded_metrics = get_metrics(y_test, encoded_pred)

    results = pd.DataFrame({
        "Original Features": baseline_metrics,
        "Autoencoder Features": encoded_metrics
    })

    st.dataframe(
        results.style.format("{:.4f}"),
        use_container_width=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Baseline Accuracy",
            f"{baseline_metrics['Accuracy']:.2%}"
        )

    with col2:
        st.metric(
            "Autoencoder Accuracy",
            f"{encoded_metrics['Accuracy']:.2%}"
        )

    # ---------------- CONFUSION MATRIX ----------------

    st.subheader("7. Confusion Matrix")

    labels = sorted(np.unique(y_test))

    cm = confusion_matrix(
        y_test,
        encoded_pred,
        labels=labels
    )

    fig, ax = plt.subplots()

    image = ax.imshow(cm, interpolation="nearest")
    fig.colorbar(image, ax=ax)

    ax.set_title("Autoencoder Features - Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center",
                va="center"
            )

    st.pyplot(fig)
    plt.close(fig)

    # ---------------- CLASSIFICATION REPORT ----------------

    st.subheader("8. Classification Report")

    report = classification_report(
        y_test,
        encoded_pred,
        zero_division=0,
        output_dict=True
    )

    st.dataframe(
        pd.DataFrame(report).transpose().round(4),
        use_container_width=True
    )

    # ---------------- DOWNLOAD RESULTS ----------------

    st.subheader("9. Download Results")

    st.download_button(
        "Download Model Comparison CSV",
        data=results.to_csv().encode("utf-8"),
        file_name="diabetes_model_comparison.csv",
        mime="text/csv"
    )

    loss_df = pd.DataFrame({
        "Epoch": range(1, epochs + 1),
        "Reconstruction_Loss": loss_history
    })

    st.download_button(
        "Download Reconstruction Loss CSV",
        data=loss_df.to_csv(index=False).encode("utf-8"),
        file_name="reconstruction_loss.csv",
        mime="text/csv"
    )

else:
    st.subheader("3. Ready to Train")
    st.write(
        "The dataset has been loaded automatically. "
        "Select your settings in the sidebar and click "
        "'Train Stacked Autoencoder' to begin."
    )

    st.caption(
        "For educational and research purposes only. "
        "This application is not a medical diagnostic tool."
    )
