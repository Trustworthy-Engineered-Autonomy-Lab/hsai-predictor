import torch
import pandas as pd
import torch.optim as optim
import numpy as np
from vae import VAE
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import os
from PIL import Image
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix


class LSTM(nn.Module):
    def __init__(
        self,
        num_classes=1,
        in_features=32,
        lstm_units=256,
        num_lstm_layers=1,
        bidirectional=False,
    ):
        super(LSTM, self).__init__()

        # 2. LSTM layers
        # We stack two LSTMs similar to your Keras example
        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=lstm_units,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        # The final output size of LSTM is lstm_units * 2 if bidirectional, else lstm_units
        lstm_output_size = lstm_units * (2 if bidirectional else 1)

        # 3. Dense layer for classification
        self.dense = nn.Linear(lstm_output_size, num_classes)

    def forward(self, x):
        x, _ = self.lstm(x)  # (batch, seq_len, lstm_output_size)

        # (C) Dense (Fully Connected) + Softmax:
        # want only the last time step:
        logits = self.dense(x)  # shape: (batch, seq_len, num_classes)

        # If you need probabilities per timestep:
        probabilities = F.sigmoid(logits)

        return probabilities

    def train_one_epoch(self, data, optimizer, device, seq_len=32):
        """
        :param data: list of dictionaries from load_data()
        :param optimizer: PyTorch optimizer
        :param device: 'cuda' or 'cpu'
        :return: average loss over the epoch
        """
        self.train()
        criterion = nn.BCELoss(reduction="sum")
        running_loss = 0.0

        # Shuffle data if desired
        # import random
        # random.shuffle(data)

        # Process the data
        data = data[0:4000]
        for i in range(0, len(data), 1):
            if i + seq_len == len(data):
                break
            batch = data[i : i + seq_len]

            embeddings = [
                item["embedding"] for item in batch
            ]  # each is shape [1, latent_size]
            labels = [item["label"] for item in batch]  # each is 0 or 1

            # Concatenate embeddings along dim=0 => shape: [seq_len, latent_size]
            embeddings = torch.cat(embeddings, dim=0).unsqueeze(
                0
            )  # shape [1, seq_len, in_features]
            labels = (
                torch.tensor(labels, dtype=torch.float32).unsqueeze(1).unsqueeze(0)
            )  # [1, seq_len, 1]

            # Move tensors to the correct device
            embeddings = embeddings.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = self.forward(embeddings)  # shape [1, seq_len, 1]
            last_time_step = outputs[-1, -1, 0]

            optimizer.zero_grad()
            loss = criterion(last_time_step, labels[0, -1, 0])

            # Backprop
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Average loss per sample
        epoch_loss = running_loss / len(data)
        return epoch_loss

    def train_model(self, data, device="cpu", epochs=50, lr=1e-3):
        """
        Main training loop.

        :param device: torch.device (e.g., 'cuda' or 'cpu')
        :param data: list of dictionaries from load_data()
        :param epochs: number of epochs to train
        :param lr: learning rate
        """
        # Move LSTM to the chosen device
        self.to(device)

        optimizer = optim.Adam(self.parameters(), lr=lr)
        print(f"Original Size: {len(data)}")
        data = data[0:4000]  # Train first 4000
        print(f"New Size: {len(data)}")
        globLoss = 10000
        for epoch in range(epochs):
            epoch_loss = self.train_one_epoch(data, optimizer, device)
            if globLoss > epoch_loss:
                torch.save(self.state_dict(), "lstm_weights.pth")
                globLoss = epoch_loss
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}")

        print(f"Best Loss: {globLoss:.4f}")


def load_image(filepath):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    img = Image.open(filepath).convert("RGB")
    img_tensor = transform(img)
    return img_tensor  # shape [C, H, W]


def load_data(
    csv_path="./safety_detection_labeled_data/Safety_Detection_Labeled.csv",
    images_folder="./safety_detection_labeled_data/",
    vae_weights="./vae_weights.pth",
    device="cpu",
):
    df = pd.read_csv(csv_path)

    model = VAE(latent_size=32)
    # Load VAE weights onto the specified device
    checkpoint = torch.load(vae_weights, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    data = []

    for _, row in df.iterrows():
        filename = row["Filename"]
        label = row["Label"]

        # Build full path to image
        img_path = os.path.join(images_folder, filename)

        # Make sure the file exists before loading
        if not os.path.isfile(img_path):
            print(f"Warning: {img_path} does not exist. Skipping.")
            continue

        # Load and process the image
        x = load_image(img_path)  # shape [C, H, W]
        # Reshape to add batch dimension => shape [1, C, H, W]
        x = x.unsqueeze(0).to(device)

        # Encode with the VAE
        with torch.no_grad():
            output, logvar = model.encode(x)

        # Optionally move the output back to CPU if you prefer
        # embedding = output.cpu()
        # For now, we can store it on CPU to avoid large GPU memory usage
        embedding = output.cpu()

        data.append({
            "filename": filename,
            "embedding": embedding,  # shape [1, latent_size]
            "label": label,
        })
    data = sorted(
        data, key=lambda item: int(item["filename"].split("_")[1].split(".")[0])
    )
    return data


def eval(
    csv_path="../safety_detection_labeled_data/Safety_Detection_Labeled.csv",
    images_folder="../safety_detection_labeled_data/",
    vae_weights="./vae_weights.pth",
    lstm_weights="./lstm_weights.pth",
    seq_len=32,
    device="cpu",
):
    # Load data on the desired device (the embeddings themselves can go to CPU or GPU)
    data = load_data(
        csv_path=csv_path,
        images_folder=images_folder,
        vae_weights=vae_weights,
        device=device,
    )

    # Evaluate on the last samples (skipping the first 4000)
    data = data[4000:-1]

    model = LSTM()
    # Load LSTM weights onto the device
    checkpoint = torch.load(lstm_weights, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    for i in range(0, len(data), 1):
        if i + seq_len == len(data):
            break
        batch = data[i : i + seq_len]

        embeddings = [
            item["embedding"] for item in batch
        ]  # list of [1, latent_size] Tensors
        labels = [item["label"] for item in batch]  # list of 0/1

        embeddings = torch.cat(embeddings, dim=0).unsqueeze(
            0
        )  # shape [1, seq_len, in_features]
        labels = (
            torch.tensor(labels, dtype=torch.float32).unsqueeze(1).unsqueeze(0)
        )  # [1, seq_len, 1]

        # Move tensors to the correct device
        embeddings = embeddings.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model.forward(embeddings)
        last_time_step = outputs[-1, -1, 0]
        pred_label = 1.0 if last_time_step > 0.5 else 0.0
        true_label = labels[0, -1, 0].item()

        # Collect predictions & labels for metric computation
        all_preds.append(pred_label)
        all_labels.append(true_label)

        # Optional: check loss for debugging
        # crit = nn.BCELoss()
        # loss = crit(last_time_step, labels[0, -1, 0])
        # print(f"Loss: {loss}, Prediction: {pred_label}, Label: {true_label}")

    # Once done collecting over the entire dataset, compute metrics:
    all_preds_tensor = torch.tensor(all_preds)
    all_labels_tensor = torch.tensor(all_labels)

    accuracy = accuracy_score(all_labels_tensor, all_preds_tensor)
    f1 = f1_score(all_labels_tensor, all_preds_tensor, zero_division=0)

    # confusion_matrix returns a 2x2 matrix in the format:
    # [[TN, FP],
    #  [FN, TP]]
    tn, fp, fn, tp = confusion_matrix(all_labels_tensor, all_preds_tensor).ravel()

    # False Positive Rate (FPR) = FP / (FP + TN)
    if (fp + tn) == 0:
        fpr = 0.0
    else:
        fpr = fp / (fp + tn)

    # False Negative Rate (FNR) = FN / (FN + TP)
    if (fn + tp) == 0:
        fnr = 0.0
    else:
        fnr = fn / (fn + tp)

    print(f"Accuracy:            {accuracy:.4f}")
    print(f"F1 Score:            {f1:.4f}")
    print(f"False Positive Rate: {fpr:.4f}")
    print(f"False Negaitve Rate: {fnr:.4f}")

    return accuracy, f1, fpr


if __name__ == "__main__":
    device_choice = "cuda" if torch.cuda.is_available() else "cpu"

    # Training
    # data = load_data(device=device_choice)
    # model = LSTM()
    # model.train_model(data=data, device=device_choice)

    # Validation Metrics
    eval(device=device_choice)
