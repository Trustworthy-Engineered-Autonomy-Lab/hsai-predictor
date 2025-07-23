import torch
import pandas as pd
import torch.optim as optim
from vae import VAE
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
import os
import numpy as np
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    precision_score,
    recall_score,
    mean_squared_error,
)


class LSTM(nn.Module):
    def __init__(
        self,
        num_classes=1,
        in_features=32,
        lstm_units=256,
        num_lstm_layers=1,
        bidirectional=True,
    ):
        super(LSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=lstm_units,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        lstm_output_size = lstm_units * (2 if bidirectional else 1)

        self.dense = nn.Linear(lstm_output_size, num_classes)

    def forward(self, x):
        x, _ = self.lstm(x)  # (batch, seq_len, lstm_output_size)

        # (C) Dense (Fully Connected) + Softmax:
        # want only the last time step:
        logits = self.dense(x)  # shape: (batch, seq_len, num_classes)

        # If you need probabilities per timestep:
        probabilities = F.sigmoid(logits)

        return probabilities

    def train_one_epoch(self, data, optimizer, device, seq_len=32, horizon=10):
        """
        :param data: list of dictionaries from load_data()
        :param optimizer: PyTorch optimizer
        :param device: 'cuda' or 'cpu'
        :return: average loss over the epoch
        """
        self.train()
        criterion = nn.BCELoss(reduction="sum")
        running_loss = 0.0

        for i in range(0, len(data), 1):
            if i + seq_len + horizon >= len(data):
                break

            batch = data[i : i + seq_len + horizon]

            embeddings = [
                item["embedding"] for item in batch[0 : len(batch) - horizon]
            ]  # each is shape [1, latent_size]
            labels = []
            for j in range(len(batch) - horizon):
                labels.append(batch[j + horizon]["label"])

            # Concatenate embeddings along dim=0 => shape: [seq_len, latent_size]
            embeddings = torch.cat(embeddings, dim=0).unsqueeze(
                0
            )  # now shape [batch_size=1, seq_len, 32]
            embeddings = embeddings.to(device)

            # Convert labels to a tensor of shape [batch = 1, seq_len, 1]
            labels = (
                torch.tensor(labels, dtype=torch.float32)
                .unsqueeze(1)
                .unsqueeze(0)
                .to(device)
            )

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
        # If reduction='sum', running_loss is the sum over all samples
        epoch_loss = running_loss / len(data)

        return epoch_loss

    def train_model(
        self, data, device="cpu", seq_len=32, horizon=10, epochs=30, lr=1e-3
    ):
        """
        Main training loop.

        :param device: torch.device (e.g., 'cuda' or 'cpu')
        :param data: list of dictionaries from load_data()
        :param epochs: number of epochs to train
        :param lr: learning rate
        """
        # Move LSTM model to the proper device
        self.to(device)

        optimizer = optim.Adam(self.parameters(), lr=lr)
        data = data[0:4000]  # Train first 5000
        finLoss = 1000
        for epoch in range(epochs):
            epoch_loss = self.train_one_epoch(data, optimizer, device, seq_len, horizon)
            if finLoss > epoch_loss:
                finLoss = epoch_loss
                torch.save(self.state_dict(), "./weights/lstm_weights_pred.pth")
                torch.save(self.state_dict(), f"./weights/lstm_weights{horizon}.pth")
            # print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}")

        print(f"Loss: {finLoss:.4f}")


def load_image(filepath):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    img = Image.open(filepath).convert("RGB")
    img_tensor = transform(img)
    return img_tensor  # shape [C, H, W]


def load_data(
    csv_path="../safety_detection_labeled_data/Safety_Detection_Labeled.csv",
    images_folder="../safety_detection_labeled_data/",
    vae_weights="./weights/vae_weights_split.pth",
    device="cpu",
):
    df = pd.read_csv(csv_path)

    model = VAE(latent_size=32)
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
        data.append({
            "filename": filename,
            "embedding": output.cpu(),  # or keep it on GPU if you prefer
            "label": label,
        })
    return data


def eval_train_cc(
    csv_path="../safety_detection_labeled_data/Safety_Detection_Labeled.csv",
    images_folder="../safety_detection_labeled_data/",
    vae_weights="./weights/vae_weights_split.pth",
    lstm_weights="./weights/lstm_weights_pred.pth",
    seq_len=32,
    horizon=10,
    load_lstm_weights=True,
    load_d=True,
    data=None,
    device="cpu",
):
    if load_d:
        data = load_data(
            csv_path=csv_path,
            images_folder=images_folder,
            vae_weights=vae_weights,
            device=device,
        )
    data_val = data[4000:-1]
    data = data[0:4000]

    model = LSTM().to(device)

    vae = VAE(latent_size=32).to(device)
    checkpoint = torch.load(vae_weights, map_location=device, weights_only=True)
    vae.load_state_dict(checkpoint)
    vae.eval()

    if load_lstm_weights:
        checkpoint = torch.load(lstm_weights, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint)
    model.eval()

    all_safety_preds = []
    all_safety_actuals = []
    all_safety_actuals_val = []
    all_safety_preds_val = []

    for i in range(0, len(data), 1):
        if i + seq_len + horizon >= len(data):
            break

        batch = data[i : i + seq_len + horizon]
        embeddings_raw = [item["embedding"] for item in batch[0 : len(batch) - horizon]]
        embeddings = torch.cat(embeddings_raw, dim=0).unsqueeze(0).to(device)

        # Gather the label we want to predict (the label "horizon" steps ahead)
        labels = []
        for j in range(len(batch) - horizon):
            labels.append(batch[j + horizon]["label"])
        labels = (
            torch.tensor(labels, dtype=torch.float32)
            .unsqueeze(1)
            .unsqueeze(0)
            .to(device)
        )

        # Forward pass through LSTM
        outputs = model.forward(embeddings)  # Shape: [1, seq_len, 1]

        # Get the predicted probability from the last time step
        last_time_step = outputs[-1, -1, 0]
        true_label = labels[0, -1, 0].item()

        all_safety_actuals.append(true_label)
        all_safety_preds.append(last_time_step.float().detach())

    data = data_val

    for i in range(0, len(data), 1):
        if i + seq_len + horizon >= len(data):
            break

        batch = data[i : i + seq_len + horizon]
        embeddings_raw = [item["embedding"] for item in batch[0 : len(batch) - horizon]]
        embeddings = torch.cat(embeddings_raw, dim=0).unsqueeze(0).to(device)

        # Gather the label we want to predict (the label "horizon" steps ahead)
        labels = []
        for j in range(len(batch) - horizon):
            labels.append(batch[j + horizon]["label"])
        labels = (
            torch.tensor(labels, dtype=torch.float32)
            .unsqueeze(1)
            .unsqueeze(0)
            .to(device)
        )

        # Forward pass through LSTM
        outputs = model.forward(embeddings)  # Shape: [1, seq_len, 1]

        # Get the predicted probability from the last time step
        last_time_step = outputs[-1, -1, 0]
        true_label = labels[0, -1, 0].item()

        all_safety_actuals_val.append(true_label)
        all_safety_preds_val.append(last_time_step.float().detach())

    return (
        np.array(all_safety_preds),
        np.array(all_safety_actuals),
        np.array(all_safety_preds_val),
        np.array(all_safety_actuals_val),
    )


def eval(
    csv_path="../safety_detection_labeled_data/Safety_Detection_Labeled.csv",
    images_folder="../safety_detection_labeled_data/",
    vae_weights="./weights/vae_weights_split.pth",
    lstm_weights="./weights/lstm_weights_pred.pth",
    seq_len=32,
    horizon=10,
    load_lstm_weights=True,
    load_d=True,
    data=None,
    device="cpu",
):
    if load_d:
        data = load_data(
            csv_path=csv_path,
            images_folder=images_folder,
            vae_weights=vae_weights,
            device=device,
        )

    # Evaluate on the last 5000 samples (skipping the first 4000)
    data = data[4000:-1]

    model = LSTM()
    if load_lstm_weights:
        checkpoint = torch.load(lstm_weights, map_location=device)
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    tot = 0
    lblOne = 0
    all_preds = []
    all_labels = []

    for i in range(0, len(data), 1):
        if i + seq_len + horizon >= len(data):
            break

        batch = data[i : i + seq_len + horizon]

        # Gather the embeddings for the input sequence
        embeddings = [item["embedding"] for item in batch[0 : len(batch) - horizon]]
        embeddings = (
            torch.cat(embeddings, dim=0).unsqueeze(0).to(device)
        )  # [1, seq_len, latent_size]

        # Gather the label we want to predict (the label "horizon" steps ahead)
        labels = []
        for j in range(len(batch) - horizon):
            labels.append(batch[j + horizon]["label"])
        labels = (
            torch.tensor(labels, dtype=torch.float32)
            .unsqueeze(1)
            .unsqueeze(0)
            .to(device)
        )

        # Forward pass through LSTM
        outputs = model.forward(embeddings)  # Shape: [1, seq_len, 1]

        # Get the predicted probability from the last time step
        last_time_step = outputs[-1, -1, 0]
        # Convert probability to binary prediction
        pred_label = 1.0 if last_time_step > 0.5 else 0.0
        true_label = labels[0, -1, 0].item()
        tot += 1
        lblOne += pred_label

        all_preds.append(pred_label)
        all_labels.append(true_label)

    # COMPUTE METRICS
    all_preds_tensor = torch.tensor(all_preds)
    all_labels_tensor = torch.tensor(all_labels)
    # new_list = [1 if item > 0.5 else item for item in my_list]
    accuracy = accuracy_score(all_labels_tensor, all_preds_tensor)
    f1 = f1_score(all_labels_tensor, all_preds_tensor, zero_division=0)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(all_labels_tensor, all_preds_tensor).ravel()
    if (fp + tn) == 0:
        fpr = 0.0
    else:
        fpr = fp / (fp + tn)
    if (fn + tp) == 0:
        fnr = 0.0
    else:
        fnr = fn / (fn + tp)

    precision = precision_score(all_labels_tensor, all_preds_tensor, zero_division=0)
    recall = recall_score(all_labels_tensor, all_preds_tensor, zero_division=0)

    mse_val = mean_squared_error(all_labels_tensor, all_preds_tensor)
    print(f"Percent Unsafe: {lblOne / tot} Total Predictions: {tot}")
    print(f"Accuracy:            {accuracy:.4f}")
    print(f"F1 Score:            {f1:.4f}")
    print(f"Precision:           {precision:.4f}")
    print(f"Recall:              {recall:.4f}")
    print(f"False Positive Rate: {fpr:.4f}")
    print(f"False Negative Rate: {fnr:.4f}")
    print(f"MSE:                 {mse_val:.4f}")

    return accuracy, f1, fpr, fnr, precision, recall, mse_val


if __name__ == "__main__":
    # Decide on the device manually (you can also just do str(device) in train_model/eval calls)
    device_choice = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Grid Search
    # Hyperparameters
    lens = [32]
    horizon_init = 10
    horizon_increment = 10
    horizon_limit = 10

    # Training
    data = load_data(device=device_choice)
    print("DATA loaded")
    model = LSTM()
    for h in range(horizon_init, horizon_limit + 1, horizon_increment):
        for l in lens:
            print(f"Results for Horizon {h} and Sequence Length {l}:")
            print("_______________________________________________")
            model.train_model(
                data=data, device=device_choice, seq_len=l, horizon=h, epochs=100
            )
            acc, f1, fpr, fnr, p, r, mse = eval(
                load_lstm_weights=True,
                load_d=False,
                data=data,
                horizon=h,
                seq_len=l,
                device=device_choice,
            )
            with open("./reliability_results/accuracy_results.txt", "a") as file:
                file.write(f"Results for Horizon {h} and Sequence Length {l}:\n")
                file.write("_______________________________________________\n")
                file.write(f"Accuracy: {acc:.4f} \n")
                file.write(f"F1 Score: {f1:.4f} \n")
                file.write(f"Precision: {p: .4f}\n")
                file.write(f"Recall: {r: .4f}\n")
                file.write(f"MSE: {mse: .4f}\n")
                file.write(f"False Positive Rate: {fpr:.4f}\n")
                file.write(f"False Negative Rate: {fnr:.4f}\n")

    # Basic Train/Test
    # model.train_model(data=data, device=device_choice)
    # eval(load_lstm_weights=True, load_d=True, device=device_choice)
