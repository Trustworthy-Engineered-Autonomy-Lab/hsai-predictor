import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from data_loader import load_embed_label_list  # new loader returns entire split


# ---------------------------------------------------------------------------
class LSTM(nn.Module):
    def __init__(
        self, in_features=32, lstm_units=256, num_layers=1, bidirectional=False
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=lstm_units,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        out_dim = lstm_units * (2 if bidirectional else 1)
        self.fc = nn.Linear(out_dim, 1)

    def forward(self, x):  # x: [B, 1, F]
        out, _ = self.lstm(x)  # out: [B, 1, out_dim]
        logits = self.fc(out[:, -1, :])  # last time-step, shape [B, 1]
        return torch.sigmoid(logits).squeeze(-1)  # [B]


# ---------------------------------------------------------------------------
def train_model(epochs=50, lr=1e-3, device="cpu"):
    # load all embeddings/labels for training split
    records = load_embed_label_list(split="train", device=device)

    model = LSTM().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for rec in records:
            # detach embedding from any previous graph
            emb = (
                rec["embedding"].unsqueeze(0).unsqueeze(1).to(device).detach()
            )  # [1,1,F]
            label = torch.tensor(
                [rec["label"]], dtype=torch.float32, device=device
            )  # [1]

            out = model(emb)  # [1]
            loss = criterion(out, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(records)
        print(f"Epoch {epoch:02}/{epochs} - Train Loss: {avg_loss:.4f}")

        # checkpoint best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "./weights/lstm_weights.pth")

    print(f"Best Train Loss: {best_loss:.4f}")
    return model


# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(lstm_weights="./weights/lstm_weights.pth", device="cpu"):
    model = LSTM().to(device)
    model.load_state_dict(torch.load(lstm_weights, map_location=device))
    model.eval()

    # load test split
    records = load_embed_label_list(split="test", device=device)
    preds, labels = [], []
    for rec in records:
        emb = rec["embedding"].unsqueeze(0).unsqueeze(1).to(device).detach()
        label = float(rec["label"])

        out = model(emb).item()  # scalar
        pred = 1.0 if out > 0.5 else 0.0

        preds.append(pred)
        labels.append(label)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    print(f"Test  Acc: {acc:.4f} | F1: {f1:.4f}")
    print(f"FPR: {fpr:.4f} | FNR: {fnr:.4f}")
    return acc, f1, fpr, fnr


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {dev}")
    train_model(device=dev)
    evaluate(device=dev)
