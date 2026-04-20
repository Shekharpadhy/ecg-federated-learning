import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class ECGCNNModel(nn.Module):
    """1D Convolutional Neural Network for ECG beat classification."""

    def __init__(self, input_length=200, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(4)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128 * 4, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)          # (batch, 1, length)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)                # (batch, 128, 4)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class ECGModel(nn.Module):
    """Original fully-connected baseline (kept for reference)."""

    def __init__(self, input_dim=200, num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def build_model(input_dim=200, num_classes=2, model_type="cnn"):
    if model_type == "cnn":
        return ECGCNNModel(input_length=input_dim, num_classes=num_classes)
    return ECGModel(input_dim=input_dim, num_classes=num_classes)


def train_model(model, X, y, epochs=5, lr=1e-3, batch_size=256, verbose=True, prefix=""):
    """Mini-batch training — returns (model, epoch_history)."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    loader = DataLoader(
        TensorDataset(X_tensor, y_tensor),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    history = []
    for epoch in range(1, epochs + 1):
        total_loss, correct, total = 0.0, 0, 0
        for xb, yb in loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(yb)
            correct += (out.argmax(1) == yb).sum().item()
            total += len(yb)

        avg_loss = total_loss / total
        acc = correct / total
        history.append({"epoch": epoch, "loss": round(avg_loss, 4), "accuracy": round(acc, 4)})
        if verbose:
            tag = f"{prefix} " if prefix else ""
            print(f"  {tag}Epoch {epoch}/{epochs}  loss={avg_loss:.4f}  acc={acc:.4f}", flush=True)

    return model, history


def evaluate_model(model, X, y):
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        logits = model(X_tensor)
        preds = torch.argmax(logits, dim=1).cpu().numpy()

    acc = float((preds == y).mean())
    return acc, preds


def get_parameters(model):
    return [val.detach().cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)
