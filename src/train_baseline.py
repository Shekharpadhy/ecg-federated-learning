import json

import torch
from sklearn.metrics import classification_report

from .config import BATCH_SIZE, MODEL_TYPE, RESULTS_DIR
from .data_utils import get_train_test_data
from .model import build_model, evaluate_model, train_model


def run_baseline():
    X_train, X_test, y_train, y_test, _, label_encoder = get_train_test_data()
    num_classes = len(label_encoder.classes_)

    model = build_model(input_dim=X_train.shape[1], num_classes=num_classes, model_type=MODEL_TYPE)
    print(f"\nTraining {MODEL_TYPE.upper()} on {len(X_train)} beats — 7 epochs")
    model, epoch_history = train_model(model, X_train, y_train, epochs=7, lr=1e-3, batch_size=BATCH_SIZE, verbose=True)

    acc, preds = evaluate_model(model, X_test, y_test)
    report = classification_report(y_test, preds, output_dict=True, zero_division=0)

    output = {
        "accuracy": acc,
        "model_type": MODEL_TYPE,
        "num_samples": len(X_train),
        "classes": label_encoder.classes_.tolist(),
        "epoch_history": epoch_history,
        "report": report,
    }

    out_file = RESULTS_DIR / "baseline_metrics.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    model_path = RESULTS_DIR / "baseline_model.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "model_type": MODEL_TYPE,
        "input_dim": X_train.shape[1],
        "num_classes": num_classes,
        "classes": label_encoder.classes_.tolist(),
    }, model_path)

    print(f"Baseline accuracy ({MODEL_TYPE.upper()}): {acc:.4f}")
    return model, X_train, X_test, y_test, label_encoder
