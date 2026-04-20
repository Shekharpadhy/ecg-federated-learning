"""Manual FedAvg simulation — no Ray/flwr.simulation dependency."""
import json

import numpy as np

from .config import (
    BATCH_SIZE,
    LEARNING_RATE,
    LOCAL_EPOCHS,
    MODEL_TYPE,
    NUM_CLIENTS,
    NUM_ROUNDS,
    RESULTS_DIR,
)
from .data_utils import get_train_test_data, split_into_clients
from .model import (
    build_model,
    evaluate_model,
    get_parameters,
    set_parameters,
    train_model,
)


def _fedavg(all_params):
    return [np.mean([cp[i] for cp in all_params], axis=0) for i in range(len(all_params[0]))]


def run_federated():
    X_train, X_test, y_train, y_test, _, label_encoder = get_train_test_data()
    num_classes = len(label_encoder.classes_)
    input_dim = X_train.shape[1]

    client_data = split_into_clients(X_train, y_train, NUM_CLIENTS)
    global_model = build_model(input_dim=input_dim, num_classes=num_classes, model_type=MODEL_TYPE)

    round_log = []

    for round_num in range(1, NUM_ROUNDS + 1):
        global_params = get_parameters(global_model)
        client_params = []

        for x_c, y_c in client_data:
            local_model = build_model(input_dim=input_dim, num_classes=num_classes, model_type=MODEL_TYPE)
            set_parameters(local_model, global_params)
            local_model, _ = train_model(local_model, x_c, y_c, epochs=LOCAL_EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE, verbose=False)
            client_params.append(get_parameters(local_model))

        set_parameters(global_model, _fedavg(client_params))

        acc, _ = evaluate_model(global_model, X_test, y_test)
        loss = 1.0 - acc
        round_log.append({"round": round_num, "accuracy": float(acc), "loss": float(loss)})
        print(f"  [Round {round_num}/{NUM_ROUNDS}] accuracy={acc:.4f}")

    json_path = RESULTS_DIR / "federated_history.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"num_clients": NUM_CLIENTS, "num_rounds": NUM_ROUNDS,
                   "model_type": MODEL_TYPE, "rounds": round_log}, f, indent=2)

    print(f"\nFederated done. Final accuracy: {round_log[-1]['accuracy']:.4f}")
    print(f"Saved → {json_path}")
