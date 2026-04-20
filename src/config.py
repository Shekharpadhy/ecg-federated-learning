from pathlib import Path

DATA_DIR = Path("data/mit-bih-arrhythmia-database-1.0.0")
PTBXL_DIR = Path("data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3")
CACHE_FILE = Path("data/processed_beats.npz")

# 10 well-annotated MIT-BIH records → ~20,000+ beats
RECORDS = ["100", "101", "103", "105", "106", "108", "109", "111", "112", "113"]

LEFT_WINDOW = 100
RIGHT_WINDOW = 100

TEST_SIZE = 0.2
RANDOM_STATE = 42

NUM_CLIENTS = 5
NUM_ROUNDS = 3
LOCAL_EPOCHS = 2
LEARNING_RATE = 1e-3
BATCH_SIZE = 512

MODEL_TYPE = "fc"

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
