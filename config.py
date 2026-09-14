

CHECKPOINT = "microsoft/deberta-v3-base"

DATASET_NAME = "kuwrom/fallacy"
DATASET_CONFIG = "classification"
TEXT_COLUMN = "text"
LABEL_COLUMN = "label"
DATA_PLOTS_DIR = "plots/data"
EVALUATION_PLOTS_DIR = "plots/evaluation"

BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
EARLY_STOPPING_PATIENCE = 3
SEED = 42
MAX_GRAD_NORM = 1.0
MIXED_PRECISION = "bf16"
RUN_TEST = True
RUN_UMAP_ANALYSIS = False

WANDB_PROJECT = "fallacy-classifier"
WANDB_MODE = None

BEST_MODEL_PATH = "checkpoints/best_model"
