# configs/finetuning_config.py

# --- Model Configuration ---
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# --- Dataset Configuration ---
DATASET_ID = "flwrlabs/shakespeare"
DATASET_INPUT_COL = "x"
DATASET_TARGET_COL = "y"
# A smaller subset for quick testing, set to None to use the full dataset
MAX_SAMPLES = 1000 

# --- Fine-Tuning Hyperparameters ---
LEARNING_RATE = 2e-4
NUM_TRAIN_EPOCHS = 3
PER_DEVICE_TRAIN_BATCH_SIZE = 4
PER_DEVICE_EVAL_BATCH_SIZE = 4
WEIGHT_DECAY = 0.01
LR_SCHEDULER_TYPE = "cosine"
WARMUP_RATIO = 0.03
LOGGING_STEPS = 10

# --- LoRA (Low-Rank Adaptation) Configuration ---
# These parameters are crucial for efficient fine-tuning.
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
# Add other layers if needed, e.g., ["q_proj", "v_proj", "k_proj", "o_proj"]
LORA_TARGET_MODULES = ["q_proj", "v_proj"] 

# --- Output and Logging ---
OUTPUT_DIR = "results/tinyllama-shakespeare-finetuned"

