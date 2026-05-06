#!/usr/bin/env python3
"""
Text Classification Fine-tuning script template
Single-node training (single GPU or DataParallel for multi-GPU)
Supports both LoRA and full fine-tuning for classification tasks (BERT, RoBERTa, DistilBERT, etc.)
"""

import os
import sys
import json
import torch
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)[DATASET_IMPORT][LORA_IMPORTS][WANDB_IMPORT]

def compute_metrics(eval_pred):
    """Compute accuracy and other metrics for classification."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    # Calculate accuracy
    accuracy = (predictions == labels).mean()

    # Calculate per-class metrics
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def main():
    print("=" * 70)
    print(" Starting Text Classification Fine-Tuning ".center(70))
    print("=" * 70)

[WANDB_INIT]

    # Load model and tokenizer
    print(f"Loading classification model: [MODEL_ID]")
    print(f"Number of classes: [NUM_LABELS]")

[MODEL_LOADING]

    tokenizer = AutoTokenizer.from_pretrained("[MODEL_ID]", trust_remote_code=True[HF_TOKEN_ARG])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Model loaded successfully")

    # GPU Detection and Configuration
    print("\n" + "=" * 70)
    print(" GPU Configuration ".center(70))
    print("=" * 70)
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Number of GPUs detected: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    print("=" * 70 + "\n")

[DATA_LOADING_SECTION]
[LORA_SETUP]
[DATASET_PROCESSING_SECTION]

    # Training arguments
    print("\nConfiguring training arguments...")
    training_args = TrainingArguments(
        output_dir="[OUTPUT_DIR]",
        num_train_epochs=[NUM_EPOCHS],
        per_device_train_batch_size=[BATCH_SIZE],
        learning_rate=[LEARNING_RATE],
        lr_scheduler_type="[LR_SCHEDULER]",
        logging_dir="[OUTPUT_DIR]/logs",
        logging_steps=[LOGGING_STEPS],
        save_strategy="epoch",
        eval_strategy="epoch",
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to=[REPORT_TO][TRAINING_ARGS_EXTRA]
    )

    # Create data collator for classification
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # Log training strategy
    print("\n" + "=" * 70)
    print(" Training Strategy ".center(70))
    print("=" * 70)
    print(f"Task: Text Classification ({[NUM_LABELS]} classes)")
    print(f"Local rank: {training_args.local_rank}")
    print(f"World size: {training_args.world_size}")
    print(f"Distributed type: {training_args.distributed_state.distributed_type if hasattr(training_args, 'distributed_state') else 'not available'}")
    print("=" * 70)

    # Start training
    print("\nStarting training...")
    print("=" * 70)
    trainer.train()

    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    metrics = trainer.evaluate()
    print(f"Validation Accuracy: {metrics['eval_accuracy']:.4f}")
    print(f"Validation F1: {metrics['eval_f1']:.4f}")

    # Save final model
    print("\nTraining complete. Saving final model...")
    output_path = os.path.join("[OUTPUT_DIR]", "final_model")
    os.makedirs(output_path, exist_ok=True)
    trainer.save_model(output_path)
    tokenizer.save_pretrained(output_path)

    print(f"Model saved successfully to {output_path}")
    print("=" * 70)
[WANDB_FINISH]

    return True

if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except Exception as e:
        print(f"Error during fine-tuning: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
