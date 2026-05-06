#!/usr/bin/env python3
"""
Text Classification Fine-tuning script with DDP (Distributed Data Parallel) support
Multi-node and multi-GPU training for classification tasks
Supports both LoRA and full fine-tuning
"""

import os
import sys
import json
import torch
import torch.distributed as dist
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)[DATASET_IMPORT][LORA_IMPORTS][WANDB_IMPORT]

def setup_distributed():
    """Initialize distributed training environment with SLURM support."""
    # Read rank information from either PyTorch or SLURM environment variables
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))

    # Set MASTER_ADDR and MASTER_PORT for DDP coordination
    if "MASTER_ADDR" not in os.environ:
        # Get first node from SLURM_NODELIST if available
        if "SLURM_NODELIST" in os.environ:
            import subprocess
            try:
                master_addr = subprocess.check_output(
                    ["scontrol", "show", "hostname", os.environ["SLURM_NODELIST"]]
                ).decode().strip().split('\n')[0]
                os.environ["MASTER_ADDR"] = master_addr
            except:
                os.environ["MASTER_ADDR"] = "localhost"
        else:
            os.environ["MASTER_ADDR"] = "localhost"

    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"

    # Export variables for consistency
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    if rank == 0:
        print("=" * 70)
        print(" Distributed Text Classification with DDP ".center(70))
        print("=" * 70)
        print(f"Master Address: {os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}")
        print(f"World Size: {world_size}")
        print(f"Rank: {rank}, Local Rank: {local_rank}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print("=" * 70)

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)

    return rank, local_rank, world_size

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
    # Setup distributed training
    rank, local_rank, world_size = setup_distributed()

    try:
        # Only rank 0 prints detailed logs
        verbose = (rank == 0)

[WANDB_INIT]

        if verbose:
            # GPU Detection (only rank 0 prints to avoid spam)
            print("\n" + "=" * 70)
            print(" GPU Configuration ".center(70))
            print("=" * 70)
            print(f"CUDA available: {torch.cuda.is_available()}")
            print(f"Total processes (world_size): {world_size}")
            print(f"GPUs visible to this process: {torch.cuda.device_count()}")
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
            print(f"Current process using GPU: {local_rank}")
            print("=" * 70 + "\n")

            print(f"Loading classification model: [MODEL_ID]")
            print(f"Number of classes: [NUM_LABELS]")

[MODEL_LOADING]

        tokenizer = AutoTokenizer.from_pretrained("[MODEL_ID]", trust_remote_code=True[HF_TOKEN_ARG])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if verbose:
            print(f"Model loaded successfully on rank {rank}")

[DATA_LOADING_SECTION]
[LORA_SETUP]
[DATASET_PROCESSING_SECTION]

        # Training arguments for DDP
        if verbose:
            print("\nConfiguring training arguments for DDP...")

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
            ddp_find_unused_parameters=False,
            ddp_bucket_cap_mb=[DDP_BUCKET_CAP],
            local_rank=local_rank,
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

        # Log training strategy (only rank 0)
        if verbose:
            print("\n" + "=" * 70)
            print(" Training Strategy ".center(70))
            print("=" * 70)
            print(f"Task: Text Classification ({[NUM_LABELS]} classes)")
            print(f"DDP Backend: {dist.get_backend()}")
            print(f"World size: {world_size}")
            print(f"Global rank: {rank}")
            print(f"Local rank: {local_rank}")
            print("=" * 70)

        # Start training
        if verbose:
            print("\nStarting distributed training...")
            print("=" * 70)

        trainer.train()

        # Evaluate on validation set (only rank 0)
        if verbose:
            print("\nEvaluating on validation set...")
            metrics = trainer.evaluate()
            print(f"Validation Accuracy: {metrics['eval_accuracy']:.4f}")
            print(f"Validation F1: {metrics['eval_f1']:.4f}")

        # Save final model (only rank 0 saves to avoid conflicts)
        if rank == 0:
            print("\nTraining complete. Saving final model...")
            output_path = os.path.join("[OUTPUT_DIR]", "final_model")
            os.makedirs(output_path, exist_ok=True)
            trainer.save_model(output_path)
            tokenizer.save_pretrained(output_path)

            print(f"Model saved successfully to {output_path}")
            print("=" * 70)

[WANDB_FINISH]

    finally:
        # Clean up distributed process group
        if dist.is_initialized():
            dist.destroy_process_group()

    return True

if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except Exception as e:
        rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
        if rank == 0:
            print(f"Error during fine-tuning: {e}")
            import traceback
            traceback.print_exc()
        sys.exit(1)
