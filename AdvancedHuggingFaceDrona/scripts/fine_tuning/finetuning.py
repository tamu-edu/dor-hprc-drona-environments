#!/usr/bin/env python3
"""
Fine-tuning script template - placeholders filled by utils.py
Single-node training (single GPU or DataParallel for multi-GPU)
Supports both LoRA and full fine-tuning
"""

import os
import sys
import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)[DATASET_IMPORT][LORA_IMPORTS][WANDB_IMPORT]

def main():
    print("=" * 70)
    print(" Starting Model Fine-Tuning ".center(70))
    print("=" * 70)

[WANDB_INIT]

    # Load model and tokenizer
    print(f"Loading model: [MODEL_ID]")

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
        fp16=True,
        report_to=[REPORT_TO][TRAINING_ARGS_EXTRA]
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )

    # Log training strategy
    print("\n" + "=" * 70)
    print(" Training Strategy ".center(70))
    print("=" * 70)
    print(f"Local rank: {training_args.local_rank}")
    print(f"World size: {training_args.world_size}")
    print(f"Distributed type: {training_args.distributed_state.distributed_type if hasattr(training_args, 'distributed_state') else 'not available'}")
    print("=" * 70)

    # Start training
    print("\nStarting training...")
    print("=" * 70)
    trainer.train()

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
