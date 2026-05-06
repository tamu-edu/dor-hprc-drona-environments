#!/usr/bin/env python3
"""
Fine-tuning script template for Graphcore IPU using Optimum Graphcore
"""

import os
import sys
import json
from optimum.graphcore import (
    IPUConfig,
    IPUTrainer,
    IPUTrainingArguments
)[DATASET_IMPORT]
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model

def main():
    print("=" * 70)
    print(" IPU Fine-Tuning with Optimum Graphcore ".center(70))
    print("=" * 70)

    # Load model and tokenizer
    print(f"Loading model: [MODEL_ID]")

    model = AutoModelForCausalLM.from_pretrained(
        "[MODEL_ID]",
        trust_remote_code=True[HF_TOKEN_ARG]
    )

    tokenizer = AutoTokenizer.from_pretrained("[MODEL_ID]", trust_remote_code=True[HF_TOKEN_ARG])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Model loaded successfully")

[DATA_LOADING_SECTION]

    # Configure IPU
    print("Configuring IPU settings...")
    ipu_config = IPUConfig()
[IPU_CONFIG_SECTION]

    # LoRA configuration (IPU uses full precision, not 4-bit quantization)
    print("Preparing model for LoRA fine-tuning...")
    lora_config = LoraConfig(
        r=[LORA_R],
        lora_alpha=[LORA_ALPHA],
        lora_dropout=[LORA_DROPOUT],
        target_modules=[LORA_TARGET_MODULES],
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    print("\nLoRA Configuration:")
    model.print_trainable_parameters()

[DATASET_PROCESSING_SECTION]

    # IPU Training arguments
    print("\nConfiguring IPU training arguments...")
    training_args = IPUTrainingArguments(
        output_dir="[OUTPUT_DIR]",
        num_train_epochs=[NUM_EPOCHS],
        per_device_train_batch_size=[BATCH_SIZE],
        learning_rate=[LEARNING_RATE],
        lr_scheduler_type="[LR_SCHEDULER]",
        logging_dir="[OUTPUT_DIR]/logs",
        logging_steps=[LOGGING_STEPS],
        save_strategy="epoch",
        evaluation_strategy="epoch",
        warmup_ratio=0.1,
        dataloader_drop_last=True,
        dataloader_num_workers=8[TRAINING_ARGS_EXTRA]
    )

    # Create IPU trainer
    trainer = IPUTrainer(
        model=model,
        ipu_config=ipu_config,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )

    # Start training
    print("\nStarting IPU training...")
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

    return True

if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except Exception as e:
        print(f"Error during IPU fine-tuning: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
