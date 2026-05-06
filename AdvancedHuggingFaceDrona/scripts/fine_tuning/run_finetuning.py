import os
import torch
import torch.distributed as dist
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

import finetuning_config as cfg

def setup_distributed():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    print(f"[Rank {rank}, LocalRank {local_rank}] Process group initialized.", flush=True)
    return rank

def main():
    rank = setup_distributed()

    try:
        # --- Model and Data Loading (with barriers) ---
        if rank == 0:
            print("[Rank 0] Downloading/verifying resources. Others wait.", flush=True)
            # Rank 0 downloads to populate the cache
            load_dataset(cfg.DATASET_ID, split="train", download_mode="force_redownload")
        dist.barrier() # All processes wait until Rank 0 is done

        # All processes now load from cache
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(
            cfg.MODEL_ID, quantization_config=bnb_config, trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_ID, trust_remote_code=True)
        dataset = load_dataset(cfg.DATASET_ID, split="train")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # --- Model and Data Preparation ---
        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=cfg.LORA_R, lora_alpha=cfg.LORA_ALPHA, lora_dropout=cfg.LORA_DROPOUT,
            target_modules=cfg.LORA_TARGET_MODULES, bias="none", task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        if rank == 0:
            model.print_trainable_parameters()

        if cfg.MAX_SAMPLES:
            dataset = dataset.select(range(cfg.MAX_SAMPLES))

        def tokenize_function(examples):
            text = [inp + " " + out for inp, out in zip(examples[cfg.DATASET_INPUT_COL], examples[cfg.DATASET_TARGET_COL])]
            return tokenizer(text, truncation=True, padding="max_length", max_length=512)

        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
        train_test_split = tokenized_dataset.train_test_split(test_size=0.1)

        # --- Configure Trainer (Simplified and Corrected) ---
        training_args = TrainingArguments(
            output_dir=cfg.OUTPUT_DIR,
            num_train_epochs=cfg.NUM_TRAIN_EPOCHS,
            per_device_train_batch_size=cfg.PER_DEVICE_TRAIN_BATCH_SIZE,
            learning_rate=cfg.LEARNING_RATE,
            lr_scheduler_type=cfg.LR_SCHEDULER_TYPE,
            logging_dir=f"{cfg.OUTPUT_DIR}/logs",
            logging_steps=cfg.LOGGING_STEPS,
            save_strategy="no",
            bf16=True,
            report_to="tensorboard",
            ddp_find_unused_parameters=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_test_split['train'],
            tokenizer=tokenizer,
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        )
        
        # --- Start Training ---
        if rank == 0:
            print("\n" + "="*70)
            print(" Starting Model Fine-Tuning ".center(70))
            print("="*70, flush=True)
        
        trainer.train()

        # --- Save Final Model ---
        dist.barrier()
        if rank == 0:
            print("\nTraining complete. Saving final model.", flush=True)
            trainer.save_model(os.path.join(cfg.OUTPUT_DIR, "final_model"))
            tokenizer.save_pretrained(os.path.join(cfg.OUTPUT_DIR, "final_model"))
            print(f"--> Model saved successfully.", flush=True)
    
    finally:
        # ====================================================================
        # THE FIX: This block ensures the process group is always cleaned up.
        # This will remove the warning message.
        # ====================================================================
        if dist.is_initialized():
            print(f"[Rank {rank}] Cleaning up process group.", flush=True)
            dist.destroy_process_group()
        # ====================================================================

if __name__ == "__main__":
    main()