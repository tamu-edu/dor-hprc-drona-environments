#!/usr/bin/env python3
"""Fine-tuning setup using drona_add_mapping for placeholders."""

from drona_utils import drona_add_mapping, drona_add_additional_file, drona_add_note, drona_add_error
import json


def setup_finetuning(operation, model_id="", dataset_id="", dataset_source="", input_column="", target_column="",
                    max_samples="", hf_token="", location="", model_source="hub",
                    gpu_type="auto", num_gpus="1", nodes="1", distributed_type="DDP",
                    learning_rate=2e-4, num_train_epochs=3, per_device_train_batch_size=4,
                    lora_r=8, lora_alpha=16, lora_dropout=0.05, custom_finetune_params=None,
                    test_size=0.1, ddp_bucket_cap_mb=25, use_lora=False,
                    enable_wandb=False, wandb_api_key="", wandb_project="hf-finetuning",
                    wandb_run_name="", wandb_tags="", wandb_notes="",
                    task_type="causal_lm", text_column="text", label_column="label", num_labels=2,
                    **kwargs):
    """Setup fine-tuning using drona mappings."""

    # Normalize use_lora (could be boolean, string "Yes"/"No"/"false", or dict)
    if isinstance(use_lora, dict):
        use_lora_value = use_lora.get("value", False) if "value" in use_lora else (use_lora.get("Yes", False) if "Yes" in use_lora else False)
    elif isinstance(use_lora, str):
        # Handle various string formats: "Yes", "No", "true", "false", "True", "False"
        use_lora_lower = use_lora.lower().strip()
        use_lora_value = (use_lora_lower in ["yes", "true", "1"])
    else:
        use_lora_value = bool(use_lora) if use_lora is not None else False

    # Debug output
    drona_add_note(f"DEBUG: use_lora parameter = {use_lora}")
    drona_add_note(f"DEBUG: use_lora_value = {use_lora_value}")

    # Validate
    if not model_id:
        drona_add_error("Model ID required")
        return ""

    if dataset_source not in ["hub", "downloaded"]:
        drona_add_error("Invalid dataset source. Only 'hub' and 'downloaded' are supported for fine-tuning.")
        return ""

    if not dataset_id:
        drona_add_error("Dataset ID required for fine-tuning")
        return ""

    # Normalize task_type
    if isinstance(task_type, dict):
        task_type_value = task_type.get("value", "causal_lm")
    else:
        task_type_value = task_type if task_type else "causal_lm"

    # Debug: Show what we received
    drona_add_note(f"DEBUG finetuning_setup: task_type = {task_type_value}, distributed_type = {repr(distributed_type)}, num_gpus = {num_gpus}")

    # Determine which template to use based on task type AND distributed type
    if task_type_value == "text_classification":
        # Classification task
        if distributed_type == "DDP":
            drona_add_additional_file("scripts/fine_tuning/finetuning_classification_ddp.py", "finetuning.py")
            drona_add_note("Using Text Classification DDP template")
            drona_add_mapping("DDP_BUCKET_CAP", str(ddp_bucket_cap_mb))
        else:
            # MULTI_GPU, NONE, or single GPU
            drona_add_additional_file("scripts/fine_tuning/finetuning_classification.py", "finetuning.py")
            drona_add_note(f"Using Text Classification template (distributed_type={distributed_type})")

        # Classification-specific mappings
        drona_add_mapping("NUM_LABELS", str(num_labels))
        drona_add_mapping("TEXT_COLUMN", text_column if text_column else "text")
        drona_add_mapping("LABEL_COLUMN", label_column if label_column else "label")
        drona_add_note(f"Classification: {num_labels} classes, text='{text_column}', label='{label_column}'")

    else:
        # Text generation (causal LM) - default behavior
        if distributed_type == "DDP":
            drona_add_additional_file("scripts/fine_tuning/finetuning_ddp.py", "finetuning.py")
            drona_add_note("Using Text Generation DDP template")
            drona_add_mapping("DDP_BUCKET_CAP", str(ddp_bucket_cap_mb))
        elif distributed_type == "MULTI_GPU":
            drona_add_additional_file("scripts/fine_tuning/finetuning.py", "finetuning.py")
            drona_add_note("Using Text Generation DataParallel template")
        else:
            # NONE or single GPU
            drona_add_additional_file("scripts/fine_tuning/finetuning.py", "finetuning.py")
            drona_add_note(f"Using Text Generation template (distributed_type={distributed_type})")

    # Map simple placeholders
    drona_add_mapping("MODEL_ID", model_id)
    drona_add_mapping("OUTPUT_DIR", location if location else "results/finetuned_model")

    # Handle HF token (for tokenizer and other uses)
    if hf_token and hf_token.strip() and not hf_token.startswith("$"):
        drona_add_mapping("HF_TOKEN_ARG", f', token="{hf_token}"')
    else:
        drona_add_mapping("HF_TOKEN_ARG", "")

    # Dataset import
    drona_add_mapping("DATASET_IMPORT", "\nfrom datasets import load_dataset")

    # Determine model class and task type based on task_type_value
    if task_type_value == "text_classification":
        model_class = "AutoModelForSequenceClassification"
        peft_task_type = "SEQ_CLS"
    else:
        # Default: causal LM
        model_class = "AutoModelForCausalLM"
        peft_task_type = "CAUSAL_LM"

    # Generate LoRA-specific or full fine-tuning code
    if use_lora_value:
        # LoRA mode: add PEFT imports, quantization, and LoRA setup
        drona_add_mapping("LORA_IMPORTS", "\nfrom transformers import BitsAndBytesConfig\nfrom peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training")

        # Build model loading with proper token handling
        token_arg = f', token="{hf_token}"' if (hf_token and hf_token.strip() and not hf_token.startswith("$")) else ""

        # Add num_labels for classification
        if task_type_value == "text_classification":
            extra_args = f",\n        num_labels={num_labels}"
        else:
            extra_args = ""

        model_loading_code = f"""    # Quantization configuration for efficient LoRA training
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    model = {model_class}.from_pretrained(
        "{model_id}",
        quantization_config=bnb_config,
        trust_remote_code=True{token_arg}{extra_args}
    )
"""
        drona_add_mapping("MODEL_LOADING", model_loading_code)

        # LoRA setup code
        lora_setup_code = f"""    # Prepare model for LoRA training
    print("Preparing model for LoRA fine-tuning...")
    model = prepare_model_for_kbit_training(model)

    # LoRA configuration
    lora_config = LoraConfig(
        r=[LORA_R],
        lora_alpha=[LORA_ALPHA],
        lora_dropout=[LORA_DROPOUT],
        target_modules=[LORA_TARGET_MODULES],
        bias="none",
        task_type="{peft_task_type}"
    )

    model = get_peft_model(model, lora_config)
    print("\\nLoRA Configuration:")
    model.print_trainable_parameters()
"""
        drona_add_mapping("LORA_SETUP", lora_setup_code)
        drona_add_note("Fine-tuning mode: LoRA (Low-Rank Adaptation)")
    else:
        # Full fine-tuning mode: no PEFT, no quantization
        drona_add_mapping("LORA_IMPORTS", "")

        # Build model loading with proper token handling
        token_arg = f', token="{hf_token}"' if (hf_token and hf_token.strip() and not hf_token.startswith("$")) else ""

        # Add num_labels for classification
        if task_type_value == "text_classification":
            extra_args = f",\n        num_labels={num_labels}"
        else:
            extra_args = ""

        model_loading_code = f"""    # Load model for full fine-tuning
    model = {model_class}.from_pretrained(
        "{model_id}",
        torch_dtype=torch.float16,
        trust_remote_code=True{token_arg}{extra_args}
    )
"""
        drona_add_mapping("MODEL_LOADING", model_loading_code)
        drona_add_mapping("LORA_SETUP", "")
        drona_add_note("Fine-tuning mode: Full model fine-tuning (no LoRA)")

    # Map data loading section
    data_section = generate_data_loading_section(dataset_id, dataset_source, hf_token)
    drona_add_mapping("DATA_LOADING_SECTION", data_section)

    # Map dataset processing section
    processing_section = generate_dataset_processing_section(
        task_type_value, input_column, target_column, text_column, label_column, max_samples, test_size
    )
    drona_add_mapping("DATASET_PROCESSING_SECTION", processing_section)

    # Map LoRA parameters
    drona_add_mapping("LORA_R", str(lora_r))
    drona_add_mapping("LORA_ALPHA", str(lora_alpha))
    drona_add_mapping("LORA_DROPOUT", str(lora_dropout))

    # Handle LoRA target modules from custom params or use defaults
    if custom_finetune_params and isinstance(custom_finetune_params, dict):
        lora_targets = custom_finetune_params.get("lora_target_modules", ["q_proj", "v_proj"])
    elif custom_finetune_params and isinstance(custom_finetune_params, str):
        # Parse JSON string
        try:
            params_dict = json.loads(custom_finetune_params)
            lora_targets = params_dict.get("lora_target_modules", ["q_proj", "v_proj"])
        except:
            lora_targets = ["q_proj", "v_proj"]
    else:
        lora_targets = ["q_proj", "v_proj"]

    # Format as Python list string
    lora_targets_str = json.dumps(lora_targets)
    drona_add_mapping("LORA_TARGET_MODULES", lora_targets_str)

    # Map training parameters
    drona_add_mapping("NUM_EPOCHS", str(num_train_epochs))
    drona_add_mapping("BATCH_SIZE", str(per_device_train_batch_size))
    drona_add_mapping("LEARNING_RATE", str(learning_rate))

    # Handle custom training args
    training_args_extra = generate_custom_training_args(custom_finetune_params, distributed_type)
    drona_add_mapping("TRAINING_ARGS_EXTRA", training_args_extra)

    # Get lr_scheduler from custom params or use default
    lr_scheduler = "cosine"
    logging_steps = 10
    if custom_finetune_params:
        if isinstance(custom_finetune_params, dict):
            lr_scheduler = custom_finetune_params.get("lr_scheduler_type", "cosine")
            logging_steps = custom_finetune_params.get("logging_steps", 10)
        elif isinstance(custom_finetune_params, str):
            try:
                params_dict = json.loads(custom_finetune_params)
                lr_scheduler = params_dict.get("lr_scheduler_type", "cosine")
                logging_steps = params_dict.get("logging_steps", 10)
            except:
                pass

    drona_add_mapping("LR_SCHEDULER", lr_scheduler)
    drona_add_mapping("LOGGING_STEPS", str(logging_steps))

    # Handle wandb configuration
    setup_wandb_integration(enable_wandb, wandb_api_key, wandb_project, wandb_run_name, wandb_tags, wandb_notes,
                           model_id, learning_rate, num_train_epochs, per_device_train_batch_size)

    # Set command - use srun for DDP, python for others
    if distributed_type == "DDP":
        cmd = "srun python finetuning.py"
        drona_add_mapping("OPERATION_CMD", cmd)
        drona_add_note(f"DEBUG: Set OPERATION_CMD = '{cmd}'")
        drona_add_note("DDP command: srun will spawn processes with proper environment variables")
    else:
        cmd = "python finetuning.py"
        drona_add_mapping("OPERATION_CMD", cmd)
        drona_add_note(f"DEBUG: Set OPERATION_CMD = '{cmd}'")

    drona_add_mapping("OPERATION_DESC", f"Fine-tuning: {model_id}")

    # Add notes
    drona_add_note(f"Model: {model_id} (source: {model_source})")
    drona_add_note(f"Dataset: {dataset_id} ({dataset_source})")
    if input_column and target_column:
        drona_add_note(f"Columns: input={input_column}, target={target_column}")
    drona_add_note(f"LoRA config: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
    drona_add_note(f"Training: {num_train_epochs} epochs, batch_size={per_device_train_batch_size}, lr={learning_rate}")
    drona_add_note(f"GPU config: {gpu_type}, {num_gpus} GPU(s), {nodes} node(s)")

    return ""


def setup_wandb_integration(enable_wandb, wandb_api_key, wandb_project, wandb_run_name, wandb_tags, wandb_notes,
                           model_id, learning_rate, num_train_epochs, per_device_train_batch_size):
    """Setup Weights & Biases integration for experiment tracking."""

    drona_add_note(f"DEBUG: setup_wandb_integration called - enable_wandb={enable_wandb} (type: {type(enable_wandb).__name__})")

    # Normalize enable_wandb (could be boolean, string "Yes"/"No", or dict)
    if isinstance(enable_wandb, dict):
        enable_wandb_value = enable_wandb.get("value", False) if "value" in enable_wandb else (enable_wandb.get("Yes", False) if "Yes" in enable_wandb else False)
    elif isinstance(enable_wandb, str):
        enable_wandb_value = (enable_wandb.lower() == "yes" or enable_wandb.lower() == "true")
    else:
        enable_wandb_value = bool(enable_wandb) if enable_wandb is not None else False

    drona_add_note(f"DEBUG: enable_wandb_value={enable_wandb_value}")

    if not enable_wandb_value:
        # Wandb disabled - no logging (tensorboard not installed)
        drona_add_note("DEBUG: Wandb DISABLED - setting empty placeholders")
        drona_add_mapping("WANDB_IMPORT", "")
        drona_add_mapping("WANDB_INIT", "")
        drona_add_mapping("WANDB_FINISH", "")
        drona_add_mapping("REPORT_TO", '"none"')
        return

    # Wandb enabled
    drona_add_note("DEBUG: Wandb ENABLED - configuring integration")
    drona_add_note("Weights & Biases tracking enabled")

    # Add wandb import
    drona_add_mapping("WANDB_IMPORT", "\nimport wandb")

    # Build wandb initialization code
    wandb_init_lines = []

    # Handle API key
    if wandb_api_key and wandb_api_key.strip() and not wandb_api_key.startswith("$"):
        wandb_init_lines.append(f'    os.environ["WANDB_API_KEY"] = "{wandb_api_key}"')
    else:
        wandb_init_lines.append('    # Using WANDB_API_KEY from environment')

    # Build wandb.init() arguments
    init_args = []

    # Project (required when wandb is enabled)
    project_value = wandb_project if wandb_project and wandb_project.strip() else "hf-finetuning"
    init_args.append(f'project="{project_value}"')

    # Run name (optional)
    if wandb_run_name and wandb_run_name.strip() and not wandb_run_name.startswith("$"):
        init_args.append(f'name="{wandb_run_name}"')

    # Tags (optional)
    if wandb_tags and wandb_tags.strip() and not wandb_tags.startswith("$"):
        # Split comma-separated tags and format as Python list
        tags_list = [tag.strip() for tag in wandb_tags.split(",") if tag.strip()]
        if tags_list:
            tags_str = json.dumps(tags_list)
            init_args.append(f'tags={tags_str}')

    # Notes (optional)
    if wandb_notes and wandb_notes.strip() and not wandb_notes.startswith("$"):
        # Escape quotes in notes
        escaped_notes = wandb_notes.replace('"', '\\"')
        init_args.append(f'notes="{escaped_notes}"')

    # Add model info to config
    init_args.append(f'config={{"model_id": "{model_id}", "learning_rate": {learning_rate}, "num_epochs": {num_train_epochs}, "batch_size": {per_device_train_batch_size}}}')

    # Build the complete init code
    wandb_init_lines.append(f'    wandb.init({", ".join(init_args)})')
    wandb_init_code = "\n".join(wandb_init_lines)

    drona_add_mapping("WANDB_INIT", wandb_init_code)
    drona_add_mapping("WANDB_FINISH", "\n    wandb.finish()")

    # Update report_to to use wandb only (tensorboard not installed)
    drona_add_mapping("REPORT_TO", '"wandb"')

    # Add note about configuration
    drona_add_note(f"Wandb project: {project_value}")
    if wandb_run_name:
        drona_add_note(f"Wandb run name: {wandb_run_name}")

    drona_add_note(f"DEBUG: Wandb mappings set - REPORT_TO='wandb', init code length={len(wandb_init_code)}")


def generate_data_loading_section(dataset_id, dataset_source, hf_token):
    """Generate data loading section based on source."""

    # Token argument
    token_arg = f', token="{hf_token}"' if hf_token and hf_token.strip() and not hf_token.startswith("$") else ""

    return f"""
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("{dataset_id}"{token_arg})

    # Get training split (try common names)
    if "train" in dataset:
        data = dataset["train"]
    elif "training" in dataset:
        data = dataset["training"]
    else:
        # Use first available split
        split_name = list(dataset.keys())[0]
        data = dataset[split_name]
        print(f"Using split: {{split_name}}")

    print(f"Loaded {{len(data)}} samples from dataset")
"""


def generate_dataset_processing_section(task_type, input_column, target_column, text_column, label_column, max_samples, test_size):
    """Generate dataset processing section with tokenization for different task types."""

    max_samples_code = ""
    # Check if max_samples is set and not a placeholder like "$max_samples"
    if max_samples and str(max_samples).strip() and not str(max_samples).startswith("$") and str(max_samples) != "":
        try:
            max_samples_int = int(max_samples)
            max_samples_code = f"""
    # Limit dataset size for testing
    if len(data) > {max_samples_int}:
        data = data.select(range({max_samples_int}))
        print(f"Limited to {{len(data)}} samples")
"""
        except (ValueError, TypeError):
            # Invalid max_samples value, skip limiting
            pass

    test_size_val = test_size if test_size else 0.1

    # Generate different processing code based on task type
    if task_type == "text_classification":
        # Text Classification processing
        has_text = text_column and str(text_column).strip() and not str(text_column).startswith("$")
        has_label = label_column and str(label_column).strip() and not str(label_column).startswith("$")

        if has_text and has_label:
            column_code = f"""
    # Use specified columns for classification
    text_col = "{text_column}"
    label_col = "{label_column}"
"""
        else:
            column_code = """
    # Auto-detect columns for classification
    columns = data.column_names
    print(f"Available columns: {columns}")

    # Try common column name patterns
    text_candidates = ['text', 'sentence', 'review', 'content', 'input', 'question']
    label_candidates = ['label', 'labels', 'category', 'class', 'sentiment', 'target']

    text_col = None
    label_col = None

    for col in columns:
        col_lower = col.lower()
        if text_col is None and any(cand in col_lower for cand in text_candidates):
            text_col = col
        if label_col is None and any(cand in col_lower for cand in label_candidates):
            label_col = col

    # Fallback to first two columns if auto-detection fails
    if text_col is None or label_col is None:
        if len(columns) >= 2:
            text_col = columns[0]
            label_col = columns[1]
        else:
            raise ValueError("Could not auto-detect text/label columns. Please specify them explicitly.")

    print(f"Using columns - text: {text_col}, label: {label_col}")
"""

        return f"""{max_samples_code}{column_code}
    # Tokenize and prepare dataset for classification
    print("Tokenizing dataset for classification...")

    def tokenize_function(examples):
        # Tokenize text for classification
        tokenized = tokenizer(examples[text_col], truncation=True, padding="max_length", max_length=512)

        # Ensure labels are integers
        labels = examples[label_col]
        if isinstance(labels[0], str):
            # Convert string labels to integers
            unique_labels = list(set(labels))
            label_to_id = {{label: i for i, label in enumerate(unique_labels)}}
            tokenized["labels"] = [label_to_id[label] for label in labels]
            print(f"Label mapping: {{label_to_id}}")
        else:
            tokenized["labels"] = labels

        return tokenized

    tokenized_dataset = data.map(tokenize_function, batched=True, remove_columns=data.column_names)

    # Split into train/eval
    train_test_split = tokenized_dataset.train_test_split(test_size={test_size_val})
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']

    print(f"Training samples: {{len(train_dataset)}}")
    print(f"Evaluation samples: {{len(eval_dataset)}}")
"""

    else:
        # Text Generation (Causal LM) processing
        has_input = input_column and str(input_column).strip() and not str(input_column).startswith("$")
        has_target = target_column and str(target_column).strip() and not str(target_column).startswith("$")

        if has_input and has_target:
            column_code = f"""
    # Use specified columns
    input_col = "{input_column}"
    target_col = "{target_column}"
"""
        else:
            column_code = """
    # Auto-detect columns
    columns = data.column_names
    print(f"Available columns: {columns}")

    # Try common column name patterns
    input_candidates = ['input', 'instruction', 'question', 'prompt', 'x', 'text']
    target_candidates = ['output', 'response', 'answer', 'completion', 'y', 'label']

    input_col = None
    target_col = None

    for col in columns:
        col_lower = col.lower()
        if input_col is None and any(cand in col_lower for cand in input_candidates):
            input_col = col
        if target_col is None and any(cand in col_lower for cand in target_candidates):
            target_col = col

    # Fallback to first two columns if auto-detection fails
    if input_col is None or target_col is None:
        if len(columns) >= 2:
            input_col = columns[0]
            target_col = columns[1]
        else:
            raise ValueError("Could not auto-detect input/target columns. Please specify them explicitly.")

    print(f"Using columns - input: {input_col}, target: {target_col}")
"""

        return f"""{max_samples_code}{column_code}
    # Tokenize and prepare dataset
    print("Tokenizing dataset...")

    def tokenize_function(examples):
        # Combine input and target for causal LM training
        texts = [str(inp) + " " + str(out) for inp, out in zip(examples[input_col], examples[target_col])]
        return tokenizer(texts, truncation=True, padding="max_length", max_length=512)

    tokenized_dataset = data.map(tokenize_function, batched=True, remove_columns=data.column_names)

    # Split into train/eval
    train_test_split = tokenized_dataset.train_test_split(test_size={test_size_val})
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']

    print(f"Training samples: {{len(train_dataset)}}")
    print(f"Evaluation samples: {{len(eval_dataset)}}")
"""


def generate_custom_training_args(custom_params, distributed_type):
    """Generate extra training arguments from custom parameters."""

    if not custom_params:
        return ""

    # Parse custom params
    if isinstance(custom_params, str):
        try:
            custom_params = json.loads(custom_params)
        except:
            return ""

    if not isinstance(custom_params, dict):
        return ""

    # Skip certain keys that are already handled
    skip_keys = {
        "lr_scheduler_type", "lora_target_modules", "learning_rate",
        "num_train_epochs", "per_device_train_batch_size", "logging_steps"
    }

    extra_args = []

    for key, value in custom_params.items():
        if key in skip_keys:
            continue

        # Format value appropriately
        if isinstance(value, str):
            extra_args.append(f'{key}="{value}"')
        elif isinstance(value, bool):
            extra_args.append(f'{key}={str(value)}')
        elif isinstance(value, (int, float)):
            extra_args.append(f'{key}={value}')

    # Add DDP-specific args if using DDP
    if distributed_type == "DDP":
        extra_args.append('ddp_find_unused_parameters=False')

    if extra_args:
        return ",\n        " + ",\n        ".join(extra_args)
    else:
        return ""
