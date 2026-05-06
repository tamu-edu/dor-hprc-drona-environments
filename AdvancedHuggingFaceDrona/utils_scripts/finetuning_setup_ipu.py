#!/usr/bin/env python3
"""IPU fine-tuning setup using drona_add_mapping for placeholders."""

from drona_utils import drona_add_mapping, drona_add_additional_file, drona_add_note, drona_add_error
import json


def setup_finetuning_ipu(operation, model_id="", dataset_id="", dataset_source="", input_column="", target_column="",
                        max_samples="", hf_token="", location="", model_source="hub",
                        ipu_type="mk2_pod4", ipu_use_pipeline=False,
                        learning_rate=2e-4, num_train_epochs=3, per_device_train_batch_size=2,
                        lora_r=8, lora_alpha=16, lora_dropout=0.05, custom_finetune_params=None,
                        test_size=0.1, **kwargs):
    """Setup IPU fine-tuning using Optimum Graphcore."""

    drona_add_note("DEBUG: Inside setup_finetuning_ipu()")
    drona_add_note(f"DEBUG: model_id={model_id}, ipu_type={ipu_type}")

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

    # Add fine-tuning IPU file
    drona_add_note("DEBUG: Adding finetuning_ipu.py file")
    drona_add_additional_file("scripts/fine_tuning/finetuning_ipu.py", "finetuning_ipu.py")
    drona_add_note("DEBUG: File added successfully")

    # Map model ID and output
    drona_add_mapping("MODEL_ID", model_id)
    drona_add_mapping("OUTPUT_DIR", location if location else "results/finetuned_model_ipu")

    # Handle HF token
    if hf_token and hf_token.strip() and not hf_token.startswith("$"):
        drona_add_mapping("HF_TOKEN_ARG", f', token="{hf_token}"')
        hf_token_code = f', token="{hf_token}"'
    else:
        drona_add_mapping("HF_TOKEN_ARG", "")
        hf_token_code = ""

    # Dataset import
    drona_add_mapping("DATASET_IMPORT", "\nfrom datasets import load_dataset")

    # Generate data loading section
    data_section = generate_data_loading_section(dataset_id, dataset_source, hf_token_code)
    drona_add_mapping("DATA_LOADING_SECTION", data_section)

    # Generate dataset processing section
    processing_section = generate_dataset_processing_section(
        input_column, target_column, max_samples, test_size
    )
    drona_add_mapping("DATASET_PROCESSING_SECTION", processing_section)

    # Map LoRA parameters
    drona_add_mapping("LORA_R", str(lora_r))
    drona_add_mapping("LORA_ALPHA", str(lora_alpha))
    drona_add_mapping("LORA_DROPOUT", str(lora_dropout))

    # Handle LoRA target modules
    if custom_finetune_params and isinstance(custom_finetune_params, dict):
        lora_targets = custom_finetune_params.get("lora_target_modules", ["q_proj", "v_proj"])
    elif custom_finetune_params and isinstance(custom_finetune_params, str):
        try:
            params_dict = json.loads(custom_finetune_params)
            lora_targets = params_dict.get("lora_target_modules", ["q_proj", "v_proj"])
        except:
            lora_targets = ["q_proj", "v_proj"]
    else:
        lora_targets = ["q_proj", "v_proj"]

    lora_targets_str = json.dumps(lora_targets)
    drona_add_mapping("LORA_TARGET_MODULES", lora_targets_str)

    # Map training parameters
    drona_add_mapping("NUM_EPOCHS", str(num_train_epochs))
    drona_add_mapping("BATCH_SIZE", str(per_device_train_batch_size))
    drona_add_mapping("LEARNING_RATE", str(learning_rate))

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

    # IPU-specific configuration
    ipu_config_section = generate_ipu_config_section(ipu_type)
    drona_add_mapping("IPU_CONFIG_SECTION", ipu_config_section)

    # Handle custom training args for IPU
    training_args_extra = generate_custom_training_args_ipu(custom_finetune_params)
    drona_add_mapping("TRAINING_ARGS_EXTRA", training_args_extra)

    # Add notes
    drona_add_note(f"Model: {model_id} (source: {model_source})")
    drona_add_note(f"Dataset: {dataset_id} ({dataset_source})")
    if input_column and target_column:
        drona_add_note(f"Columns: input={input_column}, target={target_column}")
    drona_add_note(f"LoRA config: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
    drona_add_note(f"Training: {num_train_epochs} epochs, batch_size={per_device_train_batch_size}, lr={learning_rate}")
    drona_add_note(f"Accelerator: Graphcore IPU ({ipu_type})")

    return ""


def generate_data_loading_section(dataset_id, dataset_source, hf_token_code):
    """Generate data loading section for IPU."""

    return f"""
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("{dataset_id}"{hf_token_code})

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


def generate_dataset_processing_section(input_column, target_column, max_samples, test_size):
    """Generate dataset processing section with tokenization for IPU."""

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

    # Handle column detection
    # Check if columns are actually specified (not empty or placeholders like "$input_column")
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

    test_size_val = test_size if test_size else 0.1

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


def generate_ipu_config_section(ipu_type):
    """Generate IPU-specific configuration based on IPU type."""

    # IPU configuration varies by type
    if ipu_type == "mk2_pod16":
        return """
    # Configure for POD16 (16 IPUs)
    ipu_config.replication_factor = 4
    ipu_config.gradient_accumulation_steps = 8
"""
    elif ipu_type == "mk2_pod4":
        return """
    # Configure for POD4 (4 IPUs)
    ipu_config.replication_factor = 2
    ipu_config.gradient_accumulation_steps = 4
"""
    else:  # mk2_pod1 or default
        return """
    # Configure for single IPU or POD1
    ipu_config.gradient_accumulation_steps = 16
"""


def generate_custom_training_args_ipu(custom_params):
    """Generate extra IPU training arguments from custom parameters."""

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

    if extra_args:
        return ",\n        " + ",\n        ".join(extra_args)
    else:
        return ""
