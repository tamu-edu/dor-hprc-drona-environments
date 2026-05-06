#!/usr/bin/env python3

from drona_utils import drona_add_mapping, drona_add_additional_file, drona_add_note, drona_add_error


def setup_inference_ipu(operation, model_id="", dataset_id="", dataset_source="", input_text="",
                        max_samples="5", hf_token="", location="", model_source="hub",
                        ipu_type="mk2_pod4", ipu_use_pipeline=True, **kwargs):
    """Setup IPU inference using Optimum Graphcore."""

    drona_add_note("DEBUG: Inside setup_inference_ipu()")
    drona_add_note(f"DEBUG: model_id={model_id}, ipu_type={ipu_type}, ipu_use_pipeline={ipu_use_pipeline}")

    # Validate
    if not model_id:
        drona_add_error("Model ID required")
        return ""

    if dataset_source not in ["hub", "downloaded", "text", "upload"]:
        drona_add_error("Invalid dataset source")
        return ""

    if dataset_source in ["hub", "downloaded"] and not dataset_id:
        drona_add_error("Dataset ID required for dataset source")
        return ""

    if dataset_source == "text" and not input_text:
        drona_add_error("Input text required for text source")
        return ""

    # Add inference file
    drona_add_note("DEBUG: Adding inference_ipu.py file")
    drona_add_additional_file("scripts/inference/inference_ipu.py")
    drona_add_note("DEBUG: File added successfully")

    # Map model ID
    drona_add_mapping("MODEL_ID", model_id)

    # Handle token
    if hf_token and hf_token.strip() and not hf_token.startswith("$"):
        drona_add_mapping("HF_TOKEN_ARG", f', token="{hf_token}"')
        hf_token_code = f', token="{hf_token}"'
    else:
        drona_add_mapping("HF_TOKEN_ARG", "")
        hf_token_code = ""

    # Conditional dataset import
    if dataset_source in ["hub", "downloaded"]:
        drona_add_mapping("DATASET_IMPORT", "\nfrom datasets import load_dataset")
    else:
        drona_add_mapping("DATASET_IMPORT", "")

    # Always use Pipeline API (recommended and working)
    drona_add_mapping("PIPELINE_IMPORT", "\nfrom optimum.graphcore import pipeline")

    # Generate data loading section
    data_loading = generate_data_loading_section(dataset_id, dataset_source, input_text, max_samples, hf_token_code)

    # Generate IPU inference code using Pipeline API
    ipu_code = generate_pipeline_inference(model_id, hf_token_code, data_loading)

    drona_add_mapping("IPU_INFERENCE_CODE", ipu_code)

    # Add notes
    drona_add_note(f"Model: {model_id} (source: {model_source})")
    if dataset_source in ["hub", "downloaded"]:
        drona_add_note(f"Dataset: {dataset_id} ({dataset_source}, max {max_samples} samples)")
    elif dataset_source == "text":
        preview = input_text[:50] + "..." if len(input_text) > 50 else input_text
        drona_add_note(f"Input text: {preview}")

    drona_add_note(f"Accelerator: Graphcore IPU ({ipu_type})")
    drona_add_note(f"API Mode: Pipeline API (recommended)")

    return ""


def generate_data_loading_section(dataset_id, dataset_source, input_text, max_samples, hf_token_code):
    """Generate the data loading code section."""

    if dataset_source == "text":
        escaped_text = input_text.replace('"', '\\"').replace('\n', '\\n')
        return f'''    # Input text
    samples = ["{escaped_text}"]
'''

    elif dataset_source in ["hub", "downloaded"]:
        max_samples_int = int(max_samples) if max_samples else 5

        token_arg = f"{hf_token_code}" if hf_token_code else ""

        return f'''    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("{dataset_id}"{token_arg})

    # Get the first split available
    split_name = list(dataset.keys())[0]
    data = dataset[split_name]

    # Find text column
    text_columns = [col for col in data.column_names if 'text' in col.lower() or 'content' in col.lower() or 'input' in col.lower()]
    if not text_columns:
        text_columns = [data.column_names[0]]  # Use first column as fallback
    text_col = text_columns[0]

    # Get samples
    num_samples = min({max_samples_int}, len(data))
    samples = [str(data[i][text_col]) for i in range(num_samples)]
    print(f"Loaded {{num_samples}} samples from dataset")
'''

    return ""


def generate_pipeline_inference(model_id, hf_token_code, data_loading):
    """Generate inference code using Pipeline API (simpler)."""

    # Extract token for pipeline
    if hf_token_code:
        token_value = hf_token_code.split('"')[1] if '"' in hf_token_code else ""
        token_arg = f', token="{token_value}"' if token_value else ""
    else:
        token_arg = ""

    code = f'''    # Configure IPU
    ipu_config = IPUConfig()

{data_loading}
    # Create pipeline
    print("Initializing IPU pipeline...")
    generator = pipeline(
        'text-generation',
        model="{model_id}"{token_arg},
        ipu_config=ipu_config
    )

    print(f"Processing {{len(samples)}} samples")

    # Run inference
    results = []
    for i, text in enumerate(samples):
        print(f"Processing sample {{i+1}}/{{len(samples)}}...")

        output = generator(text, max_new_tokens=50, do_sample=True, top_p=0.9)
        output_text = output[0]['generated_text']

        results.append({{
            "input": text,
            "output": output_text
        }})
'''
    return code


def generate_manual_inference(model_id, hf_token_code, data_loading):
    """Generate inference code using Manual API (more control)."""

    code = f'''    # Configure IPU
    print("Setting up IPU configuration...")
    ipu_config = IPUConfig()

    # Load model and tokenizer
    print(f"Loading model: {model_id}")
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained("{model_id}"{hf_token_code})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        "{model_id}",
        ipu_config=ipu_config{hf_token_code}
    )

    print("Model loaded on IPU")

{data_loading}
    print(f"Processing {{len(samples)}} samples")

    # Run inference
    results = []
    for i, text in enumerate(samples):
        print(f"Processing sample {{i+1}}/{{len(samples)}}...")

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            top_p=0.9,
            temperature=0.7
        )

        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results.append({{
            "input": text,
            "output": output_text
        }})
'''
    return code
