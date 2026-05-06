#!/usr/bin/env python3

import os
import sys
import json
import argparse
from pathlib import Path
import traceback

def run_inference(model_id, dataset_id=None, input_text=None, max_samples=5, token=None, 
                  gpu_type="auto", num_gpus=1, precision="auto"):
    """
    Run simple inference with a model on dataset samples or input text.
    """
    print("=" * 60)
    print(f"🚀 STARTING HUGGING FACE INFERENCE")
    print(f"📦 Model: {model_id}")
    print("=" * 60)

    # Create results directory
    os.makedirs("results", exist_ok=True)

    try:
        # Import libraries and setup
        import transformers
        import torch
        from transformers import pipeline
        
        # Print system information
        print(f"🔧 PyTorch version: {torch.__version__}")
        print(f"🔧 Transformers version: {transformers.__version__}")
        print(f"💾 CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"🎯 CUDA version: {torch.version.cuda}")
            print(f"💡 Available GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory // 1024**3}GB)")
        print("-" * 60)
        
        # Set up environment
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if token:
            os.environ["HF_TOKEN"] = token
        
        # Determine model type from model_id
        model_type = "text-generation"  # Default
        if "bert" in model_id.lower() or "roberta" in model_id.lower() or "distilbert" in model_id.lower():
            model_type = "text-classification"
        elif "t5" in model_id.lower() or "bart" in model_id.lower():
            model_type = "summarization"
        
        print(f"Using pipeline: {model_type}")
        
        # Prepare results
        results = []
        
[MODEL_CONFIG_CODE]

[DATASET_PROCESSING_CODE]



[DRIVER]

[MAPPING]

[INPUTTEXTPROCESSINGCODE]
        
        # If no input provided, error
        if not results:
            print("❌ No input provided! Either dataset_id or input_text must be specified.")
            return False
        
        # Save results
        print("💾 Saving results...")
        with open("results/inference_results.json", "w") as f:
            # Use a simple serialization that handles non-serializable objects
            serializable_results = []
            for result in results:
                serializable_result = {
                    "input": result["input"]
                }
                
                # Handle different output formats
                if isinstance(result["output"], list):
                    if all(isinstance(item, dict) for item in result["output"]):
                        serializable_result["output"] = [
                            {k: str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v 
                             for k, v in item.items()}
                            for item in result["output"]
                        ]
                    else:
                        serializable_result["output"] = [str(item) for item in result["output"]]
                else:
                    serializable_result["output"] = str(result["output"])
                
                serializable_results.append(serializable_result)
            
            json.dump(serializable_results, f, indent=2)
        
        # Create a more human-readable version
        with open("results/inference_results.txt", "w") as f:
            for i, result in enumerate(results):
                f.write(f"=== Sample {i+1} ===\n")
                f.write(f"Input: {result['input']}\n\n")
                
                if isinstance(result["output"], list):
                    if model_type == "text-generation" and all(isinstance(item, dict) for item in result["output"]):
                        for j, gen in enumerate(result["output"]):
                            if j == 0 and "generated_text" in gen:
                                print("The first generated output:")
                                print(gen['generated_text'])

                            if "generated_text" in gen:
                                f.write(f"Generated Text {j+1}:\n{gen['generated_text']}\n\n")
                            else:
                                f.write(f"Output {j+1}: {str(gen)}\n\n")
                    else:
                        for j, item in enumerate(result["output"]):
                            f.write(f"Output {j+1}: {str(item)}\n")
                else:
                    f.write(f"Output: {str(result['output'])}\n")
                
                f.write("\n" + "="*50 + "\n\n")
        
        print("=" * 60)
        print("🎉 INFERENCE COMPLETE!")
        print(f"📁 Results saved to: results/inference_results.json and results/inference_results.txt")
        if torch.cuda.is_available() and gpu_type != "none":
            print(f"🔥 Final GPU memory usage: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")
        print("=" * 60)
        return True
    
    except ImportError as e:
        print(f"Error importing required libraries: {str(e)}")
        print("Please make sure transformers and datasets are installed.")
        return False
    
    except Exception as e:
        print(f"Error running inference: {str(e)}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run simple inference with a Hugging Face model")
    parser.add_argument("--model_id", required=True, help="Model ID on Hugging Face")
    parser.add_argument("--dataset_id", help="Dataset ID on Hugging Face")
    parser.add_argument("--input_text", help="Input text for inference")
    parser.add_argument("--max_samples", default=5, help="Maximum number of samples to process")
    parser.add_argument("--token", help="Hugging Face token for gated models")
    parser.add_argument("--gpu_type", default="auto", help="GPU type (auto, none, h100, a100, etc.)")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs for multi-gpu mode")
    parser.add_argument("--precision", default="auto", help="Model precision (auto, float32, float16, etc.)")
    
    args = parser.parse_args()
    
    if not args.dataset_id and not args.input_text:
        print("Error: Either dataset_id or input_text must be provided")
        sys.exit(1)
    
    success = run_inference(
        args.model_id, 
        args.dataset_id, 
        args.input_text, 
        args.max_samples, 
        args.token,
        args.gpu_type,
        args.num_gpus,
        args.precision
    )
    
    sys.exit(0 if success else 1)
