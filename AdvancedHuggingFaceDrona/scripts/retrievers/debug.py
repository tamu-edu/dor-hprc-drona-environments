#!/usr/bin/env python3

import json

import requests

import sys



def debug_model_api(model_id):

    """Debug the actual structure of HuggingFace API response."""

    if not model_id:

        print("Usage: python debug.py <model_id>")

        print("Example: python debug.py microsoft/DialoGPT-medium")

        return

    

    try:

        # Make API request to HuggingFace Hub

        url = f"https://huggingface.co/api/models/{model_id}"

        headers = {"User-Agent": "HF-Model-Info-Debug-Script"}

        

        print(f"🔍 Fetching model info for: {model_id}")

        print(f"📡 URL: {url}")

        print("=" * 60)

        

        response = requests.get(url, headers=headers, timeout=10)

        response.raise_for_status()

        

        model_data = response.json()

        

        print("📋 FULL API RESPONSE STRUCTURE:")

        print("=" * 60)

        print(json.dumps(model_data, indent=2))

        

        print("\n" + "=" * 60)

        print("🔧 ANALYSIS:")

        print("=" * 60)

        

        # Check top-level keys

        print(f"📊 Top-level keys: {list(model_data.keys())}")

        

        # Check config structure if it exists

        if 'config' in model_data:

            print(f"\n🎛️  Config exists: {type(model_data['config'])}")

            if model_data['config']:

                print(f"🎛️  Config keys: {list(model_data['config'].keys())}")

                print(f"🎛️  Config content preview:")

                print(json.dumps(model_data['config'], indent=2)[:500] + "..." if len(str(model_data['config'])) > 500 else json.dumps(model_data['config'], indent=2))

            else:

                print("🎛️  Config is empty/null")

        else:

            print("❌ No 'config' key found")

        

        # Check siblings structure if it exists

        if 'siblings' in model_data:

            print(f"\n📁 Siblings exists: {type(model_data['siblings'])}")

            if model_data['siblings']:

                print(f"📁 Number of files: {len(model_data['siblings'])}")

                print(f"📁 First file structure:")

                if len(model_data['siblings']) > 0:

                    print(json.dumps(model_data['siblings'][0], indent=2))

                

                # Calculate total size

                total_size = sum(file.get('size', 0) for file in model_data['siblings'] if file.get('size'))

                print(f"📁 Total size: {total_size} bytes ({total_size / (1024**3):.2f} GB)")

            else:

                print("📁 Siblings is empty")

        else:

            print("❌ No 'siblings' key found")

        

        # Check tags

        if 'tags' in model_data:

            print(f"\n🏷️  Tags: {model_data['tags']}")

        else:

            print("❌ No 'tags' key found")

        

        # Check other potentially useful fields

        other_keys = [key for key in model_data.keys() if key not in ['config', 'siblings', 'tags']]

        if other_keys:

            print(f"\n🔍 Other available keys: {other_keys}")

            for key in other_keys[:5]:  # Show first 5 other keys

                value = model_data[key]

                if isinstance(value, (str, int, float, bool)):

                    print(f"   {key}: {value}")

                elif isinstance(value, (list, dict)):

                    print(f"   {key}: {type(value)} with {len(value)} items")

                else:

                    print(f"   {key}: {type(value)}")

        

        return model_data

        

    except requests.exceptions.RequestException as e:

        print(f"❌ Network error: {e}")

        return None

    except json.JSONDecodeError as e:

        print(f"❌ JSON parsing error: {e}")

        print(f"📄 Raw response: {response.text[:1000]}")

        return None

    except Exception as e:

        print(f"❌ Unexpected error: {e}")

        return None



if __name__ == "__main__":

    if len(sys.argv) > 1:

        model_id = sys.argv[1]

    else:

        # Test with a few popular models

        test_models = [

            "microsoft/DialoGPT-medium",

            "gpt2", 

            "bert-base-uncased",

            "microsoft/CodeBERT-base",

            "facebook/opt-350m"

        ]

        

        print("🧪 Testing with popular models:")

        print("=" * 60)

        

        for model in test_models:

            print(f"\n🎯 Testing: {model}")

            print("-" * 40)

            result = debug_model_api(model)

            if result:

                print("✅ Success!")

                break

            else:

                print("❌ Failed!")

                continue

        else:

            print("\n❌ All test models failed!")

            

        print("\n" + "=" * 60)

        print("💡 Try running with a specific model:")

        print("   python debug.py microsoft/DialoGPT-medium")


    

    debug_model_api(model_id)
