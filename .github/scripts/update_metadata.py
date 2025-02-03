import os
import yaml
import json
import time

def get_directories(path):
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

def create_default_manifest(env_name, env_path):
    """Create a default manifest.yml if it doesn't exist"""
    manifest = {
        "name": env_name,
        "description": f"Environment for {env_name}",
        "category": "Uncategorized",
        "version": "1.0.0",
        "author": ""
    }
    
    manifest_path = os.path.join(env_path, "manifest.yml")
    with open(manifest_path, 'w') as f:
        yaml.safe_dump(manifest, f, default_flow_style=False)

def update_metadata():
    for cluster in get_directories("./"):
        if cluster.startswith("."): 
            continue
            
        metadata = {}
        for env in get_directories(f"./{cluster}"):
            env_path = f"{cluster}/{env}"
            manifest_path = f"{env_path}/manifest.yml"
            
            # Create manifest if it doesn't exist
            if not os.path.exists(manifest_path):
                create_default_manifest(env, env_path)
            
            with open(manifest_path) as f:
                manifest = yaml.safe_load(f)
                metadata[env] = {
                    **manifest,
                    "last_updated": time.strftime("%Y-%m-%d"),
                }
                
        with open(f"{cluster}/metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    update_metadata()
