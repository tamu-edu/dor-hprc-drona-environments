import os
import yaml
import json
import time

def get_directories(path):
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

def update_metadata():
    for cluster in get_directories("./"):
        if cluster.startswith("."): continue
        
        metadata = {}
        for env in get_directories(f"./{cluster}"):
            manifest_path = f"{cluster}/{env}/manifest.yml"
            if os.path.exists(manifest_path):
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
