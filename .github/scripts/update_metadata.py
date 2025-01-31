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
                        "size": get_dir_size(f"{cluster}/{env}")
                    }
        
        with open(f"{cluster}/metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
