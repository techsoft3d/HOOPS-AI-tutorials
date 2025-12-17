"""Minimal OptStorage to JSON converter."""

import json
from pathlib import Path
from hoops_ai.storage import OptStorage, JsonStorageHandler


def opt_to_json(opt_storage: OptStorage, json_path: str) -> None:
    """
    Convert OptStorage to JsonStorageHandler.
    
    Args:
        opt_storage: OptStorage instance with data
        json_path: Path for JSON output directory
        
    Raises:
        RuntimeError: If no data found or conversion fails
    """
    # Get keys from OptStorage
    keys = opt_storage.get_keys()
    if not keys:
        raise RuntimeError("No data keys found in OptStorage")
    
    # Create JSON storage
    json_storage = JsonStorageHandler(json_path)
    
    try:
        # Copy metadata if available
        opt_path = Path(opt_storage.store_path)
        metadata_file = opt_path / "metadata.json"
        
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
            
            Path(json_path).mkdir(parents=True, exist_ok=True)
            with open(Path(json_path) / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=4)
        
        # Convert data
        for key in keys:
            data = opt_storage.load_data(key)
            json_storage.save_data(key, data)
        
        json_storage.save_metadata("stored_keys", keys)
        
    finally:
        json_storage.close()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python opt_to_json.py <opt_path> <json_path>")
        sys.exit(1)
    
    opt_storage = OptStorage(sys.argv[1])
    opt_to_json(opt_storage, sys.argv[2])
    opt_storage.close()
