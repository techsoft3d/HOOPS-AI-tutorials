"""
CAD Processing Tasks for Manufacturing Analysis

This module defines reusable task functions for HOOPS AI workflows.
These functions can be imported into Jupyter notebooks and will work
correctly with ProcessPoolExecutor for parallel execution.

CRITICAL for Windows ProcessPoolExecutor:
1. **License**: Must be set at module level (reads from HOOPS_AI_LICENSE env var)
2. **Schema**: Must be defined at module level (not in notebook)
3. **Tasks**: Must be defined in .py files (not in notebooks)

Why? When worker processes spawn on Windows, they import this module fresh.
Anything set in the notebook (like license or schema) is NOT visible to workers.

Usage in notebooks:
    # Set environment variable BEFORE launching Jupyter:
    # $env:HOOPS_AI_LICENSE = "your-license-key"
    
    from cad_tasks import gather_files, encode_manufacturing_data, cad_schema
    
    # License and schema are already configured in cad_tasks.py!
    cad_flow = hoops_ai.create_flow(
        tasks=[gather_files, encode_manufacturing_data],
        max_workers=4  # Parallel execution now works!
    )
"""

import os
import glob
import random
from typing import List
import numpy as np

import hoops_ai
from hoops_ai.flowmanager import flowtask
from hoops_ai.cadaccess import HOOPSLoader, HOOPSTools
from hoops_ai.cadencoder import BrepEncoder
from hoops_ai.storage import DataStorage, CADFileRetriever, LocalStorageProvider
from hoops_ai.storage.datasetstorage.schema_builder import SchemaBuilder

from hoops_ai.storage.label_storage import LabelStorage
from hoops_ai.storage.helpers import generate_unique_id_from_path

from hoops_ai.storage import DGLGraphStoreHandler, OptStorage
from hoops_ai.storage import JsonStorageHandler


import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from custom_flow_model_graph_classification import CustomGraphClassification

# Import conversion utility
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from opt_to_json import opt_to_json

import pathlib
import torch


# ============================================================================
# LICENSE SETUP - Must be set at module level for ProcessPoolExecutor
# ============================================================================
# CRITICAL: Worker processes need the license configured when they import this module
license_key = os.getenv("HOOPS_AI_LICENSE")
if license_key:
    hoops_ai.set_license(license_key, validate=False)
else:
    print("WARNING: HOOPS_AI_LICENSE environment variable not set in cad_tasks.py")
# ============================================================================


# ============================================================================
# SCHEMA DEFINITION - Must be defined at module level for ProcessPoolExecutor
# ============================================================================

# Configuration: Set to True to load schema from file, False to build in code
LOAD_SCHEMA_FROM_FILE = True
SCHEMA_FILE_PATH = os.path.join(os.path.dirname(__file__), "manufacturing_schema.json")

def build_schema():
    """Build the CAD schema programmatically"""
    builder = SchemaBuilder(
        domain="Manufacturing_Analysis", 
        version="1.0", 
        description="Minimal schema for manufacturing classification"
    )

    # Manufacturing group - Core manufacturing classification data
    label_group = builder.create_group("Labels", "part", "Label group for ML supervised Tasks")
    label_group.create_array("task_A", ["part"], "int32", "Original A part category (45 classes, indexed 0-44)")
    label_group.create_array("task_B", ["part"], "int32", "Original B part category (5 groups, indexed 0-44)")
    label_group.create_array("task_C", ["part"], "int32", "Simplified C part category (45 classes, indexed 0-4)")
    label_group.create_array("task_D", ["part"], "int32", "Simplified D part category (5 groups, indexed 0-4)")

    # Define metadata routing
    builder.define_categorical_metadata('task_A_description', 'str', 'Original A detailed part category name')
    builder.define_categorical_metadata('task_B_description', 'str', 'Original B part category group name')
    builder.define_categorical_metadata('task_C_description', 'str', 'Simplified C detailed part category name')
    builder.define_categorical_metadata('task_D_description', 'str', 'Simplified D part category group name')
    builder.set_metadata_routing_rules(
        categorical_patterns=['task_A_description', 'task_B_description', 'category', 'type']
    )

    return builder.build()

def export_schema(schema, file_path):
    """Export schema to a JSON file"""
    import json
    # Schema object is a dictionary, so we can directly save it as JSON
    with open(file_path, 'w') as f:
        json.dump(schema, f, indent=2)
    print(f"Schema exported to: {file_path}")

def load_schema(file_path):
    """Load schema from a JSON file"""
    import json
    with open(file_path, 'r') as f:
        schema = json.load(f)
    print(f"Schema loaded from: {file_path}")
    return schema

# Build or load schema based on configuration
if LOAD_SCHEMA_FROM_FILE and os.path.exists(SCHEMA_FILE_PATH):
    cad_schema = load_schema(SCHEMA_FILE_PATH)
else:
    cad_schema = build_schema()
    # Export schema for future use (only on first build)
    if not LOAD_SCHEMA_FROM_FILE:
        try:
            print(f"Note: Export schema to location: {SCHEMA_FILE_PATH}")
            export_schema(cad_schema, SCHEMA_FILE_PATH)
        except Exception as e:
            print(f"Note: Could not export schema: {e}")

# ============================================================================

# ============================================================================
# LABELS DESCRIPTION - Part classification labels
# ============================================================================
labels_description = {
        0: {"name": "Bearings"              , "description": " fabewave dataset sample  "},
        1: {"name": "Bolts"                 , "description": " fabewave dataset sample  "},
        2: {"name": "Brackets"              , "description": " fabewave dataset sample  "},
        3: {"name": "Bushing"               , "description": " fabewave dataset sample  "},
        4: {"name": "Bushing_Damping_Liners", "description": " fabewave dataset sample  "},
        5: {"name": "Collets"               , "description": " fabewave dataset sample  "},
        6: {"name": "Gasket"                , "description": " fabewave dataset sample  "},
        7: {"name": "Grommets"              , "description": " fabewave dataset sample  "},
        8: {"name": "HeadlessScrews"        , "description": " fabewave dataset sample  "},
        9: {"name": "Hex_Head_Screws"       , "description": " fabewave dataset sample  "},
        10: {"name": "Keyway_Shaft"         , "description": " fabewave dataset sample  "},
        11: {"name": "Machine_Key"          , "description": " fabewave dataset sample  "},
        12: {"name": "Nuts"                 , "description": " fabewave dataset sample  "},
        13: {"name": "O_Rings"              , "description": " fabewave dataset sample  "},
        14: {"name": "Thumb_Screws"        , "description": " fabewave dataset sample   "},
        15: {"name": "Pipe_Fittings"        , "description": " fabewave dataset sample   "},
        16: {"name": "Pipe_Joints"              , "description": " fabewave dataset sample  "},
        17: {"name": "Pipes"                 , "description": " fabewave dataset sample  "},
        18: {"name": "Rollers"              , "description": " fabewave dataset sample  "},
        19: {"name": "Rotary_Shaft"               , "description": " fabewave dataset sample  "},
        20: {"name": "Shaft_Collar"         , "description": " fabewave dataset sample  "},
        21: {"name": "Slotted_Flat_Head_Screws"               , "description": " fabewave dataset sample  "},
        22: {"name": "Socket_Head_Screws"               , "description": " fabewave dataset sample  "},
        23: {"name": "Washers"                , "description": " fabewave dataset sample  "},
        24: {"name": "Boxes"              , "description": " fabewave dataset sample  "},
        25: {"name": "Cotter_Pin"        , "description": " fabewave dataset sample  "},
        26: {"name": "External Retaining Rings"       , "description": " fabewave dataset sample  "},
        27: {"name": "Eyesbolts With Shoulders"         , "description": " fabewave dataset sample  "},
        28: {"name": "Fixed Cap Flange"          , "description": " fabewave dataset sample  "},
        29: {"name": "Gear Rod Stock"                 , "description": " fabewave dataset sample  "},
        30: {"name": "Gears"              , "description": " fabewave dataset sample  "},
        31: {"name": "Holebolts With Shoulders"        , "description": " fabewave dataset sample   "},
        32: {"name": "Idler Sprocket"        , "description": " fabewave dataset sample   "},
        33: {"name": "Miter Gear Set Screw"        , "description": " fabewave dataset sample   "},
        34: {"name": "Miter Gears"        , "description": " fabewave dataset sample   "},
        35: {"name": "Rectangular Gear Rack"        , "description": " fabewave dataset sample   "},
        36: {"name": "Routing EyeBolts Bent Closed Eye"        , "description": " fabewave dataset sample   "},
        37: {"name": "Sleeve Washers"        , "description": " fabewave dataset sample   "},
        38: {"name": "Socket-Connect Flanges"        , "description": " fabewave dataset sample   "},
        39: {"name": "Sprocket Taper-Lock Bushing"        , "description": " fabewave dataset sample   "},
        40: {"name": "Strut Channel Floor Mount"        , "description": " fabewave dataset sample   "},
        41: {"name": "Strut Channel Side-Side"        , "description": " fabewave dataset sample   "},
        42: {"name": "Tag Holder"        , "description": " fabewave dataset sample   "},
        43: {"name": "Webbing Guide"        , "description": " fabewave dataset sample   "},
        44: {"name": "Wide Grip External Retaining Ring"        , "description": " fabewave dataset sample   "},
    }

# Invert the dictionary
description_to_code = {v["name"]: k for k, v in labels_description.items()}
# ============================================================================

# second labeling but group instead of individual type:

# Simplified 5-group classification
# Simplified 5-group classification
label_to_simplified = {
    # Group 0: Fasteners (12 categories)
    1: 0, 8: 0, 9: 0, 11: 0, 12: 0, 14: 0, 21: 0, 22: 0, 25: 0, 27: 0, 31: 0, 36: 0,
    # Group 1: Seals & Damping (7 categories)
    3: 1, 4: 1, 6: 1, 7: 1, 13: 1, 23: 1, 37: 1,
    # Group 2: Bearings & Shafts (8 categories)
    0: 2, 5: 2, 10: 2, 18: 2, 19: 2, 20: 2, 26: 2, 44: 2,
    # Group 3: Gears & Transmission (7 categories)
    29: 3, 30: 3, 32: 3, 33: 3, 34: 3, 35: 3, 39: 3,
    # Group 4: Structural & Piping (11 categories)
    2: 4, 15: 4, 16: 4, 17: 4, 24: 4, 28: 4, 38: 4, 40: 4, 41: 4, 42: 4, 43: 4
}

simplified_groups = {
    0: "Fasteners", 1: "Seals_Damping", 2: "Bearings_Shafts", 
    3: "Gears_Transmission", 4: "Structural_Piping"
}

#print(f"Simplified mapping: 45 → 5 categories (indexed 0-4)")



@flowtask.extract(
    name="gather fabwave files",
    inputs=["cad_datasources"],
    outputs=["cad_dataset"],
    parallel_execution=True
)
def gather_fabwave_files(source: str) -> List[str]:
    """Gather CAD files ensuring at least 3 samples from each of the 45 original categories"""
    
    import random
    from collections import defaultdict
    
    # Retrieve all available files
    retriever = CADFileRetriever(
        storage_provider=LocalStorageProvider(directory_path=source),
        formats=[".stp", ".step", ".iges", ".igs"],
    )
    
    source_files = retriever.get_file_list()
    
    # Group files by their original category (45 classes)
    files_by_category = defaultdict(list)
    
    for file_path in source_files:
        folder_name = str(pathlib.Path(file_path).parent.parent.stem)
        label_code = description_to_code.get(folder_name)
        if label_code is not None:
            files_by_category[label_code].append(file_path)
    
    # Sample at least 3 files from each of the 45 categories
    random.seed(42)
    sampled_files = []
    remaining_files = []
    
    min_samples_per_category = 3
    categories_found = len(files_by_category)
    
    for label_code in range(45):  # Ensure all 45 categories are checked
        category_files = files_by_category.get(label_code, [])
        if category_files:
            # Take minimum samples for guaranteed representation
            sample_size = min(min_samples_per_category, len(category_files))
            samples = random.sample(category_files, sample_size)
            sampled_files.extend(samples)
            
            # Add remaining files to pool for random sampling
            remaining = [f for f in category_files if f not in samples]
            remaining_files.extend(remaining)
        else:
            category_name = labels_description.get(label_code, {}).get("name", f"Category {label_code}")
            print(f"WARNING: No files found for category {label_code} ({category_name})")
    
    # Fill up to target size (i.e 1000) with random samples from remaining files
    target_size = 200
    already_sampled = len(sampled_files)
    additional_needed = max(0, target_size - already_sampled)
    
    if additional_needed > 0 and remaining_files:
        random.shuffle(remaining_files)
        sampled_files.extend(remaining_files[:additional_needed])
    
    # Final shuffle for randomness
    random.shuffle(sampled_files)
    
    print(f"Sampled {len(sampled_files)} files from {categories_found} categories")
    print(f"Guaranteed: {min_samples_per_category} samples per category")
    
    return sampled_files


## Use the HOOPS AI directly integrated GraphClassification Model

nb_dir = pathlib.Path.cwd()
flows_outputdir = nb_dir.joinpath("out")

def get_flow_name():
    return "ETL_Multi_Y_Part_Classification"

flow_name = get_flow_name()
custom_graph_classification = CustomGraphClassification(num_classes=45, result_dir= str(pathlib.Path(flows_outputdir).joinpath("flows").joinpath(flow_name)))


# ============================================================================
@flowtask.transform(
    name="Preparing data for Exploring and ML training",
    inputs=["cad_dataset"],
    outputs=["cad_files_encoded"],
    parallel_execution=True
)
def encode_data_for_ml_training(cad_file: str, cad_loader :  HOOPSLoader, storage : DataStorage) -> str:
    """Logic to prepare data for exploring and machine learning training - Part Classification problem
    """
    import numpy as np
    import random

    cad_model = cad_loader.create_from_file(cad_file)
    storage.set_schema(cad_schema)

    facecount, edgecount = custom_graph_classification.encode_cad_data(cad_file, cad_loader, storage)
    
    # Add label data
    folder_with_name = str(pathlib.Path(cad_file).parent.parent.stem)
    label_code = int(description_to_code.get(folder_with_name, None))
    
    # Validate label_code - skip if unknown category
    if label_code is None:
        raise ValueError(f"Unknown category '{folder_with_name}' for file {cad_file}. Category not found in labels_description.")
    
    label_description = [{int(label_code) : labels_description[label_code]["name"]} ]
    
    # Compute simplified label using the mapping
    simplified_label = int(label_to_simplified.get(label_code, None))
    if simplified_label is None:
        raise ValueError(f"Label code {label_code} not found in label_to_simplified mapping for file {cad_file}.")
    
    simplified_label_name = simplified_groups[simplified_label]
    
    # Save label data in the schema-defined group for dataset analytics
    storage.save_metadata("task_A_description", folder_with_name)
    storage.save_metadata("task_B_description", folder_with_name)
    storage.save_metadata("task_C_description", simplified_label_name)
    storage.save_metadata("task_D_description", simplified_label_name)
    
    # ALSO save label using the key expected by GraphClassification.convert_encoded_data_to_graph
    # This is required for the DGL graph files to have the correct labels
    storage.save_data(LabelStorage.GRAPH_CADENTITY, np.array([label_code]))
    #storage.save_data("Labels/part_label", np.array([label_code]))
    
    ## EXTRA data that we will use also for training
    # task_A and task_B: Both use original 45 classes (0-44) - should produce identical results
    storage.save_data("Labels/task_A", np.array([label_code]))
    storage.save_data("Labels/task_B", np.array([label_code]))
    # task_C and task_D: Both use simplified 5 groups (0-4) - should produce identical results
    storage.save_data("Labels/task_C", np.array([simplified_label]))
    storage.save_data("Labels/task_D", np.array([simplified_label]))

    
    
    #my_workflow_for_fabewave.encode_label_data()
    dgl_storage = DGLGraphStoreHandler()

    # DGL graph Bin file
    item_no_suffix = pathlib.Path(cad_file).with_suffix("")  # Remove the suffix to get the base name
    hash_id = generate_unique_id_from_path(str(item_no_suffix))
    dgl_output_path = pathlib.Path(flows_outputdir).joinpath("flows", flow_name, "dgl", f"{hash_id}.ml")  
    dgl_output_path.parent.mkdir(parents=True, exist_ok=True)

    
    #dgl_storage.append_extra_data(storage.load_data("Labels/part_label"), feature_name="part_label", torch_type=torch.long)
    dgl_storage.append_extra_data(storage.load_data("Labels/task_A"), feature_name="task_A", torch_type=torch.long)
    dgl_storage.append_extra_data(storage.load_data("Labels/task_B"), feature_name="task_B", torch_type=torch.long)
    dgl_storage.append_extra_data(storage.load_data("Labels/task_C"), feature_name="task_C", torch_type=torch.long)
    dgl_storage.append_extra_data(storage.load_data("Labels/task_D"), feature_name="task_D", torch_type=torch.long)

    
    custom_graph_classification.convert_encoded_data_to_graph(storage, dgl_storage, str(dgl_output_path))
    
    # Save file-level metadata (will be routed to .infoset)
    storage.save_metadata("Item", str(cad_file))
    storage.save_metadata("source", "FABWAVE")
    
    # Compress the storage into a .data file
    storage.compress_store()
    
    # Convert OptStorage to JSON in parallel directory structure
    opt_path = pathlib.Path(storage.get_file_path(""))
    json_path = opt_path.parent.parent / "data_mining_json" / opt_path.name
    opt_to_json(storage, str(json_path))
    
    # Return the base storage path
    return storage.get_file_path("")




