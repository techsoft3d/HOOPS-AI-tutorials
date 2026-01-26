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
import json
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

from hoops_ai.storage import DGLGraphStoreHandler
from hoops_ai.ml.EXPERIMENTAL import GraphNodeClassification
import pathlib


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
# Define minimal CAD schema for manufacturing data
builder = SchemaBuilder(
    domain="Manufacturing_Analysis", 
    version="1.0", 
    description="Minimal schema for manufacturing classification"
)

# Manufacturing group - Core manufacturing classification data
label_group = builder.create_group("Labels", "faces", "Label group for ML supervised Tasks")
label_group.create_array("face_labels", ["faces"], "int32", "MFR label category integer (0-24)")

cad_schema = builder.build()
# ============================================================================

# ============================================================================
# LABELS DESCRIPTION - Machining Features Face classification labels
# ============================================================================

labels_description = {
        0: {"name": "no-label", "description": "No label assigned."},
        1: {"name": "rectangular_through_slot", "description": "This is a rectangular MFR feature."},
        2: {"name": "triangular_through_slot", "description": "This is a triangular MFR type, cool right!"},
        3: {"name": "rectangular_passage", "description": "Description for rectangular_passage."},
        4: {"name": "triangular_passage", "description": "Description for triangular_passage."},
        5: {"name": "6sides_passage", "description": "Description for 6sides_passage."},
        6: {"name": "rectangular_through_step", "description": "Description for rectangular_through_step."},
        7: {"name": "2sides_through_step", "description": "Description for 2sides_through_step."},
        8: {"name": "slanted_through_step", "description": "Description for slanted_through_step."},
        9: {"name": "rectangular_blind_step", "description": "Description for rectangular_blind_step."},
        10: {"name": "triangular_blind_step", "description": "Description for triangular_blind_step."},
        11: {"name": "rectangular_blind_slot", "description": "Description for rectangular_blind_slot."},
        12: {"name": "rectangular_pocket", "description": "Description for rectangular_pocket."},
        13: {"name": "triangular_pocket", "description": "Description for triangular_pocket."},
        14: {"name": "6sides_pocket", "description": "Description for 6sides_pocket."},
        15: {"name": "chamfer", "description": "Description for chamfer."},
        16: {"name": "circular through slot", "description": "Description for circular through slot."},
        17: {"name": "through hole", "description": "Description for through hole."},
        18: {"name": "circular blind step", "description": "Description for circular blind step."},
        19: {"name": "horizontal circular end blind slot", "description": "Description for horizontal circular end blind slot."},
        20: {"name": "vertical circular end blind slot", "description": "Description for vertical circular end blind slot."},
        21: {"name": "circular end pocket", "description": "Description for circular end pocket."},
        22: {"name": "o-ring", "description": "Description for o-ring."},
        23: {"name": "blind hole", "description": "Description for blind hole."},
        24: {"name": "fillet", "description": "Description for fillet."}
    }


# Invert the dictionary
description_to_code = {v["name"]: k for k, v in labels_description.items()}
# ============================================================================

@flowtask.extract(
    name="gather cadsynth files",
    inputs=["cad_datasources"],
    outputs=["cad_dataset"],
    parallel_execution=True
)
def gather_cadsynth_files(source: str) -> List[str]:
    """Gather CAD files from source directory - simplified for testing"""

    # Example 1: Basic retrieval with format filtering
    retriever = CADFileRetriever(
        storage_provider=LocalStorageProvider(directory_path=source),
        formats=[".stp", ".step", ".iges", ".igs"],
        #filter_pattern="*5*"  # Only files with "5" in name
    )
            
    # Get files using the library's retriever
    source_files = retriever.get_file_list()
    
    # Shuffle to get random sample instead of first N files in order
    import random
    random.seed(42)  # For reproducibility
    shuffled_files = source_files.copy()
    random.shuffle(shuffled_files)
    
    return shuffled_files #[:100]


## Use the HOOPS AI directly integrated GraphClassification Model

nb_dir = pathlib.Path.cwd()
flows_outputdir = nb_dir.joinpath("out")

def get_flow_name():
    return "ETL_CADSYNTH_training_b2"

flow_name = get_flow_name()
my_workflow_for_cadsynth = GraphNodeClassification(num_classes=25, result_dir= str(pathlib.Path(flows_outputdir).joinpath("flows").joinpath(flow_name)))


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

    facecount, edgecount = my_workflow_for_cadsynth.encode_cad_data(cad_file, cad_loader, storage)
    
    # Add label data
    file_name = pathlib.Path(cad_file).stem  # Get base name without extension
    label_file = pathlib.Path(cad_file).parent.parent / "label" / f"{file_name}.json"
    
    # Load and extract labels from JSON file
    with open(label_file, 'r') as f:
        data = json.load(f)
    label_codes = data.get("labels", [])
    
    
    
    
    
    # Save label data in the schema-defined group for dataset analytics
    storage.save_data("Labels/face_labels", np.array(label_codes))

    
    # ALSO save label using the key expected by GraphClassification.convert_encoded_data_to_graph
    # This is required for the DGL graph files to have the correct labels
    storage.save_data("face_labels", np.array(label_codes))
    
    #
    dgl_storage = DGLGraphStoreHandler()

    # DGL graph Bin file
    item_no_suffix = pathlib.Path(cad_file).with_suffix("")  # Remove the suffix to get the base name
    hash_id = generate_unique_id_from_path(str(item_no_suffix))
    dgl_output_path = pathlib.Path(flows_outputdir).joinpath("flows", flow_name, "dgl", f"{hash_id}.ml")  
    dgl_output_path.parent.mkdir(parents=True, exist_ok=True)

    my_workflow_for_cadsynth.convert_encoded_data_to_graph(storage, dgl_storage, str(dgl_output_path))
    
    # Save file-level metadata (will be routed to .infoset)
    storage.save_metadata("Item", str(cad_file))
    storage.save_metadata("source", "CADSYNTH")
    
    # Compress the storage into a .data file
    storage.compress_store()
    
    # Return the base storage path
    return storage.get_file_path("")

