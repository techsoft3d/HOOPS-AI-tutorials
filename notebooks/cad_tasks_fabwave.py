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

from hoops_ai.storage import DGLGraphStoreHandler
from hoops_ai.ml.EXPERIMENTAL import GraphClassification
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
label_group = builder.create_group("Labels", "part", "Label group for ML supervised Tasks")
label_group.create_array("part_label", ["part"], "int32", "Part category integer (1-26)")
label_group.create_array("part_source", ["part"], "str", "Part category integer (1-26)")

# Define metadata routing
builder.define_categorical_metadata('part_label_description', 'str', 'String value of the integer label Part Category')
builder.set_metadata_routing_rules(
    categorical_patterns=['part_label_description', 'category', 'type']
)

cad_schema = builder.build()
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

@flowtask.extract(
    name="gather fabwave files",
    inputs=["cad_datasources"],
    outputs=["cad_dataset"],
    parallel_execution=True
)
def gather_fabwave_files(source: str) -> List[str]:
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
    return "ETL_Fabwave_training_b2"

flow_name = get_flow_name()
my_workflow_for_fabewave = GraphClassification(num_classes=45, result_dir= str(pathlib.Path(flows_outputdir).joinpath("flows").joinpath(flow_name)))


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

    facecount, edgecount = my_workflow_for_fabewave.encode_cad_data(cad_file, cad_loader, storage)
    
    # Add label data
    folder_with_name = str(pathlib.Path(cad_file).parent.parent.stem)
    label_code = description_to_code.get(folder_with_name, None)
    
    # Validate label_code - skip if unknown category
    if label_code is None:
        raise ValueError(f"Unknown category '{folder_with_name}' for file {cad_file}. Category not found in labels_description.")
    
    label_description = [{int(label_code) : labels_description[label_code]["name"]} ]
    
    # Save label data in the schema-defined group for dataset analytics
    storage.save_data("Labels/part_label", np.array([label_code]))
    storage.save_metadata("part_label_description", folder_with_name)
    
    # ALSO save label using the key expected by GraphClassification.convert_encoded_data_to_graph
    # This is required for the DGL graph files to have the correct labels
    storage.save_data(LabelStorage.GRAPH_CADENTITY, np.array([label_code]))
    
    #my_workflow_for_fabewave.encode_label_data()
    dgl_storage = DGLGraphStoreHandler()

    # DGL graph Bin file
    item_no_suffix = pathlib.Path(cad_file).with_suffix("")  # Remove the suffix to get the base name
    hash_id = generate_unique_id_from_path(str(item_no_suffix))
    dgl_output_path = pathlib.Path(flows_outputdir).joinpath("flows", flow_name, "dgl", f"{hash_id}.ml")  
    dgl_output_path.parent.mkdir(parents=True, exist_ok=True)

    my_workflow_for_fabewave.convert_encoded_data_to_graph(storage, dgl_storage, str(dgl_output_path))
    
    # Save file-level metadata (will be routed to .infoset)
    storage.save_metadata("Item", str(cad_file))
    storage.save_metadata("source", "FABWAVE")
    
    # Compress the storage into a .data file
    storage.compress_store()
    
    # Return the base storage path
    return storage.get_file_path("")

