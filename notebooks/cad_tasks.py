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
from hoops_ai.storage import DataStorage
from hoops_ai.storage.datasetstorage.schema_builder import SchemaBuilder


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
machining_group = builder.create_group("machining", "part", "Manufacturing and machining classification data")
machining_group.create_array("machining_category", ["part"], "int32", "Machining complexity category (1-5)")
machining_group.create_array("material_type", ["part"], "int32", "Material type (1-5)")
machining_group.create_array("estimated_machining_time", ["part"], "float32", "Estimated machining time in hours")

# Define metadata routing
builder.define_categorical_metadata('material_type_description', 'str', 'Material classification')
builder.set_metadata_routing_rules(
    categorical_patterns=['material_type_description', 'category', 'type']
)

cad_schema = builder.build()
# ============================================================================


@flowtask.extract(
    name="gather cad files",
    inputs=["cad_datasources"],
    outputs=["cad_dataset"],
    parallel_execution=True
)
def gather_files(source: str) -> List[str]:
    """Custom implementation of Data ingestion
    """
    # Use simple glob pattern matching for ProcessPoolExecutor compatibility
    patterns = ["*.stp", "*.step", "*.iges", "*.igs"]
    source_files = []
    
    for pattern in patterns:
        search_path = os.path.join(source, pattern)
        files = glob.glob(search_path)
        source_files.extend(files)
    
    print(f"Found {len(source_files)} CAD files in {source}")
    return source_files


@flowtask.transform(
    name="Manufacturing data encoding",
    inputs=["cad_dataset"],
    outputs=["cad_files_encoded"],
    parallel_execution=True
)
def encode_manufacturing_data(cad_file: str, cad_loader: HOOPSLoader, storage: DataStorage) -> str:
    """custom implementation of a flowtask.transform
    """
    # Load CAD model using the process-local HOOPSLoader
    cad_model = cad_loader.create_from_file(cad_file)
    
    # Set the schema for structured data organization
    # Schema is defined at module level, so it's available in all worker processes
    storage.set_schema(cad_schema)
    
    # Prepare BREP for feature extraction
    hoopstools = HOOPSTools()
    hoopstools.adapt_brep(cad_model, None)
    
    # Extract geometric features using BrepEncoder
    brep_encoder = BrepEncoder(cad_model.get_brep(), storage)
    brep_encoder.push_face_indices()
    brep_encoder.push_face_attributes()
    
    # Generate manufacturing classification data
    file_basename = os.path.basename(cad_file)
    file_name = os.path.splitext(file_basename)[0]
    
    # Set seed for reproducible results based on filename
    random.seed(hash(file_basename) % 1000)
    
    # Generate classification values
    machining_category = random.randint(1, 5)
    material_type = random.randint(1, 5)
    estimated_time = random.uniform(0.5, 10.0)
    
    # Material type descriptions
    material_descriptions = ["Steel", "Aluminum", "Titanium", "Plastic", "Composite"]
    
    # Save data using the OptStorage API (data_key format: "group/array_name")
    storage.save_data("machining/machining_category", np.array([machining_category], dtype=np.int32))
    storage.save_data("machining/material_type", np.array([material_type], dtype=np.int32))
    storage.save_data("machining/estimated_machining_time", np.array([estimated_time], dtype=np.float32))
    
    # Save categorical metadata (will be routed to .attribset)
    storage.save_metadata("material_type_description", material_descriptions[material_type - 1])
    
    # Save file-level metadata (will be routed to .infoset)
    storage.save_metadata("Item", str(cad_file))
    storage.save_metadata("Flow name", "minimal_manufacturing_flow")
    
    # Compress the storage into a .data file
    storage.compress_store()

    
    return storage.get_file_path("")
