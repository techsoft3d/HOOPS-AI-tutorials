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
from hoops_ai.ml.EXPERIMENTAL import EmbeddingFlowModel
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

@flowtask.extract(
    name="Gather CAD files from datasources",
    inputs=["cad_datasources"],
    outputs=["cad_dataset"],
    parallel_execution=True
)
def gather_cad_files(source: str) -> List[str]:
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
    
    return shuffled_files[:200]


nb_dir = pathlib.Path.cwd()
flows_outputdir = nb_dir.joinpath("out")

def get_flow_name():
    return "HOOPS Embedding Training"

flow_name = get_flow_name()
EmbeddingModel = EmbeddingFlowModel(result_dir= str(pathlib.Path(flows_outputdir).joinpath("flows").joinpath(flow_name)),
                                     log_file= str(pathlib.Path(flows_outputdir).joinpath("flows").joinpath(flow_name).joinpath("flow.log")) )


#dgl_storage = DGLGraphStoreHandler()

@flowtask.transform(
    name="Extracting CAD ML-input for EmbeddingModel",
    inputs=["cad_dataset"],
    outputs=["cad_files_encoded"],
    parallel_execution=True
)
def encode_data_for_ml_training(cad_file: str, cad_loader :  HOOPSLoader, storage : DataStorage) -> str:
    """Logic to prepare data for exploring and machine learning training - Shape Embedding's Model
    """

    facecount, edgecount = EmbeddingModel.encode_cad_data(cad_file, cad_loader, storage)
    
    dgl_storage = DGLGraphStoreHandler()

    # DGL graph Bin file
    item_no_suffix = pathlib.Path(cad_file).with_suffix("")  # Remove the suffix to get the base name
    hash_id = generate_unique_id_from_path(str(item_no_suffix))
    dgl_output_path = pathlib.Path(flows_outputdir).joinpath("flows", flow_name, "dgl", f"{hash_id}.ml")  
    dgl_output_path.parent.mkdir(parents=True, exist_ok=True)

    EmbeddingModel.convert_encoded_data_to_graph(storage, dgl_storage, str(dgl_output_path))
    
    # Save file-level metadata (will be routed to .infoset)
    storage.save_metadata("Item", str(cad_file))
    storage.save_metadata("source", "FABWAVE")
    
    # Compress the storage into a .data file
    storage.compress_store()
    
    # Return the base storage path
    return storage.get_file_path("")


