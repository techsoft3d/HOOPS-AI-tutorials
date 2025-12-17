########################################################################################################################
# Copyright (c) 2025 by Tech Soft 3D, Inc.
# The information contained herein is confidential and proprietary to Tech Soft 3D, Inc., and considered a trade secret
# as defined under civil and criminal statutes. Tech Soft 3D shall pursue its civil and criminal remedies in the event
# of unauthorized use or misappropriation of its trade secrets. Use of this information by anyone other than authorized
# employees of Tech Soft 3D, Inc. is granted only under a written non-disclosure agreement, expressly prescribing the
# scope and manner of such use.
#
########################################################################################################################

from typing import Any, Dict, Tuple, List
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any
from dgl.data.utils import load_graphs
import os
os.environ["DGLBACKEND"] = "pytorch"
import dgl
import pathlib

from hoops_ai.ml.EXPERIMENTAL.flow_model_graph_classification import GraphClassification
from hoops_ai.storage.datastorage.data_storage_handler_base import DataStorage
from hoops_ai.storage.graphstorage.graph_storage_handler import MLStorage
from hoops_ai.storage.label_storage import LabelStorage
from hoops_ai.cadaccess.cad_loader import CADLoader
from hoops_ai.cadaccess.hoops_exchange.hoops_access import HOOPSTools
from hoops_ai.cadencoder.brep_encoder import BrepEncoder
from hoops_ai.storage.datastorage.zarr_storage_handler import OptStorage
from hoops_ai.storage.metric_storage import MetricStorage


class CustomGraphClassification(GraphClassification):
    """
    CUSTOM GraphClassification is a user-friendly wrapper around the GraphClassiication model.
    It provides default hyperparameters and a way to dinaically change the Y in the supervise task.
    This allows multiple training from the same dataset by calling the method set_label_for_training 
    before calling hte FLowTrainer train method. 

    Args:
        num_classes (int, optional): Number of classes for classification. Default: 10
        log_file (str, optional): Path to the log file. Default: 'training_errors.log'
    """

    def __init__(
        self,
        num_classes: int = 10,
        result_dir: str = None,
        log_file: str = 'custom_graph_classification_training_errors.log',
        generate_stream_cache_for_visu: bool = False
    ):
        # Call parent constructor to initialize all base functionality
        super().__init__(
            num_classes=num_classes,
            result_dir=result_dir,
            log_file=log_file,
            generate_stream_cache_for_visu=generate_stream_cache_for_visu
        )

        self.input_label_for_training = "graph_label"

    def set_label_for_training(self, label_for_training: str = "graph_label") -> None:
        """
        From the list of labels store in the graph_handler: MLStorage choose the name of the key to be used as the Y in the classification task
        
        """
        self.input_label_for_training = label_for_training
    # ========================================================================
    # OVERRIDABLE METHODS - Customize these methods for custom behavior
    # ========================================================================
    
    def encode_cad_data(self, cad_file: str, cad_loader: CADLoader, storage: DataStorage) -> Tuple[int, int]:
        """
        Opens the CAD file and encodes its data into a format suitable for machine learning.
        Override this method to customize CAD encoding logic.
        
        Returns:
            Tuple[int, int]: (face_count, edge_count)
        """
        return super().encode_cad_data(cad_file, cad_loader, storage)

    def encode_label_data(self, label_storage: LabelStorage, storage: DataStorage) -> Tuple[str, int]:
        """
        Encodes label data for the model.
        Override this method to customize label encoding logic.
        
        Returns:
            Tuple[str, int]: (label_cadentity, label_count)
        """
        return super().encode_label_data(label_storage, storage)
    
    def convert_encoded_data_to_graph(self, storage: DataStorage, graph_handler: MLStorage, filename: str) -> Dict[str, Any]:
        """
        Converts encoded data from storage into a graph representation.
        Override this method to customize graph conversion logic.
        """
        return super().convert_encoded_data_to_graph(storage, graph_handler, filename)



    def load_model_input_from_files(self, graph_file: str, data_id: int, label_file: str = None) -> Any:
        """
        Loads a single graph from a file to be used as input for the machine learning model.
        Override this method to customize data loading logic.
        """
        graphs, aux_dict = load_graphs(str(graph_file))
        graph = graphs[0]

        # for the training we will here decide which label to use
        key_label = self.input_label_for_training
        label = aux_dict.get(key_label, None)
        graph_file_path = pathlib.Path(graph_file)  # Convert to Path object
        sample = {"graph": graph, "label" : label} # "filename": graph_file_path.stem,

        return sample

    def collate_function(self, batch) -> Any:
        """
        Collates a batch of samples for model input.
        Override this method to customize batching logic.
        """
        return super().collate_function(batch)
    
    def predict_and_postprocess(self, batch) -> Any:
        """
        Post-processes and formats the raw model output into a structured prediction.
        Override this method to customize prediction output format.
        
        Returns:
            numpy.ndarray: Array with shape (batch_size, 2, 3), where:
            - First dimension: batch items
            - Second dimension [0]: class indices (int)
            - Second dimension [1]: probability percentages (int)
        """
        return super().predict_and_postprocess(batch)

    def model_name(self) -> str:
        """
        Returns the name of this custom model.
        Override this method to provide a custom model name.
        """
        return f"CUSTOM GRAPH CLASSIFICATION MODEL : Y={self.input_label_for_training}"


  
