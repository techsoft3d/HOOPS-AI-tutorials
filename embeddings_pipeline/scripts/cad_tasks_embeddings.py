"""Flow tasks of the HOOPS Embeddings data preparation.

The encoding workers import this module, so it must not import torch. The workers only encode
CAD files. The main process builds the training graph files through graph_export, while the
workers keep encoding.
"""
import pathlib
import random
from typing import TYPE_CHECKING

from hoops_ai.cadaccess import HOOPSLoader
from hoops_ai.cadencoder.encode_cad_data import encode_cad_data
from hoops_ai.flowmanager import GraphExport, flowtask
from hoops_ai.ml.EXPERIMENTAL.embedding_encoding import EmbeddingEncodingConfig
from hoops_ai.storage import CADFileRetriever, DataStorage, LocalStorageProvider

if TYPE_CHECKING:
    from hoops_ai.ml.EXPERIMENTAL import EmbeddingFlowModel

nb_dir = pathlib.Path.cwd()
flows_outputdir = nb_dir.joinpath("out")

# Encoding settings shared by the workers and the model, so graphs match what the model expects.
ENCODING = EmbeddingEncodingConfig()


def get_flow_name() -> str:
    return "HOOPS_Embedding_Training"


flow_name = get_flow_name()


@flowtask.extract(
    name="Gather CAD files from datasources",
    inputs=["cad_datasources"],
    outputs=["cad_dataset"],
    parallel_execution=False
)
def gather_cad_files(source: str) -> list[str]:
    """Gather the CAD files of a source directory in a fixed shuffled order."""
    retriever = CADFileRetriever(
        storage_provider=LocalStorageProvider(directory_path=source),
        formats=[".stp", ".step", ".iges", ".igs", ".sldprt", ".SLDPRT"],
    )
    shuffled_files = list(retriever.get_file_list())
    random.seed(42)
    random.shuffle(shuffled_files)
    return shuffled_files


@flowtask.transform(
    name="Extracting CAD ML input for EmbeddingFlowModel",
    inputs=["cad_dataset"],
    outputs=["cad_files_encoded"],
    parallel_execution=True
)
def encode_data_for_ml_training(cad_file: str, cad_loader: HOOPSLoader, storage: DataStorage) -> str:
    """Encode one CAD file with the embedding model settings."""
    encode_cad_data(ENCODING.to_encode_specifications(), cad_file, cad_loader, storage)

    # Saved as file-level metadata, routed to the .infoset.
    storage.save_metadata("Item", str(cad_file))
    storage.save_metadata("source", "FABWAVE")
    return storage.get_file_path("")


def create_embedding_model() -> "EmbeddingFlowModel":
    """Build the flow model that writes the graph files. Runs in the main process only."""
    from hoops_ai.ml.EXPERIMENTAL import EmbeddingFlowModel  # loads torch, so keep it out of the workers

    flow_dir = flows_outputdir / "flows" / flow_name
    return EmbeddingFlowModel(
        result_dir=str(flow_dir),
        log_file=str(flow_dir / "flow.log"),
        **ENCODING.model_kwargs(),
    )


graph_export = GraphExport(flow_model_factory=create_embedding_model)
