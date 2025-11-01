# Flow Model Architecture Documentation

## Overview

This document provides comprehensive technical documentation for the **Flow Model architecture** in HOOPS AI. This architecture addresses a critical challenge in machine learning pipelines for CAD data: maintaining consistency between dataset preparation (training phase) and single-file inference (deployment phase).

### The Core Problem

Machine learning workflows for CAD data involve two distinct phases with different requirements:

1. **Training Phase (Dataset Processing)**
   - Process large CAD datasets into ML-ready input files
   - Encoded datasets (no longer CAD files) are used for data science experimentation
   - Training, validation, and test splits require careful management
   - Experimentation cycles work exclusively with ML-ready files for efficiency

2. **Inference Phase (Single File Processing)**
   - Trained model receives a new CAD file as input
   - Must encode the CAD data exactly as done during training
   - Operates on single files without dataset infrastructure
   - Requires "memory" of the encoding process used during training

**The Challenge:** How do we ensure that the encoding logic used to prepare training data is identical to the encoding used during inference? How do we avoid messy code duplication when handling batch datasets vs. single files?

**The Solution:** The `FlowModel` interface provides a unified abstraction that encapsulates:
- CAD data encoding strategies
- Label processing logic
- Graph conversion methods
- Model input preparation
- Model architecture references

This interface is consumed by both `FlowTrainer` (for batch dataset training) and `FlowInference` (for single-file predictions), guaranteeing encoding consistency across the ML lifecycle.

---

## ⚠️ EXPERIMENTAL STATUS

**Important:** This architecture is currently **EXPERIMENTAL** as it primarily focuses on fitting PyTorch Lightning model wrappers into a standardized interface. The schema and API may change in future releases based on:

- Evolving requirements for different ML architectures
- Performance optimization needs
- Integration with additional ML frameworks beyond PyTorch Lightning
- Community feedback and use cases

Users should expect potential breaking changes in upcoming versions as the architecture matures.

---

## Architecture Overview

The Flow Model architecture follows this relationship:

```
┌─────────────────────────────────────────────────────────────────┐
│                        FlowModel (Abstract)                      │
│  - encode_cad_data()                                            │
│  - encode_label_data()                                          │
│  - convert_encoded_data_to_graph()                              │
│  - load_model_input_from_files()                                │
│  - collate_function()                                           │
│  - retrieve_model()                                             │
│  - predict_and_postprocess()                                    │
│  - metrics()                                                    │
└─────────────────────────────────────────────────────────────────┘
                              ▲
                              │
                    Implemented by
                              │
         ┌────────────────────┴────────────────────┐
         │                                         │
┌────────────────────────┐              ┌─────────────────────────┐
│  GraphClassification   │              │ GraphNodeClassification │
│  (Graph classifier)    │              │  (Graph node classifier)│
└────────────────────────┘              └─────────────────────────┘
         │                                         │
         └────────────────────┬────────────────────┘
                              │
                    Consumed by
                              │
         ┌────────────────────┴────────────────────┐
         │                                         │
┌────────────────────────┐              ┌─────────────────────────┐
│     FlowTrainer        │              │     FlowInference       │
│  - Batch processing    │              │  - Single file          │
│  - Train/Val/Test      │              │  - Real-time encoding   │
│  - Checkpointing       │              │  - Model deployment     │
└────────────────────────┘              └─────────────────────────┘
```

---

## Table of Contents

1. [FlowModel Abstract Interface](#flowmodel-abstract-interface)
2. [Available Implementations](#available-implementations)
3. [FlowTrainer: Training Pipeline](#flowtrainer-training-pipeline)
4. [FlowInference: Deployment Pipeline](#flowinference-deployment-pipeline)
5. [Third-Party Models and Licensing](#third-party-models-and-licensing)
6. [Best Practices](#best-practices)

---

## FlowModel Abstract Interface

The `FlowModel` class is an abstract base class (ABC) defining the contract that all Flow Models must implement.

**Location:** `src/hoops_ai/ml/EXPERIMENTAL/flow_model.py`

### Core Philosophy

A `FlowModel` encapsulates **how** to:
1. Transform a CAD file into encoded features
2. Process labels for supervised learning
3. Convert encoded features into graph structures
4. Load model inputs from persisted files
5. Batch multiple inputs together (collation)
6. Retrieve the underlying PyTorch Lightning model
7. Post-process predictions into interpretable results
8. Access training metrics

### Abstract Methods

#### Data Processing Methods

##### `encode_cad_data(cad_file: str, cad_access: CADLoader, storage: DataStorage) -> Tuple[int, int]`

**Purpose:** Opens a CAD file and encodes its geometric/topological data into a format suitable for machine learning.

**Parameters:**
- `cad_file` (str): Path to the CAD file
- `cad_access` (CADLoader): CAD file loading interface
- `storage` (DataStorage): Storage handler for persisting encoded data

**Returns:**
- Tuple containing face count and edge count

**Workflow:**
1. Configure CAD loading options (features, solids, BREP settings)
2. Load the CAD model
3. Extract BREP representation
4. Use `BrepEncoder` to compute geometric features
5. Push features to storage

---

##### `encode_label_data(label_storage: LabelStorage, storage: DataStorage) -> Tuple[str, int]`

**Purpose:** Retrieves labeling information and stores it according to the ML task requirements.

**Parameters:**
- `label_storage` (LabelStorage): Interface to label data
- `storage` (DataStorage): Storage handler for persisting labels

**Returns:**
- Tuple containing label key and label count

**Key Considerations:**
- Handles different label granularities (graph-level vs. node-level)
- Validates label-entity compatibility (e.g., graph labels for graph classification)
- Stores both label codes and descriptions

---

##### `convert_encoded_data_to_graph(storage: DataStorage, graph: MLStorage, filename: str) -> Dict[str, Any]`

**Purpose:** Converts encoded features from storage into a graph representation suitable as ML model input.

**Parameters:**
- `storage` (DataStorage): Source of encoded features
- `graph` (MLStorage): Graph storage handler (e.g., DGL, PyTorch Geometric)
- `filename` (str): Output filename for the serialized graph

**Returns:**
- Dictionary with graph metadata (file size, node/edge counts, etc.)

**Workflow:**
1. Load graph structure (edges, nodes)
2. Attach node features (e.g., face discretization samples)
3. Attach edge features (e.g., edge curve grids)
4. Attach labels (if available)
5. Save graph to file

**Mathematical Representation:**

For a CAD model with faces $\mathcal{F} = \{f_0, \ldots, f_{N_f-1}\}$ and edges $\mathcal{E} = \{e_0, \ldots, e_{N_e-1}\}$:

- **Graph:** $G = (V, E)$ where $V = \mathcal{F}$ and $E \subseteq V \times V$
- **Node Features:** $\mathbf{X}_v \in \mathbb{R}^{d_v}$ for each $v \in V$
- **Edge Features:** $\mathbf{X}_e \in \mathbb{R}^{d_e}$ for each $e \in E$
- **Labels:** $y \in \{0, \ldots, C-1\}$ (graph-level) or $\mathbf{y} \in \{0, \ldots, C-1\}^{|V|}$ (node-level)

---

##### `load_model_input_from_files(graph_file: str, data_id: int, label_file: str = None) -> Any`

**Purpose:** Loads a persisted graph and prepares it as model input. Used by DataLoader during training and inference.

**Parameters:**
- `graph_file` (str): Path to serialized graph file
- `data_id` (int): Unique identifier for this data sample
- `label_file` (str, optional): Path to label file (None during inference)

**Returns:**
- Model-specific input format (e.g., DGL graph, PyTorch Geometric Data object)

**Key Design Point:** This method is called multiple times by `DatasetLoader`, both during training (with labels) and inference (without labels). It must handle both cases gracefully.

---

##### `collate_function(batch) -> Any`

**Purpose:** Combines multiple graph samples into a single batched input for the model.

**Parameters:**
- `batch`: List of samples returned by `load_model_input_from_files`

**Returns:**
- Batched model input (framework-specific format)

**Framework Examples:**
- **DGL:** Use `dgl.batch(graphs)` to create a batched graph
- **PyTorch Geometric:** Use `Batch.from_data_list(data_list)`

---

#### Model Interface Methods

##### `retrieve_model(check_point: str = None) -> pl.LightningModule`

**Purpose:** Returns the PyTorch Lightning model instance, optionally loaded from a checkpoint.

**Parameters:**
- `check_point` (str, optional): Path to saved model checkpoint

**Returns:**
- PyTorch Lightning module ready for training or inference

---

##### `predict_and_postprocess(batch) -> Any`

**Purpose:** Runs model inference on a batch and formats the output into interpretable predictions.

**Parameters:**
- `batch`: Batched model input from `collate_function`

**Returns:**
- Post-processed predictions (e.g., class labels, probabilities, segmentation masks)

**Typical Workflow:**
1. Set model to eval mode
2. Disable gradient computation
3. Forward pass through model
4. Apply softmax/argmax for classification
5. Convert to numpy/lists for downstream use

---

##### `model_name() -> str`

**Purpose:** Returns a human-readable name for the model.

---

##### `get_citation_info() -> Dict[str, Any]`

**Purpose:** Provides citation information for the underlying ML architecture.

**Returns:**
- Dictionary with keys: `author`, `paper`, `year`, `url`, `architecture`, `applications`

---

##### `metrics() -> MetricStorage`

**Purpose:** Returns the metric storage object containing training/validation metrics.

**Returns:**
- `MetricStorage` instance with logged metrics (loss, accuracy, etc.)

---

## Available Implementations

HOOPS AI provides two concrete implementations of the `FlowModel` interface, each based on state-of-the-art open-source architectures (see Acknowledgements section):

### 1. GraphClassification - Graph-Level Classifier

**Use Case:** Graph-level classification (e.g., part classification, shape categorization)

**Documentation:** [GraphClassification.md](./GraphClassification.md)

**Key Features:**
- Whole-model classification
- 2D CNNs on face discretization samples
- 1D CNNs on edge U-grids
- Ideal for: Part type identification, shape similarity

**Technology:** Based on a CNN+GNN architecture for learning from Boundary Representations

---

### 2. GraphNodeClassification - Graph Node Classifier

**Use Case:** Node-level classification (e.g., machining feature recognition, face segmentation)

**Documentation:** [GraphNodeClassification.md](./GraphNodeClassification.md)

**Key Features:**
- Per-face classification
- Transformer-based GNN architecture
- Rich topological feature encoding
- Ideal for: Feature recognition, semantic segmentation

**Technology:** Based on a Transformer+GNN architecture with enhanced topological encoding

---

## FlowTrainer: Training Pipeline

**File:** `src/hoops_ai/ml/EXPERIMENTAL/flow_trainer.py`

### Purpose

`FlowTrainer` orchestrates the complete training workflow for Flow Models, handling:
- Dataset loading and splitting (train/validation/test)
- Model initialization and checkpointing
- Training loop with PyTorch Lightning
- Metric logging and visualization
- Data quality validation (purify method)

### Key Advantage

By consuming the `FlowModel` interface, `FlowTrainer` **automatically knows** how to:
1. Load encoded graph files using `load_model_input_from_files()`
2. Batch multiple samples using `collate_function()`
3. Initialize the correct model architecture via `retrieve_model()`
4. Access training metrics through `metrics()`

**Result:** Write the encoding logic once in your `FlowModel` implementation, and both training and inference use it consistently.

---

### Initialization

```python
from hoops_ai.ml import FlowTrainer
from hoops_ai.ml import GraphNodeClassification
from hoops_ai.dataset import DatasetLoader

# Create your flow model
flowmodel = GraphNodeClassification(
    num_classes=25,
    n_layers_encode=8,
    result_dir="./results"
)

# Create dataset loader
dataset_loader = DatasetLoader(
    graph_files=["path/to/graphs/*.bin"],
    label_files=["path/to/labels/*.json"]
)
dataset_loader.split_data(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

# Create trainer
trainer = FlowTrainer(
    flowmodel=flowmodel,
    datasetLoader=dataset_loader,
    batch_size=32,
    num_workers=4,
    experiment_name="machining_feature_recognition",
    accelerator='gpu',
    devices=1,
    gradient_clip_val=1.0,
    max_epochs=100,
    learning_rate=0.002,
    result_dir="./results"
)
```

**Parameters:**

**Core Components:**
- `flowmodel` (FlowModel): Initialized FlowModel implementation
- `datasetLoader` (DatasetLoader): Dataset with train/val/test splits

**Training Configuration:**
- `batch_size` (int): Samples per training batch (default: 64)
- `num_workers` (int): DataLoader worker processes (default: 0)
- `experiment_name` (str): Name for logging and checkpoints
- `accelerator` (str): 'cpu', 'gpu', or 'tpu' (default: 'cpu')
- `devices` (int or 'auto'): Number of devices to use (default: 'auto')
- `gradient_clip_val` (float): Gradient clipping threshold (default: 1.0)
- `max_epochs` (int): Maximum training epochs (default: 100)
- `learning_rate` (float): Initial learning rate (default: 0.002)
- `result_dir` (str): Output directory for results

---

### Key Methods

#### `train() -> str`

Executes the full training loop and returns the path to the best checkpoint.

**Returns:**
- `str`: Path to best model checkpoint

**Workflow:**
1. Initialize model from `flowmodel.retrieve_model()`
2. Create DataLoaders for train/val/test datasets
3. Configure PyTorch Lightning Trainer with callbacks and loggers
4. Execute training loop with automatic validation
5. Save best checkpoint based on validation loss
6. Log metrics via TensorBoard

**Example:**
```python
best_checkpoint_path = trainer.train()
print(f"Training complete! Best model: {best_checkpoint_path}")
```

---

#### `test(trained_model_path: str)`

Evaluates the trained model on the test dataset.

**Example:**
```python
trainer.test(trained_model_path=best_checkpoint_path)
```

---

#### `purify(num_processes: int = 1, chunks_per_process: int = 1)`

Validates data quality by attempting to load all samples and identifying corrupted/problematic data.

**Purpose:** ML training can fail due to corrupted graph files, mismatched dimensions, or encoding errors. This method proactively identifies problematic samples.

---

#### `metrics_storage() -> MetricStorage`

Returns the metric storage object containing logged training metrics.

---

## FlowInference: Deployment Pipeline

**File:** `src/hoops_ai/ml/EXPERIMENTAL/flow_inference.py`

### Purpose

`FlowInference` handles single-file CAD inference using trained Flow Models, providing:
- On-the-fly CAD encoding (identical to training encoding)
- Model checkpoint loading
- Single-file prediction pipeline
- Clean separation from batch training infrastructure

### Key Advantage

By consuming the same `FlowModel` interface used during training, `FlowInference` **guarantees** that:
1. CAD files are encoded using the exact same logic as training data
2. Graph construction follows the same schema
3. Feature dimensions match model expectations
4. No code duplication between training and inference

**Result:** Train once, deploy confidently knowing the encoding pipeline is consistent.

---

### Initialization

```python
from hoops_ai.ml import FlowInference
from hoops_ai.ml import GraphNodeClassification
from hoops_ai.cadaccess import HOOPSLoader

# Initialize CAD loader
cad_loader = HOOPSLoader()

# Create the same flow model used during training
flowmodel = GraphNodeClassification(
    num_classes=25,
    n_layers_encode=8,
    result_dir="./inference_results"
)

# Create inference pipeline
inference = FlowInference(
    cad_loader=cad_loader,
    flowmodel=flowmodel,
    log_file='inference_errors.log'
)

# Load trained model
inference.load_from_checkpoint("path/to/best.ckpt")
```

---

### Key Methods

#### `load_from_checkpoint(checkpoint_path: str)`

Loads a trained model from a checkpoint file.

---

#### `preprocess(file_path: str) -> Dict[str, torch.Tensor]`

Encodes a single CAD file into a model-ready input batch.

**Returns:**
- `Dict[str, torch.Tensor]`: Batched model input

**Example:**
```python
batch = inference.preprocess("path/to/new_part.step")
```

---

#### `predict_and_postprocess(batch: Dict[str, torch.Tensor]) -> np.ndarray`

Runs model inference and returns formatted predictions.

**Example:**
```python
predictions = inference.predict_and_postprocess(batch)
print(f"Predicted class: {predictions['predictions'][0]}")
```

---

### Complete Inference Example

```python
from hoops_ai.ml import FlowInference, GraphNodeClassification
from hoops_ai.cadaccess import HOOPSLoader
import time

# Setup
cad_loader = HOOPSLoader()
flowmodel = GraphNodeClassification(num_classes=25)
inference = FlowInference(cad_loader, flowmodel)
inference.load_from_checkpoint("./trained_models/best.ckpt")

# Inference on new CAD file
start_time = time.time()
batch = inference.preprocess("new_part.step")
predictions = inference.predict_and_postprocess(batch)
total_time = time.time() - start_time

# Results
print(f"Inference completed in {total_time:.2f} seconds")
print(f"Face predictions: {predictions['node_predictions']}")
```

---



## Related Documentation

- [GraphClassification Documentation](./GraphClassification.md)
- [GraphNodeClassification Documentation](./GraphNodeClassification.md)
- [Acknowledgements](./Acknowledgements.md) - Attribution and citations for third-party architectures
- [Module Access & Encoder Documentation](./Module_Access_and_Encoder.md)
- [DataStorage Documentation](./DataStorage_Documentation.md)
- [Flow Documentation](./Flow_Documentation.md)

---


