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
│  (UV-Net wrapper)      │              │  (BrepMFR wrapper)      │
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
2. [Concrete Implementations](#concrete-implementations)
   - [GraphClassification (UV-Net)](#graphclassification-uv-net)
   - [GraphNodeClassification (BrepMFR)](#graphnodeclassification-brepmfr)
3. [FlowTrainer: Training Pipeline](#flowtrainer-training-pipeline)
4. [FlowInference: Deployment Pipeline](#flowinference-deployment-pipeline)
5. [Third-Party Models and Licensing](#third-party-models-and-licensing)
6. [Usage Examples](#usage-examples)

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

**Example Implementation Pattern:**
```python
def encode_cad_data(self, cad_file: str, cad_loader: CADLoader, storage: DataStorage):
    # Configure loading
    general_options = cad_loader.get_general_options()
    general_options["read_feature"] = True 
    general_options["read_solid"] = True 
    
    # Load model
    model = cad_loader.create_from_file(cad_file)
    
    # Configure BREP
    hoopstools = HOOPSTools()
    brep_options = hoopstools.brep_options()
    brep_options["force_compute_uv"] = True       
    brep_options["force_compute_3d"] = True 
    hoopstools.adapt_brep(model, brep_options)
    
    # Encode features
    brep_encoder = BrepEncoder(model.get_brep(body_index=0), storage)
    brep_encoder.push_face_adjacency_graph()
    brep_encoder.push_facegrid(10, 10)
    brep_encoder.push_edge_attributes()
    # ... other encoding methods
```

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
2. Attach node features (e.g., face UV grids)
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

## Concrete Implementations

### GraphClassification (UV-Net)

**File:** `src/hoops_ai/ml/EXPERIMENTAL/flow_model_graph_classification.py`

#### Model Overview

`GraphClassification` is a wrapper around the **UV-Net** architecture for CAD model classification tasks.

**Original Paper:**
> Jayaraman, P. K., Sanghi, A., Lambourne, J. G., Willis, K. D. D., Davies, T., Shayani, H., & Morris, N. (2021). UV-Net: Learning from Boundary Representations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (pp. 11703-11712). https://doi.org/10.1109/CVPR46437.2021.01153

**Model Architecture:**
UV-Net operates directly on Boundary Representation (B-rep) data from 3D CAD models. It represents:
- **Face Geometry:** 2D UV-grids sampled uniformly in the parameter domain
- **Edge Geometry:** 1D U-grids along edge curves
- **Topology:** Face-adjacency graph where node features encode faces and edge features encode edges

The architecture applies:
- 2D CNNs to face UV-grids
- 1D CNNs to edge U-grids
- Graph neural networks to aggregate topological information

**Original Applications:**
- Auto-complete of modeling operations
- Smart selection tools in CAD software
- Shape similarity search

**GitHub Repository:** https://github.com/AutodeskAILab/UV-Net

---

#### Initialization

```python
from hoops_ai.ml import GraphClassification

model = GraphClassification(
    num_classes=10,
    result_dir="./results",
    log_file="training_errors.log",
    generate_stream_cache_for_visu=False
)
```

**Parameters:**
- `num_classes` (int): Number of classification categories (default: 10)
- `result_dir` (str): Directory for saving results and metrics
- `log_file` (str): Path to error logging file
- `generate_stream_cache_for_visu` (bool): Whether to generate visualization cache

---

#### Key Implementation Details

##### CAD Encoding Strategy

```python
def encode_cad_data(self, cad_file, cad_loader, storage):
    # Force UV computation for face parameterization
    brep_options["force_compute_uv"] = True       
    brep_options["force_compute_3d"] = True 
    
    brep_encoder = BrepEncoder(model.get_brep(body_index=0), storage)
    
    # Graph structure
    brep_encoder.push_face_adjacency_graph()
    
    # Node features (faces)
    brep_encoder.push_face_attributes()
    brep_encoder.push_facegrid(10, 10)  # 10x10 UV grid
    
    # Edge features
    brep_encoder.push_edge_attributes()
    brep_encoder.push_curvegrid(10)  # 10 points along edge
    
    # Additional topological features
    brep_encoder.push_face_pair_edges_path(16)
```

**Feature Dimensions:**
- **Node Features:** Face UV-grids of shape $(10, 10, 7)$ where 7 = $(x, y, z, n_x, n_y, n_z, visibility)$
- **Edge Features:** Edge U-grids encoding curve geometry

---

##### Graph Construction

```python
def convert_encoded_data_to_graph(self, storage, graph_handler, filename):
    # Load topology
    graph_data = storage.load_data("graph")
    src = graph_data["edges"]["source"]
    dst = graph_data["edges"]["destination"]
    graph_handler.setup_graph(source_nodes=src, target_nodes=dst, 
                              num_nodes=graph_data["num_nodes"])
    
    # Attach features
    face_uv = storage.load_data("face_uv_grids")
    graph_handler.append_ndata(face_uv, feature_name="x", torch_type=torch.float32)
    
    edge_u_grids = storage.load_data("edge_u_grids")
    graph_handler.append_edata(edge_u_grids, feature_name="x", torch_type=torch.float32)
    
    # Attach label (if available)
    if LabelStorage.GRAPH_CADENTITY in storage.get_keys():
        graph_label = storage.load_data(LabelStorage.GRAPH_CADENTITY)
        graph_handler.append_extra_data(graph_label, feature_name="graph_label", 
                                       torch_type=torch.long)
    
    graph_handler.save_graph(filename)
```

---

##### Model Input Loading

```python
def load_model_input_from_files(self, graph_file, data_id, label_file=None):
    graphs, _ = load_graphs(graph_file)
    graph = graphs[0]
    graph.ndata["x"] = graph.ndata["x"].float()
    graph.edata["x"] = graph.edata["x"].float()
    
    if label_file:
        # Training mode: load labels
        label_data = load_label_data(label_file)
        graph.graph_label = label_data
    else:
        # Inference mode: no labels needed
        pass
    
    return graph
```

---

##### Prediction and Post-processing

```python
def predict_and_postprocess(self, batch):
    self.model.eval()
    with torch.no_grad():
        # Extract components
        graph = batch['graph']
        graph_labels = batch.get('graph_label')
        
        # Forward pass
        logits = self.model(graph.ndata['x'], 
                           graph.edata['x'], 
                           graph)
        
        # Convert to predictions
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        
        return {
            'predictions': preds.cpu().numpy(),
            'probabilities': probs.cpu().numpy(),
            'labels': graph_labels.cpu().numpy() if graph_labels is not None else None
        }
```

---

### GraphNodeClassification (BrepMFR)

**File:** `src/hoops_ai/ml/EXPERIMENTAL/flow_model_graphnode_classification.py`

#### Model Overview

`GraphNodeClassification` is a wrapper around the **BrepMFR** architecture for node-level classification tasks (e.g., machining feature recognition).

**Original Paper:**
> Zhang, S., Guan, Z., Jiang, H., Wang, X., & Tan, P. (2024). BrepMFR: Enhancing machining feature recognition in B-rep models through deep learning and domain adaptation. Computer Aided Geometric Design, 111, 102318. https://www.sciencedirect.com/science/article/abs/pii/S0167839624000529

**Model Architecture:**
BrepMFR converts B-rep models into graph representations where:
- **Nodes:** Individual faces of the CAD model
- **Edges:** Adjacency relationships between faces
- **Architecture:** Graph neural network based on Transformer architecture with graph attention mechanisms

The network encodes:
- Local geometric shapes (face-level features)
- Global topological relationships (graph structure)
- Achieves high-level semantic extraction for feature classification

**Original Applications:**
- Machining feature recognition in CAD/CAM workflows
- Recognizing highly intersecting features with complex geometries
- Transfer learning from synthetic to real-world CAD models

**GitHub Repository:** https://github.com/zhangshuming0668/BrepMFR

---

#### Initialization

```python
from hoops_ai.ml import GraphNodeClassification

model = GraphNodeClassification(
    num_classes=25,
    n_layers_encode=8,
    dim_node=256,
    d_model=512,
    n_heads=32,
    dropout=0.3,
    attention_dropout=0.3,
    act_dropout=0.3,
    learning_rate=0.002,
    optimizer_betas=(0.99, 0.999),
    scheduler_factor=0.5,
    scheduler_patience=5,
    scheduler_threshold=1e-4,
    scheduler_min_lr=1e-6,
    scheduler_cooldown=2,
    max_warmup_steps=5000,
    log_file='training_errors.log',
    result_dir="./results",
    generate_stream_cache_for_visu=False
)
```

**Parameters:**

**Model Architecture:**
- `num_classes` (int): Number of machining feature classes (default: 25)
- `n_layers_encode` (int): Number of Transformer encoder layers (default: 8)
- `dim_node` (int): Node embedding dimension (default: 256)
- `d_model` (int): Transformer model dimension (default: 512)
- `n_heads` (int): Number of attention heads (default: 32)
- `dropout` (float): Classifier dropout rate (default: 0.3)
- `attention_dropout` (float): Attention mechanism dropout (default: 0.3)
- `act_dropout` (float): Activation layer dropout (default: 0.3)

**Training Hyperparameters:**
- `learning_rate` (float): Initial learning rate (default: 0.002)
- `optimizer_betas` (Tuple[float, float]): AdamW optimizer betas (default: (0.99, 0.999))
- `scheduler_factor` (float): LR reduction factor (default: 0.5)
- `scheduler_patience` (int): Patience for LR scheduler (default: 5)
- `scheduler_threshold` (float): Threshold for scheduler (default: 1e-4)
- `scheduler_min_lr` (float): Minimum learning rate (default: 1e-6)
- `scheduler_cooldown` (int): Cooldown period after LR reduction (default: 2)
- `max_warmup_steps` (int): Warmup steps for learning rate (default: 5000)

**Logging and Output:**
- `log_file` (str): Error logging file path
- `result_dir` (str): Results output directory
- `generate_stream_cache_for_visu` (bool): Generate visualization cache

---

#### Key Implementation Details

##### CAD Encoding Strategy

```python
def encode_cad_data(self, cad_file, cad_loader, storage):
    # Force UV computation
    brep_options["force_compute_uv"] = True       
    brep_options["force_compute_3d"] = True 
    
    brep_encoder = BrepEncoder(model.get_brep(body_index=0), storage)
    
    # Graph structure
    brep_encoder.push_face_adjacency_graph()
    
    # Rich node features
    brep_encoder.push_face_attributes()
    brep_encoder.push_facegrid(10, 10)
    
    # Edge features
    brep_encoder.push_edge_attributes()
    brep_encoder.push_curvegrid(10)
    
    # Extended topological information
    brep_encoder.push_extended_adjacency()
    brep_encoder.push_face_neighbors_count()
    brep_encoder.push_face_pair_edges_path(16)
    
    # Geometric relationships
    brep_encoder.push_average_face_pair_angle_histograms(5, 64)
    brep_encoder.push_average_face_pair_distance_histograms(5, 64)
```

**Feature Richness:** BrepMFR uses more extensive topological features compared to UV-Net, including:
- Extended adjacency information
- Face neighbor counts
- Face pair geometric relationships (angles, distances)
- Histogram-based geometric encodings

---

##### Label Processing (Node-Level)

```python
def encode_label_data(self, label_storage, storage):
    mlTask = "Machining_feature_recognition"
    
    # Load node-level labels
    label_code_list = label_storage.load_face_labels(mlTask)
    label_description = label_storage.load_description(mlTask)
    label_cadentity = label_storage.load_cadentity(mlTask)
    
    # Validate entity type
    if label_cadentity != LabelStorage.FACE_CADENTITY:
        raise ValueError(f"Expected face-level labels, got {label_cadentity}")
    
    # Store per-face labels
    storage.save_data(LabelStorage.FACE_CADENTITY, 
                     np.array(label_code_list))
    storage.save_metadata(f"descriptions/{LabelStorage.FACE_CADENTITY}", 
                         label_description)
```

**Key Difference:** Unlike graph-level classification, node classification requires labels for each node (face) in the graph.

---

##### Graph Construction

```python
def convert_encoded_data_to_graph(self, storage, graph_handler, filename):
    # Load topology
    graph_data = storage.load_data("graph")
    src = graph_data["edges"]["source"]
    dst = graph_data["edges"]["destination"]
    
    graph_handler.setup_graph(source_nodes=src, target_nodes=dst, 
                              num_nodes=graph_data["num_nodes"])
    
    # Node features
    face_uv = storage.load_data("face_uv_grids")
    graph_handler.append_ndata(face_uv, feature_name="x", 
                              torch_type=torch.float32)
    
    # Edge features
    edge_u_grids = storage.load_data("edge_u_grids")
    graph_handler.append_edata(edge_u_grids, feature_name="x", 
                              torch_type=torch.float32)
    
    # Node labels (per-face)
    if LabelStorage.FACE_CADENTITY in storage.get_keys():
        face_labels = storage.load_data(LabelStorage.FACE_CADENTITY)
        graph_handler.append_ndata(face_labels, feature_name="y", 
                                  torch_type=torch.long)
    
    graph_handler.save_graph(filename)
```

---

##### Prediction and Post-processing

```python
def predict_and_postprocess(self, batch):
    self.model.eval()
    with torch.no_grad():
        # Extract components
        graph = batch['graph']
        face_labels = batch.get('labels')
        
        # Forward pass
        logits = self.model(graph.ndata['x'], 
                           graph.edata['x'], 
                           graph)
        
        # Per-node predictions
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        
        return {
            'node_predictions': preds.cpu().numpy(),
            'node_probabilities': probs.cpu().numpy(),
            'node_labels': face_labels.cpu().numpy() if face_labels is not None else None,
            'num_faces': graph.number_of_nodes()
        }
```

**Output Structure:** Returns per-face predictions rather than a single graph-level prediction.

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
- `str`: Path to best model checkpoint (e.g., `"./results/ml_output/experiment/1030/143022/best.ckpt"`)

**Workflow:**
1. Initialize model from `flowmodel.retrieve_model()`
2. Create DataLoaders for train/val/test datasets using `flowmodel.load_model_input_from_files()` and `flowmodel.collate_function()`
3. Configure PyTorch Lightning Trainer with callbacks and loggers
4. Execute training loop with automatic validation
5. Save best checkpoint based on validation loss
6. Log metrics via TensorBoard

**Example:**
```python
best_checkpoint_path = trainer.train()
print(f"Training complete! Best model: {best_checkpoint_path}")
```

**Output Structure:**
```
results/
└── ml_output/
    └── machining_feature_recognition/
        └── 1030/  # Month-Day
            └── 143022/  # Hour-Minute-Second
                ├── best.ckpt
                ├── last.ckpt
                └── epoch=X-step=Y.ckpt
```

---

#### `test(trained_model_path: str)`

Evaluates the trained model on the test dataset.

**Parameters:**
- `trained_model_path` (str): Path to trained model checkpoint

**Workflow:**
1. Load model from checkpoint
2. Run inference on test dataset
3. Compute test metrics
4. Save results to output directory

**Example:**
```python
trainer.test(trained_model_path=best_checkpoint_path)
```

---

#### `purify(num_processes: int = 1, chunks_per_process: int = 1)`

Validates data quality by attempting to load all samples and identifying corrupted/problematic data.

**Parameters:**
- `num_processes` (int): Number of parallel processes
- `chunks_per_process` (int): Chunks per process

**Purpose:** ML training can fail due to corrupted graph files, mismatched dimensions, or encoding errors. This method proactively identifies problematic samples.

**Output:** Creates `chunk_errors/` directory with JSON files listing problematic samples:
```json
{
  "chunk_index": 0,
  "chunk_size": 100,
  "train_errors": [
    {"index": 42, "error": "Shape mismatch in face_uv_grids"}
  ],
  "val_errors": [],
  "test_errors": []
}
```

---

#### `metrics_storage() -> MetricStorage`

Returns the metric storage object containing logged training metrics.

**Returns:**
- `MetricStorage`: Object with training/validation metrics

**Example:**
```python
metrics = trainer.metrics_storage()
train_loss = metrics.get("train_loss")
val_accuracy = metrics.get("val_accuracy")
```

---

### Internal Architecture

#### Dataset Handling

`FlowTrainer` converts `DatasetLoader` outputs into PyTorch-compatible datasets:

```python
# Create item loader function from flowmodel
def item_loader_func(graph_file, data_id, label_file):
    return flowmodel.load_model_input_from_files(graph_file, data_id, label_file)

# Set on dataset loader
datasetLoader.item_loader_func = item_loader_func

# Get PyTorch datasets
collate_fn = flowmodel.collate_function
train_data = datasetLoader.get_dataset("train").to_torch(collate_fn=collate_fn)
val_data = datasetLoader.get_dataset("validation").to_torch(collate_fn=collate_fn)
test_data = datasetLoader.get_dataset("test").to_torch(collate_fn=collate_fn)
```

**Key Insight:** The `FlowModel` interface bridges the gap between HOOPS AI's storage system and PyTorch's DataLoader requirements.

---

#### Checkpoint Management

Uses PyTorch Lightning's `ModelCheckpoint` callback:

```python
checkpoint_callback = ModelCheckpoint(
    monitor="eval_loss",
    dirpath=str(checkpointdir),
    filename="best",
    save_top_k=3,
    save_last=True
)
```

**Behavior:**
- Monitors validation loss (`eval_loss`)
- Saves top 3 checkpoints
- Always saves last checkpoint
- Automatically loads best checkpoint for testing

---

#### Logging

Integrates with TensorBoard for metric visualization:

```python
logger = TensorBoardLogger(
    save_dir=str(results_path),
    name=month_day,
    version=hour_min_second
)
```

**View Logs:**
```bash
tensorboard --logdir=./results/ml_output/experiment_name/
```

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

**Parameters:**
- `cad_loader` (CADLoader): CAD file loading interface
- `flowmodel` (FlowModel): Same FlowModel implementation used during training
- `log_file` (str): Error logging file path

---

### Key Methods

#### `load_from_checkpoint(checkpoint_path: str)`

Loads a trained model from a checkpoint file.

**Parameters:**
- `checkpoint_path` (str): Path to `.ckpt` file from training

**Example:**
```python
inference.load_from_checkpoint("./results/ml_output/experiment/1030/143022/best.ckpt")
```

**Behavior:**
- Calls `flowmodel.retrieve_model(checkpoint_path)` to load architecture + weights
- Sets model to evaluation mode
- Moves model to CPU (or specified device)

---

#### `preprocess(file_path: str) -> Dict[str, torch.Tensor]`

Encodes a single CAD file into a model-ready input batch.

**Parameters:**
- `file_path` (str): Path to CAD file (e.g., `.step`, `.stp`, `.sat`)

**Returns:**
- `Dict[str, torch.Tensor]`: Batched model input (same format as training batches)

**Workflow:**
1. Create in-memory storage (no disk I/O)
2. Call `flowmodel.encode_cad_data()` to extract features
3. Call `flowmodel.convert_encoded_data_to_graph()` to build graph
4. Generate unique ID for the file
5. Call `flowmodel.load_model_input_from_files()` to prepare model input
6. Call `flowmodel.collate_function()` to create batch (size 1)
7. Clean up temporary files

**Example:**
```python
batch = inference.preprocess("path/to/new_part.step")
print(f"Preprocessed in {elapsed_time:.2f} seconds")
```

**Key Design:** Uses `MemoryStorage` to avoid disk I/O overhead during real-time inference.

---

#### `predict_and_postprocess(batch: Dict[str, torch.Tensor]) -> np.ndarray`

Runs model inference and returns formatted predictions.

**Parameters:**
- `batch` (Dict[str, torch.Tensor]): Output from `preprocess()`

**Returns:**
- `np.ndarray`: Post-processed predictions (format depends on FlowModel implementation)

**Example:**
```python
predictions = inference.predict_and_postprocess(batch)

# For graph classification
print(f"Predicted class: {predictions['predictions'][0]}")
print(f"Confidence: {predictions['probabilities'][0]}")

# For node classification
print(f"Face predictions: {predictions['node_predictions']}")
print(f"Number of faces: {predictions['num_faces']}")
```

**Workflow:**
1. Set model to evaluation mode
2. Disable gradients for efficiency
3. Forward pass through model
4. Post-process via `flowmodel.predict_and_postprocess(batch)`
5. Return formatted results

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
print(f"Confidence scores: {predictions['node_probabilities']}")
```

---

### Performance Considerations

#### Memory Optimization

Uses `MemoryStorage` instead of `OptStorage` (Zarr) for inference:
- **No disk I/O:** All encoding happens in RAM
- **Faster preprocessing:** Eliminates serialization overhead
- **Temporary data:** Graph files are cleaned up immediately

#### Timing Breakdown

```python
# Typical timing for a 100-face CAD model on CPU:
# - CAD loading: 0.5s
# - Feature encoding: 2.0s
# - Graph construction: 0.3s
# - Model inference: 0.2s
# Total: ~3.0s
```

---

## Third-Party Models and Licensing

### Overview

HOOPS AI integrates open-source machine learning architectures to provide state-of-the-art CAD analysis capabilities. These models are located in the `src/hoops_ai/ml/_thirdparty/` directory and are used under their respective open-source licenses.

---

### Included Models

#### 1. UV-Net (Autodesk AI Lab)

**Source:** https://github.com/AutodeskAILab/UV-Net  
**License:** MIT License  
**Authors:** Jayaraman, P. K., Sanghi, A., Lambourne, J. G., Willis, K. D. D., Davies, T., Shayani, H., & Morris, N.  
**Year:** 2021  
**Location:** `src/hoops_ai/ml/_thirdparty/uvnet/`

**Modifications:**
- Integration with HOOPS AI storage system
- Custom collation functions for DGL graphs
- Metric logging via `MetricStorage`
- Configuration for PyTorch Lightning training

**Original Copyright:**
```
Copyright (c) 2021 Autodesk, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

#### 2. BrepMFR (Zhang et al.)

**Source:** https://github.com/zhangshuming0668/BrepMFR  
**License:** MIT License  
**Authors:** Zhang, S., Guan, Z., Jiang, H., Wang, X., & Tan, P.  
**Year:** 2024  
**Location:** `src/hoops_ai/ml/_thirdparty/brepmfr/`

**Modifications:**
- Adapter for HOOPS AI graph storage format
- Integration with `FlowTrainer` for training pipelines
- Custom learning rate schedulers
- Error logging and debugging enhancements

**Original Copyright:**
```
Copyright (c) 2024 Zhang Shuming and contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

### Attribution and Compliance

**Tech Soft 3D Stance on Third-Party Code:**

While HOOPS AI provides convenient wrappers (`GraphClassification`, `GraphNodeClassification`) to integrate these architectures into the Flow Model framework, **the original authors retain full credit for their pioneering work**. 

Our modifications are limited to:
1. **Interface Adaptation:** Implementing the `FlowModel` abstract interface
2. **Storage Integration:** Connecting to HOOPS AI's data storage system
3. **Training Infrastructure:** Enabling use with `FlowTrainer` and `FlowInference`

**We do NOT claim authorship of the underlying ML architectures.** Users of HOOPS AI should cite the original papers when publishing results using these models:

**For UV-Net:**
```bibtex
@inproceedings{jayaraman2021uvnet,
  title={UV-Net: Learning from Boundary Representations},
  author={Jayaraman, Pradeep Kumar and Sanghi, Aditya and Lambourne, Joseph G and Willis, Karl DD and Davies, Thomas and Shayani, Hooman and Morris, Nigel},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11703--11712},
  year={2021}
}
```

**For BrepMFR:**
```bibtex
@article{zhang2024brepmfr,
  title={BrepMFR: Enhancing machining feature recognition in B-rep models through deep learning and domain adaptation},
  author={Zhang, Shuming and Guan, Zhiguang and Jiang, Han and Wang, Xiaojun and Tan, Ping},
  journal={Computer Aided Geometric Design},
  volume={111},
  pages={102318},
  year={2024},
  publisher={Elsevier}
}
```

---

### MIT License Compliance

Both integrated models are distributed under the **MIT License**, which permits:
- ✅ Commercial use
- ✅ Modification
- ✅ Distribution
- ✅ Private use

**Requirements:**
- Include the original copyright notice
- Include the MIT license text
- Acknowledge modifications made by Tech Soft 3D

HOOPS AI complies with these requirements by:
1. Preserving original copyright notices in source files
2. Including LICENSE files in `_thirdparty/` subdirectories
3. Clearly documenting modifications in this technical document
4. Providing citation information in model wrappers

---

## Usage Examples

### Example 1: Train a Graph Classification Model

```python
from hoops_ai.ml import GraphClassification, FlowTrainer
from hoops_ai.dataset import DatasetLoader
from hoops_ai.cadaccess import HOOPSLoader

# Step 1: Create Flow Model
flowmodel = GraphClassification(
    num_classes=10,
    result_dir="./experiments/part_classification"
)

# Step 2: Prepare Dataset
dataset_loader = DatasetLoader(
    graph_files=["./encoded_data/graphs/*.bin"],
    label_files=["./encoded_data/labels/*.json"]
)
dataset_loader.split_data(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

# Step 3: Create Trainer
trainer = FlowTrainer(
    flowmodel=flowmodel,
    datasetLoader=dataset_loader,
    batch_size=32,
    num_workers=4,
    experiment_name="part_classifier_v1",
    accelerator='gpu',
    max_epochs=50,
    result_dir="./experiments"
)

# Step 4: Train Model
best_checkpoint = trainer.train()
print(f"Training complete! Best model: {best_checkpoint}")

# Step 5: Evaluate
trainer.test(trained_model_path=best_checkpoint)

# Step 6: Access Metrics
metrics = trainer.metrics_storage()
```

---

### Example 2: Inference on New CAD File

```python
from hoops_ai.ml import GraphNodeClassification, FlowInference
from hoops_ai.cadaccess import HOOPSLoader

# Step 1: Setup
cad_loader = HOOPSLoader()
flowmodel = GraphNodeClassification(num_classes=25)

# Step 2: Create Inference Pipeline
inference = FlowInference(
    cad_loader=cad_loader,
    flowmodel=flowmodel
)

# Step 3: Load Trained Model
inference.load_from_checkpoint("./experiments/best.ckpt")

# Step 4: Preprocess CAD File
batch = inference.preprocess("./new_parts/engine_block.step")

# Step 5: Predict
results = inference.predict_and_postprocess(batch)

# Step 6: Interpret Results
face_predictions = results['node_predictions']
face_confidences = results['node_probabilities']

print(f"Detected {len(face_predictions)} faces")
for i, (pred, conf) in enumerate(zip(face_predictions, face_confidences)):
    print(f"Face {i}: Class {pred} (confidence: {conf[pred]:.2%})")
```

---

### Example 3: Custom Encoding Strategy

If you need custom encoding logic, implement your own `FlowModel`:

```python
from hoops_ai.ml import FlowModel
from hoops_ai.cadencoder import BrepEncoder

class CustomFlowModel(FlowModel):
    def encode_cad_data(self, cad_file, cad_loader, storage):
        # Load CAD file
        model = cad_loader.create_from_file(cad_file)
        brep_encoder = BrepEncoder(model.get_brep(), storage)
        
        # Custom encoding logic
        brep_encoder.push_face_adjacency_graph()
        brep_encoder.push_facegrid(20, 20)  # Higher resolution
        brep_encoder.push_custom_features()  # Your custom method
        
        return (model.num_faces, model.num_edges)
    
    # Implement other abstract methods...
```

---

## Best Practices

### 1. Dataset Preparation

**Always validate your data before training:**
```python
# Run purify to identify problematic samples
trainer.purify(num_processes=4)

# Review error reports in chunk_errors/
# Remove problematic samples from dataset
```

---

### 2. Hyperparameter Tuning

**Start with default parameters, then tune:**
```python
# Default configuration
flowmodel = GraphNodeClassification(num_classes=25)

# After initial training, adjust based on validation metrics
flowmodel = GraphNodeClassification(
    num_classes=25,
    n_layers_encode=12,  # Deeper for complex datasets
    learning_rate=0.001,  # Lower LR for fine-tuning
    dropout=0.5  # Higher dropout if overfitting
)
```

---

### 3. Checkpoint Management

**Save multiple checkpoints during training:**
```python
checkpoint_callback = ModelCheckpoint(
    save_top_k=5,  # Keep top 5 models
    save_last=True,  # Always save last epoch
    monitor="eval_loss"
)
```

**Test with different checkpoints:**
```python
# Test best validation loss
trainer.test("./checkpoints/best.ckpt")

# Test last epoch (might be better for some metrics)
trainer.test("./checkpoints/last.ckpt")
```

---

### 4. Inference Optimization

**For production deployments:**
```python
# Move model to GPU if available
inference.model.to('cuda')

# Use model.eval() and torch.no_grad()
# (handled automatically by predict_and_postprocess)

# Batch multiple files if possible
files = ["part1.step", "part2.step", "part3.step"]
batches = [inference.preprocess(f) for f in files]
combined_batch = combine_batches(batches)  # Your custom logic
results = inference.predict_and_postprocess(combined_batch)
```

---

## Troubleshooting

### Common Issues

#### 1. Shape Mismatch Errors

**Symptom:** `RuntimeError: shape mismatch in face_uv_grids`

**Cause:** Inconsistent UV grid sampling between training and inference

**Solution:**
```python
# Ensure same grid sizes in both training and inference
brep_encoder.push_facegrid(10, 10)  # Use same values everywhere
brep_encoder.push_curvegrid(10)
```

---

#### 2. Out of Memory (OOM)

**Symptom:** `RuntimeError: CUDA out of memory`

**Solution:**
```python
# Reduce batch size
trainer = FlowTrainer(batch_size=16)  # Instead of 32

# Use gradient accumulation
trainer = FlowTrainer(accumulate_grad_batches=2)

# Enable mixed precision training
trainer = FlowTrainer(precision=16)
```

---

#### 3. Label Dimension Mismatch

**Symptom:** `AssertionError: Expected face-level labels, got graph-level`

**Cause:** Using node classification model with graph-level labels

**Solution:**
```python
# For node classification, ensure labels are per-face
label_storage.save_face_labels(mlTask, face_label_list)

# For graph classification, ensure labels are per-graph
label_storage.save_graph_label(mlTask, graph_label)
```

---

## Future Directions

The Flow Model architecture is **under active development**. Planned enhancements include:

1. **Multi-Framework Support:** Extend beyond PyTorch Lightning to support TensorFlow, JAX
2. **Streaming Inference:** Handle CAD files larger than RAM via streaming encoding
3. **Model Zoo:** Pre-trained models for common CAD tasks
4. **Distributed Training:** Multi-GPU and multi-node training support
5. **AutoML Integration:** Automatic hyperparameter tuning
6. **ONNX Export:** Export models for deployment in non-Python environments

---

## References

### Papers

1. **UV-Net:**  
   Jayaraman, P. K., Sanghi, A., Lambourne, J. G., Willis, K. D. D., Davies, T., Shayani, H., & Morris, N. (2021). UV-Net: Learning from Boundary Representations. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 11703-11712).

2. **BrepMFR:**  
   Zhang, S., Guan, Z., Jiang, H., Wang, X., & Tan, P. (2024). BrepMFR: Enhancing machining feature recognition in B-rep models through deep learning and domain adaptation. *Computer Aided Geometric Design*, 111, 102318.

### Repositories

- **UV-Net:** https://github.com/AutodeskAILab/UV-Net
- **BrepMFR:** https://github.com/zhangshuming0668/BrepMFR
- **HOOPS AI:** Contact Tech Soft 3D for repository access

### Related Documentation

- [Module Access & Encoder Documentation](./Module_Access_and_Encoder.md)
- [DataStorage Documentation](./DataStorage_Documentation.md)
- [Flow Documentation](./Flow_Documentation.md)
- [SchemaBuilder Documentation](./SchemaBuilder_Documentation.md)

---

## Conclusion

The Flow Model architecture provides a **unified interface** for integrating machine learning models with CAD data processing pipelines in HOOPS AI. By separating concerns between:

- **Data encoding** (how to process CAD files)
- **Model architecture** (what neural network to use)
- **Training infrastructure** (how to train at scale)
- **Inference deployment** (how to make predictions)

...developers can focus on their specific domain expertise while leveraging a robust, tested framework for the rest.

The architecture's **key innovation** is ensuring that the encoding logic used during training is automatically reused during inference, eliminating a common source of bugs in ML deployment.

While currently **EXPERIMENTAL**, the Flow Model pattern demonstrates the potential for standardized ML workflows in the CAD/CAM domain, and we welcome feedback from the community as it evolves.

---

**Document Version:** 1.0  
**Last Updated:** October 30, 2025  
**Status:** Experimental  
**Maintainer:** Tech Soft 3D ML Team
