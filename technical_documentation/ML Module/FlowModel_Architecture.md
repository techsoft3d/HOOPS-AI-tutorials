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

## Available Implementations

HOOPS AI provides two concrete implementations of the `FlowModel` interface, each wrapping a state-of-the-art open-source architecture:

### 1. GraphClassification (UV-Net)

**Use Case:** Graph-level classification (e.g., part classification, shape categorization)

**Documentation:** [GraphClassification_UVNet.md](./GraphClassification_UVNet.md)

**Key Features:**
- Whole-model classification
- 2D CNNs on face UV-grids
- 1D CNNs on edge U-grids
- Ideal for: Part type identification, shape similarity

---

### 2. GraphNodeClassification (BrepMFR)

**Use Case:** Node-level classification (e.g., machining feature recognition, face segmentation)

**Documentation:** [GraphNodeClassification_BrepMFR.md](./GraphNodeClassification_BrepMFR.md)

**Key Features:**
- Per-face classification
- Transformer-based GNN architecture
- Rich topological feature encoding
- Ideal for: Feature recognition, semantic segmentation

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

**Citation:**
```bibtex
@inproceedings{jayaraman2021uvnet,
  title={UV-Net: Learning from Boundary Representations},
  author={Jayaraman, Pradeep Kumar and Sanghi, Aditya and Lambourne, Joseph G and Willis, Karl DD and Davies, Thomas and Shayani, Hooman and Morris, Nigel},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11703--11712},
  year={2021}
}
```

---

#### 2. BrepMFR (Zhang et al.)

**Source:** https://github.com/zhangshuming0668/BrepMFR  
**License:** MIT License  
**Authors:** Zhang, S., Guan, Z., Jiang, H., Wang, X., & Tan, P.  
**Year:** 2024  
**Location:** `src/hoops_ai/ml/_thirdparty/brepmfr/`

**Citation:**
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

### Attribution and Compliance

**Tech Soft 3D Stance on Third-Party Code:**

While HOOPS AI provides convenient wrappers (`GraphClassification`, `GraphNodeClassification`) to integrate these architectures into the Flow Model framework, **the original authors retain full credit for their pioneering work**. 

Our modifications are limited to:
1. **Interface Adaptation:** Implementing the `FlowModel` abstract interface
2. **Storage Integration:** Connecting to HOOPS AI's data storage system
3. **Training Infrastructure:** Enabling use with `FlowTrainer` and `FlowInference`

**We do NOT claim authorship of the underlying ML architectures.** Users of HOOPS AI should cite the original papers when publishing results using these models.

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
3. Clearly documenting modifications in technical documents
4. Providing citation information in model wrappers

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

---

### 4. Inference Optimization

**For production deployments:**
```python
# Move model to GPU if available
inference.model.to('cuda')

# Batch multiple files if possible
files = ["part1.step", "part2.step", "part3.step"]
batches = [inference.preprocess(f) for f in files]
```

---

## Troubleshooting

### Common Issues

#### 1. Shape Mismatch Errors

**Symptom:** `RuntimeError: shape mismatch in face_uv_grids`

**Cause:** Inconsistent UV grid sampling between training and inference

**Solution:** Ensure same grid sizes in both training and inference
```python
brep_encoder.push_facegrid(10, 10)  # Use same values everywhere
```

---

#### 2. Out of Memory (OOM)

**Symptom:** `RuntimeError: CUDA out of memory`

**Solution:**
```python
# Reduce batch size
trainer = FlowTrainer(batch_size=16)

# Use gradient accumulation
trainer = FlowTrainer(accumulate_grad_batches=2)

# Enable mixed precision training
trainer = FlowTrainer(precision=16)
```

---

#### 3. Label Dimension Mismatch

**Symptom:** `AssertionError: Expected face-level labels, got graph-level`

**Cause:** Using node classification model with graph-level labels

**Solution:** Ensure label granularity matches model type
```python
# For node classification: labels per face
label_storage.save_face_labels(mlTask, face_label_list)

# For graph classification: labels per graph
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

## Related Documentation

- [GraphClassification (UV-Net) Documentation](./GraphClassification_UVNet.md)
- [GraphNodeClassification (BrepMFR) Documentation](./GraphNodeClassification_BrepMFR.md)
- [Module Access & Encoder Documentation](./Module_Access_and_Encoder.md)
- [DataStorage Documentation](./DataStorage_Documentation.md)
- [Flow Documentation](./Flow_Documentation.md)

---

## Conclusion

The Flow Model architecture provides a **unified interface** for integrating machine learning models with CAD data processing pipelines in HOOPS AI. By separating concerns between data encoding, model architecture, training infrastructure, and inference deployment, developers can focus on their specific domain expertise while leveraging a robust, tested framework.

The architecture's **key innovation** is ensuring that the encoding logic used during training is automatically reused during inference, eliminating a common source of bugs in ML deployment.

While currently **EXPERIMENTAL**, the Flow Model pattern demonstrates the potential for standardized ML workflows in the CAD/CAM domain, and we welcome feedback from the community as it evolves.


