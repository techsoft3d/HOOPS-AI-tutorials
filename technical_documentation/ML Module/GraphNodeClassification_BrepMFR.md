# GraphNodeClassification (BrepMFR) Documentation

## Overview

`GraphNodeClassification` is a HOOPS AI wrapper around the **BrepMFR** architecture for node-level (per-face) classification tasks. This implementation encapsulates the complete pipeline from CAD file to face-level predictions, following the `FlowModel` interface for seamless integration with HOOPS AI's training and inference infrastructure.

**Use Cases:**
- Machining feature recognition (holes, pockets, slots, chamfers)
- Face semantic segmentation
- Manufacturing process planning
- Design rule checking
- Feature-based similarity search

---

## Table of Contents

1. [Model Architecture](#model-architecture)
2. [Citation and Licensing](#citation-and-licensing)
3. [Initialization](#initialization)
4. [CAD Encoding Strategy](#cad-encoding-strategy)
5. [Integration with Flow Tasks](#integration-with-flow-tasks)
6. [Complete Example: CADSynth-AAG Dataset](#complete-example-cadsynth-aag-dataset)
7. [Training Workflow](#training-workflow)
8. [Inference Workflow](#inference-workflow)
9. [Hyperparameter Tuning](#hyperparameter-tuning)

---

## Model Architecture

### Original Paper

> **Zhang, S., Guan, Z., Jiang, H., Wang, X., & Tan, P. (2024).**  
> BrepMFR: Enhancing machining feature recognition in B-rep models through deep learning and domain adaptation.  
> *Computer Aided Geometric Design*, 111, 102318.  
> https://www.sciencedirect.com/science/article/abs/pii/S0167839624000529

### Architecture Description

BrepMFR converts B-rep models into graph representations for per-face classification:

**Graph Representation:**
- **Nodes:** Individual faces of the CAD model
- **Edges:** Adjacency relationships between faces (shared edges)
- **Node Features:** Rich geometric and topological encodings
- **Edge Features:** Curve geometry and inter-face relationships

**Neural Network Components:**
- **Transformer-based Encoder:** Multi-layer attention mechanism
- **Graph Attention Networks:** Aggregate neighbor information
- **MLP Classifier:** Per-node classification head

**Key Innovations:**
- **Local Geometric Encoding:** Face-level UV-grids capture surface shape
- **Global Topological Encoding:** Graph structure captures part-level context
- **Transfer Learning:** Two-step training from synthetic to real CAD models
- **Attention Mechanisms:** Focus on relevant geometric relationships

**Output:**
- Classification label for **each face** in the CAD model

### Original Applications

- Machining feature recognition in CAD/CAM workflows
- Recognizing highly intersecting features with complex geometries
- Automated process planning for CNC machining
- Design for manufacturability analysis

### GitHub Repository

https://github.com/zhangshuming0668/BrepMFR

---

## Citation and Licensing

### BibTeX Citation

When publishing research using this model, please cite the original paper:

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

### MIT License

BrepMFR is distributed under the **MIT License** by Zhang et al.

**HOOPS AI Modifications:**
- Implementation of `FlowModel` interface
- Integration with HOOPS AI storage system
- Adaptation for `FlowTrainer` and `FlowInference`
- Custom learning rate schedulers
- Error logging and debugging enhancements

**Original Authors:** Zhang Shuming and contributors  
**HOOPS AI Integration:** Tech Soft 3D  
**Location:** `src/hoops_ai/ml/_thirdparty/brepmfr/`

---

## Initialization

### Basic Usage

```python
from hoops_ai.ml.EXPERIMENTAL import GraphNodeClassification

# Create model with default parameters
flow_model = GraphNodeClassification(
    num_classes=25,
    result_dir="./results"
)
```

### Parameters

#### Model Architecture Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_classes` | int | 25 | Number of feature classes (e.g., hole, pocket, slot) |
| `n_layers_encode` | int | 8 | Number of Transformer encoder layers |
| `dim_node` | int | 256 | Node embedding dimension |
| `d_model` | int | 512 | Transformer model dimension |
| `n_heads` | int | 32 | Number of attention heads |
| `dropout` | float | 0.3 | Classifier dropout rate |
| `attention_dropout` | float | 0.3 | Attention mechanism dropout |
| `act_dropout` | float | 0.3 | Activation layer dropout |

#### Training Hyperparameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `learning_rate` | float | 0.002 | Initial learning rate |
| `optimizer_betas` | Tuple[float, float] | (0.99, 0.999) | AdamW optimizer betas |
| `scheduler_factor` | float | 0.5 | LR reduction factor |
| `scheduler_patience` | int | 5 | Patience for LR scheduler |
| `scheduler_threshold` | float | 1e-4 | Threshold for scheduler |
| `scheduler_min_lr` | float | 1e-6 | Minimum learning rate |
| `scheduler_cooldown` | int | 2 | Cooldown period after LR reduction |
| `max_warmup_steps` | int | 5000 | Warmup steps for learning rate |

#### Logging and Output

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `log_file` | str | `'training_errors.log'` | Error logging file path |
| `result_dir` | str | None | Results output directory |
| `generate_stream_cache_for_visu` | bool | False | Generate visualization cache |

### Advanced Configuration

```python
flow_model = GraphNodeClassification(
    # Architecture
    num_classes=25,
    n_layers_encode=12,     # Deeper network for complex features
    dim_node=512,           # Larger embeddings
    d_model=1024,
    n_heads=16,
    
    # Regularization
    dropout=0.5,            # Higher dropout if overfitting
    attention_dropout=0.4,
    act_dropout=0.4,
    
    # Training
    learning_rate=0.001,    # Lower LR for fine-tuning
    max_warmup_steps=10000, # Longer warmup for stability
    
    # Output
    result_dir="./experiments/machining_features",
    generate_stream_cache_for_visu=True  # Enable for debugging
)
```

---

## CAD Encoding Strategy

### Encoding Pipeline

The `GraphNodeClassification` model uses a **richer** encoding strategy compared to `GraphClassification`:

```python
def encode_cad_data(self, cad_file: str, cad_loader: CADLoader, storage: DataStorage):
    # 1. Configure CAD loading
    general_options = cad_loader.get_general_options()
    general_options["read_feature"] = True 
    general_options["read_solid"] = True 
    
    # 2. Load model
    model = cad_loader.create_from_file(cad_file)
    
    # 3. Configure BREP with UV computation
    hoopstools = HOOPSTools()
    brep_options = hoopstools.brep_options()
    brep_options["force_compute_uv"] = True       
    brep_options["force_compute_3d"] = True 
    hoopstools.adapt_brep(model, brep_options)
    
    # 4. Encode features (MORE COMPREHENSIVE than GraphClassification)
    brep_encoder = BrepEncoder(model.get_brep(body_index=0), storage)
    
    # Graph structure
    brep_encoder.push_face_adjacency_graph()
    
    # Node features (faces)
    brep_encoder.push_face_attributes()
    brep_encoder.push_facegrid(10, 10)
    
    # Edge features
    brep_encoder.push_edge_attributes()
    brep_encoder.push_curvegrid(10)
    
    # ADDITIONAL TOPOLOGICAL FEATURES (unique to BrepMFR)
    brep_encoder.push_extended_adjacency()
    brep_encoder.push_face_neighbors_count()
    brep_encoder.push_face_pair_edges_path(16)
    
    # GEOMETRIC RELATIONSHIP FEATURES
    brep_encoder.push_average_face_pair_angle_histograms(5, 64)
    brep_encoder.push_average_face_pair_distance_histograms(5, 64)
```

### Feature Specifications

**Node Features (Per Face):**
1. **UV-grids:** `(10, 10, 7)` - Surface geometry
2. **Face attributes:** Surface type, area, loop count
3. **Neighbor count:** Number of adjacent faces
4. **Angle histograms:** Distribution of dihedral angles with neighbors
5. **Distance histograms:** Distribution of distances to neighbor centroids

**Edge Features (Per Face-Face Connection):**
1. **U-grids:** `(10, 6)` - Shared edge curve geometry
2. **Edge attributes:** Curve type, length, dihedral angle
3. **Path information:** Shortest path metrics

**Graph Structure:**
- **Nodes:** All faces in the CAD model
- **Edges:** Face adjacency (two faces share an edge)
- **Extended Adjacency:** Multi-hop neighbor relationships

### Mathematical Representation

For each face $f_i$ with neighbors $\mathcal{N}(f_i)$:

**Node Embedding:**
$$
\mathbf{h}_i^{(0)} = \text{Concat}\left(
    \text{CNN}_{2D}(\mathbf{UV}_i),\, 
    \mathbf{a}_i,\, 
    \text{Hist}(\{\theta_{ij}\}_{j \in \mathcal{N}(i)}),\,
    \text{Hist}(\{d_{ij}\}_{j \in \mathcal{N}(i)})
\right)
$$

where:
- $\mathbf{UV}_i$ is the face UV-grid
- $\mathbf{a}_i$ are face attributes (type, area, etc.)
- $\theta_{ij}$ is the dihedral angle between faces $i$ and $j$
- $d_{ij}$ is the distance between face centroids

**Transformer-based Message Passing:**
$$
\mathbf{h}_i^{(\ell+1)} = \text{TransformerLayer}^{(\ell)}\left(
    \mathbf{h}_i^{(\ell)},\, 
    \{\mathbf{h}_j^{(\ell)}\}_{j \in \mathcal{N}(i)},\,
    \{\mathbf{e}_{ij}\}_{j \in \mathcal{N}(i)}
\right)
$$

**Per-Face Classification:**
$$
\hat{y}_i = \text{softmax}\left(\text{MLP}\left(\mathbf{h}_i^{(L)}\right)\right)
$$

---

## Integration with Flow Tasks

### Overview

Like `GraphClassification`, the `GraphNodeClassification` model integrates with HOOPS AI's Flow framework. However, there are **key differences** due to node-level labels.

### Pattern: Wrapping FlowModel Methods

```python
from hoops_ai.flowmanager import flowtask

# 1. Create FlowModel instance
flow_model = GraphNodeClassification(num_classes=25, result_dir="./results")

# 2. Wrap encode_cad_data() in a Flow task
@flowtask.transform(
    name="advanced_cad_encoder",
    inputs=["cad_file", "cad_loader", "storage"],
    outputs=["face_count", "edge_count"]
)
def my_encoder(cad_file: str, cad_loader, storage):
    # Call the FlowModel's encoding method
    face_count, edge_count = flow_model.encode_cad_data(cad_file, cad_loader, storage)
    
    # NOTE: For node classification, labels are PER FACE
    # Optionally process face-level labels here
    # flow_model.encode_label_data(label_storage, storage)
    
    # Convert to graph
    flow_model.convert_encoded_data_to_graph(storage, graph_handler, filename)
    
    return face_count, edge_count
```

### Key Difference: Node-Level Labels

Unlike graph classification (one label per file), node classification requires **one label per face**:

```python
# Graph classification (GraphClassification)
storage.save_data("file_label", np.array([3]))  # Single label

# Node classification (GraphNodeClassification)
storage.save_data("face_labels", np.array([0, 1, 1, 2, 0, ...]))  # Label per face
```

---

## Complete Example: CADSynth-AAG Dataset

This example demonstrates processing the CADSynth-AAG segmentation dataset (162k models) using `GraphNodeClassification` integrated with HOOPS AI Flows.

### Dataset Structure

```
Cadsynth_aag/
└── step/
    ├── model_0001.step
    ├── model_0002.step
    └── ...  (162,000 models)
```

**Note:** This dataset is for **unsupervised** preprocessing. Labels would be loaded separately during training.

### Complete Implementation

```python
"""
CADSynth-AAG Segmentation Preprocessing using GraphNodeClassification FlowModel
"""

import pathlib
from typing import Tuple, List

# Flow framework imports
from hoops_ai.flowmanager import flowtask
import hoops_ai
from hoops_ai.cadaccess import HOOPSLoader, CADLoader
from hoops_ai.storage import (
    DataStorage, 
    MLStorage, 
    CADFileRetriever, 
    LocalStorageProvider,
    DGLGraphStoreHandler
)
from hoops_ai.dataset import DatasetExplorer

# FlowModel import
from hoops_ai.ml.EXPERIMENTAL import GraphNodeClassification
from hoops_ai.storage.label_storage import LabelStorage

# ==================================================================================
# CONFIGURATION
# ==================================================================================

flows_inputdir = r"C:\Temp\Cadsynth_aag\step"
flows_outputdir = str(pathlib.Path(flows_inputdir))
datasources_dir = str(flows_inputdir)

# ==================================================================================
# FLOWMODEL INSTANTIATION
# ==================================================================================

flowName = "cadsynth_aag_162k_flowtask"
flow_model = GraphNodeClassification(
    num_classes=25,  # Will be updated when labels are available
    result_dir=str(pathlib.Path(flows_inputdir).joinpath("flows").joinpath(flowName))
)

# ==================================================================================
# FLOW TASK DEFINITIONS
# ==================================================================================

@flowtask.extract(
    name="gather_cad_files_to_be_treated",
    inputs=["cad_datasources"],
    outputs=["cad_dataset"]
)
def my_demo_gatherer(source: str) -> List[str]:
    """
    Gather all CAD files from the CADSynth-AAG dataset directory.
    """
    cad_formats = [".stp", ".step"]
    local_provider = LocalStorageProvider(directory_path=source)
    retriever = CADFileRetriever(
        storage_provider=local_provider,
        formats=cad_formats
    )
    return retriever.get_file_list()


@flowtask.transform(
    name="advanced_cad_encoder",
    inputs=["cad_file", "cad_loader", "storage"],
    outputs=["face_count", "edge_count"]
)
def my_demo_encoder(cad_file: str, cad_loader: HOOPSLoader, storage: DataStorage) -> Tuple[int, int]:
    """
    Encode CAD data using GraphNodeClassification FlowModel.
    
    This task wraps the FlowModel's encode_cad_data() method and adds:
    1. Rich topological feature extraction (BrepMFR-specific)
    2. Graph conversion for ML training
    3. Optional label processing (commented out for unsupervised preprocessing)
    """
    
    # ===== CALL FLOWMODEL METHOD =====
    face_count, edge_count = flow_model.encode_cad_data(cad_file, cad_loader, storage)
    # =================================
    
    # Optional: Encode labels if available
    # This would be used if you have a LabelStorage with face-level annotations
    # flow_model.encode_label_data(label_storage, storage)
    
    # Convert encoded data to DGL graph file
    location = pathlib.Path(storage.get_file_path("."))
    dgl_output_path = pathlib.Path(location.parent.parent / "dgl" / f"{location.stem}.ml")
    dgl_output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # ===== CALL FLOWMODEL METHOD =====
    flow_model.convert_encoded_data_to_graph(storage, DGLGraphStoreHandler(), str(dgl_output_path))
    # =================================
    
    return face_count, edge_count

# ==================================================================================
# FLOW ORCHESTRATION
# ==================================================================================

def main():
    """
    Execute the CADSynth-AAG preprocessing flow using GraphNodeClassification FlowModel.
    """
    # Create Flow with tasks
    cad_flow = hoops_ai.create_flow(
        name=flowName,
        tasks=[
            my_demo_gatherer,      # Gather CAD files
            my_demo_encoder        # Encode using FlowModel
        ],
        max_workers=16,  # 16 parallel workers
        flows_outputdir=str(flows_outputdir),
        ml_task="Machine Features Classification with Auto-Schema",
    )
    
    # Execute Flow
    output, dict_data, flow_file = cad_flow.process(
        inputs={'cad_datasources': [datasources_dir]}
    )
    
    # Print summary
    print(output.summary())
    
    # Explore dataset
    explorer = DatasetExplorer(flow_output_file=str(flow_file))
    explorer.print_table_of_contents()
    
    # Filter files with medium face count
    import time
    start_time = time.time()
    
    facecount_is_medium = lambda ds: ds['num_nodes'] > 40
    
    try:
        filelist = explorer.get_file_list(group="graph", where=facecount_is_medium)
        print(f"Filtering completed in {(time.time() - start_time):.2f} seconds")
        print(f"Files with num_nodes > 40: {len(filelist)}")
        print(filelist[:5] if len(filelist) > 5 else filelist)
    except Exception as e:
        print(f"Error during filtering: {e}")
        print("Filtering functionality is having issues, but dataset export was successful!")


if __name__ == "__main__":
    main()
```

### Key Integration Points

#### 1. FlowModel Instantiation

```python
# Instantiate ONCE at module level
flow_model = GraphNodeClassification(num_classes=25, result_dir="./results")
```

#### 2. Encoding Task Wrapper

```python
@flowtask.transform(...)
def my_demo_encoder(cad_file, cad_loader, storage):
    # Call FlowModel method directly
    face_count, edge_count = flow_model.encode_cad_data(cad_file, cad_loader, storage)
    
    # Optional: Add face-level labels
    # flow_model.encode_label_data(label_storage, storage)
    
    # Call another FlowModel method
    flow_model.convert_encoded_data_to_graph(storage, graph_handler, filename)
    
    return face_count, edge_count
```

#### 3. Rich Feature Encoding

The key advantage of `GraphNodeClassification` is the **richer feature set**:

```python
# Inside flow_model.encode_cad_data():
brep_encoder.push_face_adjacency_graph()           # Standard
brep_encoder.push_facegrid(10, 10)                 # Standard
brep_encoder.push_extended_adjacency()             # BrepMFR-specific
brep_encoder.push_face_neighbors_count()           # BrepMFR-specific
brep_encoder.push_average_face_pair_angle_histograms(5, 64)  # BrepMFR-specific
brep_encoder.push_average_face_pair_distance_histograms(5, 64)  # BrepMFR-specific
```

### Output Structure

```
Cadsynth_aag/
└── flows/
    └── cadsynth_aag_162k_flowtask/
        ├── encoded/           # Individual .data files (Zarr format)
        │   ├── model_0001.data
        │   ├── model_0002.data
        │   └── ...
        ├── dgl/              # DGL graph files for ML training
        │   ├── model_0001.ml
        │   ├── model_0002.ml
        │   └── ...
        ├── info/             # Metadata
        └── flow_output.json  # Flow execution summary
```

---

## Training Workflow

### Step 1: Preprocess Dataset

Use the Flow example above to create ML-ready graph files.

### Step 2: Prepare Face-Level Labels

**Important:** Node classification requires labels for **each face** in each model.

```python
# Example: Load face labels from annotation file
import json

def load_face_labels(cad_file, annotation_file):
    """
    Load face-level labels from annotation file.
    
    Returns:
        np.ndarray: Label for each face [face_0_label, face_1_label, ...]
    """
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    
    # Extract face labels (format depends on your annotation system)
    face_labels = np.array(annotations['face_labels'])
    return face_labels
```

### Step 3: Load Dataset

```python
from hoops_ai.dataset import DatasetLoader

# Load preprocessed graphs
# Note: For node classification, labels are stored IN the graph files
dataset_loader = DatasetLoader(
    graph_files=["./flows/cadsynth_aag_162k_flowtask/dgl/*.ml"]
    # No separate label files - labels are per-face in graph
)

# Split into train/val/test
dataset_loader.split_data(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
```

### Step 4: Create Trainer

```python
from hoops_ai.ml import FlowTrainer

trainer = FlowTrainer(
    flowmodel=flow_model,  # Same instance used in Flow
    datasetLoader=dataset_loader,
    batch_size=16,  # Smaller batch size due to per-face predictions
    num_workers=4,
    experiment_name="machining_feature_recognition",
    accelerator='gpu',
    devices=1,
    max_epochs=100,
    result_dir="./experiments"
)
```

### Step 5: Train Model

```python
# Train and get best checkpoint
best_checkpoint = trainer.train()
print(f"Training complete! Best model: {best_checkpoint}")

# Evaluate on test set
trainer.test(trained_model_path=best_checkpoint)

# Access metrics
metrics = trainer.metrics_storage()
train_loss = metrics.get("train_loss")
val_accuracy = metrics.get("val_node_accuracy")  # Note: node-level accuracy
```

### Step 6: Monitor Training

```bash
# Launch TensorBoard
tensorboard --logdir=./experiments/ml_output/machining_feature_recognition/
```

**Metrics to Monitor:**
- `train_loss`: Per-face cross-entropy loss
- `val_node_accuracy`: Percentage of correctly classified faces
- `val_per_class_accuracy`: Accuracy for each feature class

---

## Inference Workflow

### Single File Inference

```python
from hoops_ai.ml import FlowInference
from hoops_ai.cadaccess import HOOPSLoader

# Setup
cad_loader = HOOPSLoader()
inference = FlowInference(
    cad_loader=cad_loader,
    flowmodel=flow_model,  # Same instance used in training
    log_file='inference_errors.log'
)

# Load trained model
inference.load_from_checkpoint("./experiments/best.ckpt")

# Predict on new CAD file
batch = inference.preprocess("new_part.step")
predictions = inference.predict_and_postprocess(batch)

# Interpret results (PER-FACE predictions)
face_predictions = predictions['node_predictions']  # [N_faces]
face_confidences = predictions['node_probabilities']  # [N_faces, N_classes]
num_faces = predictions['num_faces']

print(f"Model has {num_faces} faces")
for i in range(num_faces):
    pred_class = face_predictions[i]
    confidence = face_confidences[i][pred_class]
    print(f"Face {i}: Class {pred_class} (confidence: {confidence:.2%})")
```

### Visualizing Face Predictions

```python
# Map predictions to CAD model for visualization
feature_names = {
    0: "Base",
    1: "Hole",
    2: "Pocket",
    3: "Slot",
    # ... etc
}

# Create color-coded face map
face_colors = []
for pred in face_predictions:
    feature = feature_names[pred]
    face_colors.append(get_color_for_feature(feature))

# Export for visualization
# (Use HOOPS Communicator or other CAD viewer)
```

---

## Hyperparameter Tuning

### Architecture Hyperparameters

```python
# Baseline (default)
flow_model = GraphNodeClassification(
    num_classes=25,
    n_layers_encode=8,
    dim_node=256,
    d_model=512,
    n_heads=32,
)

# Larger model for complex features
flow_model = GraphNodeClassification(
    num_classes=25,
    n_layers_encode=12,     # Deeper
    dim_node=512,           # Larger embeddings
    d_model=1024,           # Larger Transformer
    n_heads=16,             # More attention heads
)

# Smaller model for faster training
flow_model = GraphNodeClassification(
    num_classes=25,
    n_layers_encode=4,
    dim_node=128,
    d_model=256,
    n_heads=8,
)
```

### Regularization

```python
# Default regularization
flow_model = GraphNodeClassification(
    dropout=0.3,
    attention_dropout=0.3,
    act_dropout=0.3,
)

# Stronger regularization if overfitting
flow_model = GraphNodeClassification(
    dropout=0.5,
    attention_dropout=0.5,
    act_dropout=0.5,
)

# Weaker regularization if underfitting
flow_model = GraphNodeClassification(
    dropout=0.1,
    attention_dropout=0.1,
    act_dropout=0.1,
)
```

### Learning Rate Schedule

```python
# Default schedule
flow_model = GraphNodeClassification(
    learning_rate=0.002,
    scheduler_factor=0.5,      # Reduce LR by 50%
    scheduler_patience=5,      # After 5 epochs without improvement
    scheduler_min_lr=1e-6,     # Don't go below this
    max_warmup_steps=5000,
)

# Aggressive schedule (faster convergence, risk instability)
flow_model = GraphNodeClassification(
    learning_rate=0.005,
    scheduler_factor=0.3,
    scheduler_patience=3,
    max_warmup_steps=2000,
)

# Conservative schedule (slower, more stable)
flow_model = GraphNodeClassification(
    learning_rate=0.001,
    scheduler_factor=0.7,
    scheduler_patience=10,
    max_warmup_steps=10000,
)
```

### Training Hyperparameters

```python
trainer = FlowTrainer(
    flowmodel=flow_model,
    
    # Batch size (lower for node classification due to memory)
    batch_size=8,  # Try: 4, 8, 16, 32
    
    # Gradient clipping (important for stability)
    gradient_clip_val=1.0,  # Try: 0.5, 1.0, 2.0
    
    # Epochs
    max_epochs=200,
    
    # Device
    accelerator='gpu',
    devices=1,
)
```

---

## Performance Optimization

### GPU Memory Management

Node classification uses more memory than graph classification (predictions for every face):

```python
# Reduce batch size
trainer = FlowTrainer(batch_size=4)

# Enable gradient accumulation
trainer = FlowTrainer(
    batch_size=4,
    accumulate_grad_batches=4  # Effective batch size: 16
)

# Use mixed precision
trainer = FlowTrainer(precision=16)
```

### Parallel Preprocessing

```python
# Increase workers for Flow preprocessing
cad_flow = hoops_ai.create_flow(
    name="my_flow",
    tasks=[my_gatherer, my_encoder],
    max_workers=32,  # More workers for large datasets
    flows_outputdir="./output"
)
```

### Feature Engineering

```python
# Adjust UV grid resolution
def my_encoder(cad_file, cad_loader, storage):
    face_count, edge_count = flow_model.encode_cad_data(...)
    
    # Override with custom resolution
    brep_encoder = BrepEncoder(model.get_brep(), storage)
    brep_encoder.push_facegrid(15, 15)  # Higher resolution
    brep_encoder.push_curvegrid(15)
```

---

## Troubleshooting

### Issue: Class Imbalance

**Symptom:** Model predicts only majority class

**Cause:** Some feature classes (e.g., "base" faces) are much more common than others (e.g., "chamfer")

**Solution:**
```python
# Option 1: Use weighted loss (requires model modification)
class_weights = compute_class_weights(train_labels)

# Option 2: Oversample minority classes during data loading
dataset_loader.set_sampling_weights(class_weights)

# Option 3: Data augmentation (CAD-specific transforms)
```

---

### Issue: Per-Face Predictions Don't Form Valid Features

**Symptom:** Adjacent faces predicted as different feature classes, but should be same feature

**Cause:** Model lacks global context or training data has inconsistent annotations

**Solution:**
```python
# Post-processing: Merge predictions using connected component analysis
def merge_face_predictions(predictions, adjacency_graph):
    """
    Merge adjacent faces with same predicted class into features.
    """
    features = []
    visited = set()
    
    for face_id in range(len(predictions)):
        if face_id in visited:
            continue
        
        # BFS to find all connected faces with same class
        feature_faces = bfs_same_class(face_id, predictions, adjacency_graph)
        features.append(feature_faces)
        visited.update(feature_faces)
    
    return features
```

---

### Issue: Low Accuracy on Real CAD Models

**Symptom:** Good accuracy on synthetic data, poor on real CAD

**Cause:** Domain gap between training and inference data

**Solution:**
```python
# Use BrepMFR's two-step training strategy:
# 1. Pre-train on synthetic data (e.g., CADSynth-AAG)
trainer1 = FlowTrainer(...)
checkpoint1 = trainer1.train()

# 2. Fine-tune on real CAD data
flow_model_finetuned = GraphNodeClassification.load_from_checkpoint(checkpoint1)
trainer2 = FlowTrainer(
    flowmodel=flow_model_finetuned,
    learning_rate=0.0001,  # Lower LR for fine-tuning
)
checkpoint2 = trainer2.train()
```

---

## Advanced Usage

### Custom Feature Types

Extend the model for domain-specific features:

```python
# Define custom feature classes
CUSTOM_FEATURES = {
    0: "Base",
    1: "Through Hole",
    2: "Blind Hole",
    3: "Counterbore",
    4: "Countersink",
    5: "Rectangular Pocket",
    6: "Circular Pocket",
    7: "Slot",
    8: "Chamfer",
    9: "Fillet",
    # ... add your domain-specific features
}

flow_model = GraphNodeClassification(
    num_classes=len(CUSTOM_FEATURES),
    result_dir="./custom_features"
)
```

### Multi-Task Learning

Train for multiple tasks simultaneously:

```python
# Task 1: Feature recognition
# Task 2: Manufacturing difficulty
# Task 3: Tool accessibility

# (Requires model architecture modification)
```

---

## Comparison with GraphClassification

| Aspect | GraphClassification | GraphNodeClassification |
|--------|---------------------|-------------------------|
| **Task** | Whole-model classification | Per-face classification |
| **Output** | Single label per CAD file | Label for each face |
| **Features** | Basic UV-grids, edges | + Extended adjacency, histograms |
| **Architecture** | CNN + GNN | Transformer + GNN |
| **Memory** | Lower | Higher (per-face predictions) |
| **Training Time** | Faster | Slower |
| **Use Cases** | Part categorization | Feature recognition |
| **Batch Size** | 32-64 typical | 4-16 typical |

---

## Related Documentation

- [Flow Model Architecture](./FlowModel_Architecture.md)
- [GraphClassification (UV-Net)](./GraphClassification_UVNet.md)
- [Module Access & Encoder Documentation](./Module_Access_and_Encoder.md)
- [Flow Documentation](./Flow_Documentation.md)

---

## Conclusion

`GraphNodeClassification` provides a production-ready wrapper around BrepMFR for CAD face-level classification tasks. Its rich feature encoding and Transformer-based architecture make it particularly suitable for complex machining feature recognition tasks.

**Key Takeaways:**
1. Use for **per-face** classification (feature recognition, segmentation)
2. Requires more memory than graph classification
3. Benefits from richer topological features
4. Supports two-step training (synthetic → real)
5. Post-processing can merge faces into complete features

**When to Use:**
- ✅ Machining feature recognition
- ✅ Face semantic segmentation  
- ✅ Manufacturing process planning
- ✅ Tasks requiring local geometric reasoning

**When to Use GraphClassification Instead:**
- ✅ Whole-part classification
- ✅ Design style recognition
- ✅ Part categorization
- ✅ Faster inference required

---

**Document Version:** 1.0  
**Last Updated:** October 30, 2025  
**Status:** Experimental  
**Maintainer:** Tech Soft 3D ML Team
