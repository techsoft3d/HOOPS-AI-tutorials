# GraphClassification - Graph-Level Classifier Documentation

## Overview

`GraphClassification` is a HOOPS AI implementation of a graph-level classification model for CAD data. This implementation encapsulates the complete pipeline from CAD file to graph-level predictions, following the `FlowModel` interface for seamless integration with HOOPS AI's training and inference infrastructure.

**Use Cases:**
- Part type classification (e.g., bearings, bolts, brackets)
- Shape categorization
- Design style recognition
- Manufacturing process selection

**Note:** This implementation is based on a third-party architecture. For complete attribution and citation information, see [Acknowledgements.md](./Acknowledgements.md).

---

## Table of Contents

1. [Model Architecture](#model-architecture)
2. [Initialization](#initialization)
3. [CAD Encoding Strategy](#cad-encoding-strategy)
4. [Integration with Flow Tasks](#integration-with-flow-tasks)
5. [Complete Example: FABWAVE Dataset](#complete-example-fabwave-dataset)
6. [Training Workflow](#training-workflow)
7. [Inference Workflow](#inference-workflow)
8. [Hyperparameter Tuning](#hyperparameter-tuning)

---

## Model Architecture

### Overview

The `GraphClassification` model operates directly on Boundary Representation (B-rep) data from 3D CAD models using a CNN+GNN approach:

**Geometric Encoding:**
- **Face Geometry:** Discretized sample points sampled on face surfaces
- **Edge Geometry:** 1D U-grids along edge curves

**Neural Network Components:**
- **2D CNNs:** Applied to face discretization samples to extract surface features
- **1D CNNs:** Applied to edge U-grids to extract curve features
- **Graph Neural Networks:** Aggregate topological information via face-adjacency graph

**Topology Representation:**
- **Nodes:** Individual faces of the CAD model
- **Edges:** Adjacency relationships between faces
- **Node Features:** Encoded face discretization samples
- **Edge Features:** Encoded edge U-grids

**Output:**
- Single classification label for the entire CAD model

### Technology Details

This implementation is based on a state-of-the-art architecture for learning from boundary representations. For complete technical details, original paper citation, and licensing information, please refer to [Acknowledgements.md](./Acknowledgements.md).

---

## Initialization

### Basic Usage

```python
from hoops_ai.ml.EXPERIMENTAL import GraphClassification

# Create model with default parameters
flow_model = GraphClassification(
    num_classes=10,
    result_dir="./results"
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_classes` | int | 10 | Number of classification categories |
| `result_dir` | str | None | Directory for saving results and metrics |
| `log_file` | str | `'cnn_graph_training_errors.log'` | Path to error logging file |
| `generate_stream_cache_for_visu` | bool | False | Generate visualization cache for debugging |

### Advanced Configuration

```python
flow_model = GraphClassification(
    num_classes=45,  # FABWAVE dataset has 45 classes
    result_dir="./experiments/part_classification",
    log_file="training_errors.log",
    generate_stream_cache_for_visu=False
)
```

---

## CAD Encoding Strategy

### Encoding Pipeline

The `GraphClassification` model uses the following encoding strategy in its `encode_cad_data()` method:

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
    
    # 4. Encode features
    brep_encoder = BrepEncoder(model.get_brep(body_index=0), storage)
    
    # Graph structure
    brep_encoder.push_face_adjacency_graph()
    
    # Node features (faces)
    brep_encoder.push_face_attributes()
    brep_encoder.push_face_discretization(pointsamples=25)  # Sample points per face
    
    # Edge features
    brep_encoder.push_edge_attributes()
    brep_encoder.push_curvegrid(10)  # 10 points along edge
    
    # Additional topological features
    brep_encoder.push_face_pair_edges_path(16)
```

### Feature Specifications

**Node Features (Face Discretization):**
- **Shape:** `(pointsamples, 7)` where pointsamples is typically 25
- **Components:** `(x, y, z, nx, ny, nz, visibility)`
- **Encoding:** 2D CNN processes each face's discretized sample points

**Edge Features (Edge U-grids):**
- **Shape:** `(10, 6)`
- **Components:** 3D points and tangent vectors along edge curve
- **Encoding:** 1D CNN processes each edge's U-grid

**Graph Structure:**
- **Nodes:** Faces of the CAD model
- **Edges:** Face-face adjacency (shared edges)

### Mathematical Representation

For each face $f_i$:

$$
\mathbf{X}_{f_i} = \text{CNN}_{2D}\left(\mathbf{S}_{f_i}^{n_{\text{samples}} \times 7}\right) \in \mathbb{R}^{d_{\text{face}}}
$$

For each edge $e_{ij}$ between faces $f_i$ and $f_j$:

$$
\mathbf{X}_{e_{ij}} = \text{CNN}_{1D}\left(\mathbf{U}_{e_{ij}}^{10 \times 6}\right) \in \mathbb{R}^{d_{\text{edge}}}
$$

Graph classification via message passing:

$$
\hat{y} = \text{MLP}\left(\text{READOUT}\left(\text{GNN}(G, \{\mathbf{X}_{f_i}\}, \{\mathbf{X}_{e_{ij}}\})\right)\right)
$$

---

## Integration with Flow Tasks

### Overview

The `GraphClassification` model integrates seamlessly with HOOPS AI's Flow framework via the `@flowtask` decorator pattern. This allows you to wrap FlowModel methods inside Flow tasks for batch processing of CAD datasets.

### Pattern: Wrapping FlowModel Methods

The key insight is to **instantiate the FlowModel once** at the module level, then **call its methods** inside decorated Flow tasks:

```python
from hoops_ai.flowmanager import flowtask

# 1. Create FlowModel instance
flow_model = GraphClassification(num_classes=45, result_dir="./results")

# 2. Wrap encode_cad_data() in a Flow task
@flowtask.transform(
    name="advanced_cad_encoder",
    inputs=["cad_file", "cad_loader", "storage"],
    outputs=["face_count", "edge_count"]
)
def my_encoder(cad_file: str, cad_loader, storage):
    # Call the FlowModel's encoding method
    face_count, edge_count = flow_model.encode_cad_data(cad_file, cad_loader, storage)
    
    # Optional: Add custom label processing
    # ... your label code here ...
    
    # Optional: Convert to graph
    flow_model.convert_encoded_data_to_graph(storage, graph_handler, filename)
    
    return face_count, edge_count
```

### Benefits of This Pattern

1. **Consistency:** Encoding logic defined once in `FlowModel`, reused in Flow
2. **Maintainability:** Changes to encoding strategy only need to update `FlowModel`
3. **Reusability:** Same `FlowModel` used for both training (Flow) and inference
4. **Type Safety:** Flow decorators provide clear input/output contracts

---

## Complete Example: FABWAVE Dataset

This example demonstrates processing the FABWAVE dataset (45 part classes) using `GraphClassification` integrated with HOOPS AI Flows.

### Dataset Structure

```
fabwave/
├── Bearings/
│   ├── bearing_001.step
│   ├── bearing_002.step
│   └── ...
├── Bolts/
│   ├── bolt_001.step
│   └── ...
├── Brackets/
└── ...  (45 categories total)
```

### Complete Implementation

```python
"""
FABWAVE Part Classification using GraphClassification FlowModel
"""

import pathlib
import numpy as np
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
from hoops_ai.storage.datastorage.schema_builder import SchemaBuilder
from hoops_ai.dataset import DatasetExplorer

# FlowModel import
from hoops_ai.ml.EXPERIMENTAL import GraphClassification

# ==================================================================================
# CONFIGURATION
# ==================================================================================

flows_inputdir = pathlib.Path(r"C:\path\to\fabwave")
flows_outputdir = pathlib.Path(r"C:\path\to\output")
datasources_dir = str(flows_inputdir)

# Define label mapping (folder names to class indices)
labels_description = {
    0: {"name": "Bearings", "description": "FABWAVE dataset sample"},
    1: {"name": "Bolts", "description": "FABWAVE dataset sample"},
    2: {"name": "Brackets", "description": "FABWAVE dataset sample"},
    # ... 45 classes total
    44: {"name": "Wide Grip External Retaining Ring", "description": "FABWAVE dataset sample"},
}

# Invert for lookup: folder_name -> class_id
description_to_code = {v["name"]: k for k, v in labels_description.items()}

# Define schema for part classification
builder = SchemaBuilder(
    domain="Part_classification", 
    version="1.0", 
    description="Schema for part classification"
)
file_group = builder.create_group("file", "file", "Information related to the cad file")
file_group.create_array("file_label", ["file"], "int32", "FABWAVE part label as integer (0-44)")
builder.define_categorical_metadata('file_label_description', 'str', 'Part classification')
builder.set_metadata_routing_rules(
    categorical_patterns=['file_label_description', 'category', 'type']
)
cad_schema = builder.build()

# ==================================================================================
# FLOWMODEL INSTANTIATION
# ==================================================================================

flowname = "FABWAVE_v2_45classes"
flow_model = GraphClassification(
    num_classes=45, 
    result_dir=str(pathlib.Path(flows_outputdir).joinpath("flows").joinpath(flowname))
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
    Gather all CAD files from the FABWAVE dataset directory.
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
    Encode CAD data using GraphClassification FlowModel.
    
    This task wraps the FlowModel's encode_cad_data() method and adds:
    1. Schema configuration
    2. Label extraction from folder name
    3. Graph conversion for ML training
    """
    # Set schema for storage
    storage.set_schema(cad_schema)
    
    # ===== CALL FLOWMODEL METHOD =====
    face_count, edge_count = flow_model.encode_cad_data(cad_file, cad_loader, storage)
    # =================================
    
    # Extract label from folder structure (FABWAVE-specific logic)
    folder_with_name = str(pathlib.Path(cad_file).parent.parent.stem)
    label_code = description_to_code.get(folder_with_name, -1)
    
    # Save label to storage
    storage.save_data("file_label", np.array([label_code]).astype(np.int64))
    storage.save_metadata(f"file_label_description", [
        {str(label_code): labels_description[label_code]["name"]}
    ])
    
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
    Execute the FABWAVE preprocessing flow using GraphClassification FlowModel.
    """
    # Create Flow with tasks
    cad_flow = hoops_ai.create_flow(
        name=flowname,
        tasks=[
            my_demo_gatherer,      # Gather CAD files
            my_demo_encoder        # Encode using FlowModel
        ],
        max_workers=40,  # 40 parallel workers
        flows_outputdir=str(flows_outputdir),
        ml_task="Part Classification with GraphClassification",
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
    facecount_is_medium = lambda ds: ds['num_nodes'] > 40
    filelist = explorer.get_file_list(group="graph", where=facecount_is_medium)
    print(f"Files with num_nodes > 40: {len(filelist)}")


if __name__ == "__main__":
    main()
```

### Key Integration Points

#### 1. FlowModel Instantiation

```python
# Instantiate ONCE at module level
flow_model = GraphClassification(num_classes=45, result_dir="./results")
```

#### 2. Encoding Task Wrapper

```python
@flowtask.transform(...)
def my_demo_encoder(cad_file, cad_loader, storage):
    # Call FlowModel method directly
    face_count, edge_count = flow_model.encode_cad_data(cad_file, cad_loader, storage)
    
    # Add custom logic (labels, schema, etc.)
    # ...
    
    # Call another FlowModel method
    flow_model.convert_encoded_data_to_graph(storage, graph_handler, filename)
    
    return face_count, edge_count
```

#### 3. Schema Configuration

```python
# Define schema before encoding
storage.set_schema(cad_schema)

# FlowModel encoding respects schema
flow_model.encode_cad_data(cad_file, cad_loader, storage)
```

### Output Structure

```
output/
└── flows/
    └── FABWAVE_v2_45classes/
        ├── encoded/           # Individual .data files (Zarr format)
        │   ├── bearing_001.data
        │   ├── bolt_001.data
        │   └── ...
        ├── dgl/              # DGL graph files for ML training
        │   ├── bearing_001.ml
        │   ├── bolt_001.ml
        │   └── ...
        ├── info/             # Metadata (.infoset/.attribset)
        └── flow_output.json  # Flow execution summary
```

---

## Training Workflow

### Step 1: Preprocess Dataset

Use the Flow example above to create ML-ready graph files.

### Step 2: Load Dataset

```python
from hoops_ai.dataset import DatasetLoader

# Load preprocessed graphs and labels
dataset_loader = DatasetLoader(
    graph_files=["./output/flows/FABWAVE_v2_45classes/dgl/*.ml"],
    label_files=["./output/flows/FABWAVE_v2_45classes/info/*.attribset"]
)

# Split into train/val/test
dataset_loader.split_data(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
```

### Step 3: Create Trainer

```python
from hoops_ai.ml import FlowTrainer

trainer = FlowTrainer(
    flowmodel=flow_model,  # Same instance used in Flow
    datasetLoader=dataset_loader,
    batch_size=32,
    num_workers=4,
    experiment_name="fabwave_classification",
    accelerator='gpu',
    devices=1,
    max_epochs=100,
    result_dir="./experiments"
)
```

### Step 4: Train Model

```python
# Train and get best checkpoint
best_checkpoint = trainer.train()
print(f"Training complete! Best model: {best_checkpoint}")

# Evaluate on test set
trainer.test(trained_model_path=best_checkpoint)

# Access metrics
metrics = trainer.metrics_storage()
```

### Step 5: Monitor Training

```bash
# Launch TensorBoard
tensorboard --logdir=./experiments/ml_output/fabwave_classification/
```

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

# Interpret results
predicted_class = predictions['predictions'][0]
confidence = predictions['probabilities'][0][predicted_class]
class_name = labels_description[predicted_class]["name"]

print(f"Predicted: {class_name} (confidence: {confidence:.2%})")
```

### Batch Inference

```python
import os

# Get all STEP files from directory
test_dir = "./test_parts"
cad_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.step')]

# Predict for each file
results = []
for cad_file in cad_files:
    batch = inference.preprocess(cad_file)
    pred = inference.predict_and_postprocess(batch)
    results.append({
        'file': cad_file,
        'prediction': pred['predictions'][0],
        'confidence': pred['probabilities'][0].max()
    })

# Print summary
for result in results:
    print(f"{result['file']}: {result['prediction']} ({result['confidence']:.2%})")
```

---

## Hyperparameter Tuning

### Default Configuration

The `GraphClassification` model uses default hyperparameters which are embedded in the underlying architecture.

### Tuning Strategy

Since the model uses a pre-defined architecture, hyperparameter tuning focuses on:

1. **Training Hyperparameters** (via `FlowTrainer`)
2. **Face Discretization Resolution** (in encoding)
3. **Data Augmentation** (custom preprocessing)

For architecture-level modifications, you would need to extend the `GraphClassification` class.

### Training Hyperparameters

```python
trainer = FlowTrainer(
    flowmodel=flow_model,
    datasetLoader=dataset_loader,
    
    # Batch size (larger = faster, more memory)
    batch_size=64,  # Try: 16, 32, 64, 128
    
    # Learning rate
    learning_rate=0.001,  # Try: 0.0001, 0.001, 0.01
    
    # Epochs
    max_epochs=200,  # Try: 50, 100, 200
    
    # Gradient clipping
    gradient_clip_val=1.0,  # Try: 0.5, 1.0, 2.0
    
    # Device
    accelerator='gpu',
    devices=1,
)
```

### Face Discretization Resolution

Higher number of sample points captures more geometric detail but increases memory:

```python
# In your encoding task:
def my_encoder(cad_file, cad_loader, storage):
    face_count, edge_count = flow_model.encode_cad_data(cad_file, cad_loader, storage)
    
    # Override default encoding with custom resolution
    brep_encoder = BrepEncoder(model.get_brep(), storage)
    brep_encoder.push_face_discretization(pointsamples=50)  # Higher resolution (default: 25)
    brep_encoder.push_curvegrid(20)     # Higher resolution (default: 10)
```

**Trade-offs:**
- **Low resolution (10 points):** Fast, less memory, lower accuracy
- **Medium resolution (25 points):** Balanced (default)
- **High resolution (50+ points):** Slower, more memory, higher accuracy

### Model Architecture Modifications

To modify the underlying architecture, you would need to:

1. Extend the `GraphClassification` class
2. Override the `retrieve_model()` method
3. Customize the model initialization in `_thirdparty/` directory

**Example:**
```python
from hoops_ai.ml.EXPERIMENTAL import GraphClassification

class CustomGraphClassification(GraphClassification):
    def __init__(self, num_classes, **kwargs):
        super().__init__(num_classes, **kwargs)
        
        # Override with custom model parameters if needed
        # See the underlying architecture documentation for available options
```

---

## Performance Optimization

### GPU Acceleration

```python
# Use GPU for training
trainer = FlowTrainer(
    accelerator='gpu',
    devices=1,  # Or devices=[0, 1] for multi-GPU
    precision=16,  # Mixed precision for faster training
)
```

### Parallel Preprocessing

```python
# Increase workers for Flow preprocessing
cad_flow = hoops_ai.create_flow(
    name="my_flow",
    tasks=[my_gatherer, my_encoder],
    max_workers=50,  # More workers = faster preprocessing
    flows_outputdir="./output"
)
```

### Memory Management

```python
# Reduce batch size if OOM
trainer = FlowTrainer(batch_size=16)

# Enable gradient accumulation for large effective batch size
trainer = FlowTrainer(
    batch_size=16,
    accumulate_grad_batches=4  # Effective batch size: 16 * 4 = 64
)
```

---

## Troubleshooting

### Issue: Shape Mismatch During Training

**Symptom:** `RuntimeError: expected shape [B, 25, 7], got [B, 40, 7]`

**Cause:** Inconsistent face discretization resolution between files

**Solution:**
```python
# Ensure all files use same number of sample points
brep_encoder.push_face_discretization(pointsamples=25)  # Always use 25 points
```

---

### Issue: Label Not Found

**Symptom:** `KeyError: 'Unknown_Folder'`

**Cause:** Folder name not in `description_to_code` mapping

**Solution:**
```python
# Add default label for unknown classes
label_code = description_to_code.get(folder_name, -1)

# Filter out unknown classes during dataset loading
dataset_loader = DatasetLoader(...)
dataset_loader.filter(lambda x: x['file_label'] != -1)
```

---

### Issue: Low Accuracy

**Possible Causes:**
1. **Insufficient training data:** Collect more samples per class
2. **Class imbalance:** Use weighted loss or data augmentation
3. **Poor UV parameterization:** Some CAD files may have degenerate UV coordinates
4. **Hyperparameters:** Try different learning rates, batch sizes

**Solutions:**
```python
# Check class distribution
explorer = DatasetExplorer(flow_output_file="...")
labels = explorer.get_column_data("file", "file_label")
print(np.bincount(labels))

# Use class weights during training (requires model modification)
```

---

## Related Documentation

- [Flow Model Architecture](./FlowModel_Architecture.md)
- [GraphNodeClassification (Graph Node Classifier)](./GraphNodeClassification.md)
- [Acknowledgements](./Acknowledgements.md) - Attribution and citations
- [Module Access & Encoder Documentation](./Module_Access_and_Encoder.md)
- [Flow Documentation](./Flow_Documentation.md)

---

## Conclusion

`GraphClassification` provides a production-ready implementation of a graph-level classifier for CAD part classification. By following the `FlowModel` interface, it seamlessly integrates with HOOPS AI's Flow framework for batch preprocessing and supports both training and inference workflows with guaranteed encoding consistency.

**Key Takeaways:**
1. Instantiate `GraphClassification` once at module level
2. Wrap its methods in `@flowtask` decorated tasks
3. Use the same instance for training (`FlowTrainer`) and inference (`FlowInference`)
4. Customize encoding by modifying the Flow task, not the FlowModel

**Attribution:** This implementation is based on a third-party architecture. When publishing research using this model, please refer to [Acknowledgements.md](./Acknowledgements.md) for proper citation.

---

**Document Version:** 1.0  
**Last Updated:** October 30, 2025  
**Status:** Experimental  
**Maintainer:** Tech Soft 3D ML Team
