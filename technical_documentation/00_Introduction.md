# HOOPS AI Technical Documentation

## Welcome

Welcome to the comprehensive technical documentation for **HOOPS AI** – a Python framework designed to transform CAD (Computer-Aided Design) data into machine learning-ready datasets. This documentation provides in-depth guidance for data scientists, ML engineers, and CAD software developers who want to leverage state-of-the-art machine learning techniques for CAD analysis.

---

## What is HOOPS AI?

HOOPS AI is a **flow-based data processing framework** that bridges the gap between CAD files and machine learning pipelines. The framework is organized into five core modules:

### **Access & Encode Module**
Your 3D CAD data accessible from Python with operators to transform 3D data into numerical representations that can be ingested by ML algorithms. Load CAD files and extract geometric/topological features using HOOPS Exchange.

### **Flow Module**
Orchestrator to accelerate working with large datasets. Provides parallel encoding of thousands of CAD files with process-level isolation, and generates assets for data and visualization.

### **Dataset Module**
Serve your encoded data for analysis and subset splits. Offers schema-driven storage, automatic merging, metadata management, and dataset preparation for training.

### **ML Module**
Prebuilt architectures for graph classification and segmentation. Features state-of-the-art models (UV-Net, BrepMFR) wrapped for immediate use with unified interfaces for both dataset-scale training and single-file deployment.

### **Insight Module**
Your eyes during the journey. Tools for data exploration, visualization, and analysis throughout the ML pipeline.

---

## Architecture Overview

HOOPS AI follows a **modular pipeline architecture** where each module handles a specific stage of the CAD-to-ML workflow:

```
┌──────────────────────────────────────────────────────────────────────┐
│                    HOOPS AI Module Architecture                      │
└──────────────────────────────────────────────────────────────────────┘

MODULE 1: ACCESS & ENCODE
┌─────────────────────────────────────────────────────────────────────┐
│  CAD Files → HOOPSLoader → BrepEncoder → Storage (per-file)         │
│  • Load CAD models with HOOPS Exchange                              │
│  • Extract B-Rep geometry and topology                              │
│  • Compute features (areas, normals, adjacency, histograms)         │
│  • Store in compressed Zarr format (.data files)                    │
└─────────────────────────────────────────────────────────────────────┘
                               ↓
MODULE 2: FLOW
┌─────────────────────────────────────────────────────────────────────┐
│  Flow Manager → ParallelTask → SchemaBuilder → DataStorage          │
│  • Define tasks with @flowtask decorators                           │
│  • Automatic parallel execution (ProcessPoolExecutor)               │
│  • Schema-driven data validation and organization                   │
│  • Progress tracking, error handling, logging                       │
│  • Generate assets for data and visualization                       │
└─────────────────────────────────────────────────────────────────────┘
                               ↓
MODULE 3: DATASET
┌─────────────────────────────────────────────────────────────────────┐
│  DatasetMerger → DatasetExplorer → DatasetLoader                    │
│  • Merge thousands of .data files into unified dataset              │
│  • Query arrays by group and metadata                               │
│  • Statistical analysis and distribution creation                   │
│  • Stratified train/val/test splitting                              │
│  • Serve encoded data for analysis and subset splits                │
└─────────────────────────────────────────────────────────────────────┘
                               ↓
MODULE 4: ML
┌─────────────────────────────────────────────────────────────────────┐
│  FlowModel → FlowTrainer / FlowInference → Trained Models           │
│  • Prebuilt architectures for graph classification                  │
│  • Prebuilt architectures for graph segmentation                    │
│  • PyTorch Lightning integration                                    │
│  • Single-file inference with encoding consistency                  │
└─────────────────────────────────────────────────────────────────────┘
                               ↓
MODULE 5: INSIGHT
┌─────────────────────────────────────────────────────────────────────┐
│  Visualization & Analysis Tools                                     │
│  • Dataset statistics and distributions                             │
│  • CAD feature visualization                                        │
│  • Training metrics and loss curves                                 │
│  • Model prediction visualization                                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Key Design Principles

### 1. **Declarative Over Imperative**
Use `@flowtask` decorators to define **what** to process, not **how** to parallelize it. The framework handles threading, process pools, and error management automatically.

### 2. **Schema-Driven Data Organization**
Define your data structure once using `SchemaBuilder`, and it propagates through storage, validation, merging, and querying. No manual bookkeeping of array dimensions or metadata routing.

### 3. **Flow-Based Processing**
All operations are organized into **Flows** – pipelines of tasks that transform data step-by-step. Flows handle dependency resolution, logging, and output management.

### 5. **Modular Separation of Concerns**
- **Access & Encode Module**: Load CAD files and extract features (geometric, topological, shape descriptors)
- **Flow Module**: Orchestrate pipelines with parallel execution, error handling, and asset generation
- **Dataset Module**: Merge, analyze, and prepare data with schema-driven storage and provenance tracking
- **ML Module**: Train and deploy models with prebuilt architectures (PyTorch Lightning wrappers)
- **Insight Module**: Visualize and analyze data throughout the ML pipeline

### 5. **Encoding Consistency**
Use the same `FlowModel` EXPERIMENTAL interface for both training (batch processing) and inference (single-file). Encoding logic is defined once and reused across the ML lifecycle.

---

## Who Should Read This Documentation?

### **Data Scientists & ML Engineers**
- Want to train models on CAD datasets
- Need reproducible data pipelines
- Require efficient large-scale data processing
- **Start with:** [Flow Module](#flow-module) and [Dataset Module](#dataset-module)

### **CAD Software Developers**
- Building CAD analysis tools
- Integrating ML into CAD workflows
- Extracting geometric features programmatically
- **Start with:** [Access & Encode Module](#access--encode-module)

### **Research Scientists**
- Publishing papers on CAD ML
- Implementing custom encoders or architectures
- Understanding state-of-the-art methods
- **Start with:** [ML Module](#ml-module) and [FlowModel Architecture](#flowmodel-architecture-experimental)

---

## Documentation Structure

This technical documentation is organized into **11 focused documents** grouped by the five core modules. Below is the complete table of contents with descriptions.

---

## Table of Contents

### Access & Encode Module

#### **[Module Access & Encoder Documentation](./Acces%20and%20Encode%20Modules/Module_Access_and_Encoder.md)**
**Load CAD files and extract geometric/topological features**

**What You'll Learn:**
- HOOPSLoader singleton for CAD file loading
- HOOPSModel and HOOPSBrep interfaces
- BrepEncoder push-based architecture
- Geometric features (face areas, edge lengths, UV grids)
- Topological features (adjacency graphs, extended adjacency)
- Shape descriptors (D2 distance histograms, A3 angle histograms)

**Key Topics:**
- `push_face_indices`, `push_face_attributes`, `push_facegrid`
- `push_edge_attributes`, `push_curvegrid`
- `push_face_adjacency_graph`, `push_extended_adjacency`
- Histogram methods for pairwise face comparisons
- Mathematical formulations for all encoding methods

**Best For:** Understanding how CAD files are encoded into feature arrays

---

### Flow Module

#### **[Flow Documentation](./Flow%20Module/Flow_Documentation.md)** ⭐ Start Here
**Core orchestration system for building CAD processing pipelines**

**What You'll Learn:**
- How to use `@flowtask` decorators to define processing steps
- Automatic parallel execution with ProcessPoolExecutor
- Flow creation with `hoops_ai.create_flow()`
- Error handling, logging, and progress tracking
- Windows-specific requirements for multiprocessing
- Complete workflow examples from CAD files to datasets

**Key Topics:**
- Task decorators (`@flowtask.extract`, `@flowtask.transform`, `@flowtask.custom`)
- ParallelTask and SequentialTask base classes
- Automatic dataset export (`auto_dataset_export=True`)
- HOOPSLoader lifecycle management per worker
- Flow output structure (`.flow`, `.dataset`, `.infoset`, `.attribset`)

**Best For:** Understanding how to build end-to-end CAD data pipelines

#### **[Flow Quick Reference](./Flow%20Module/Flow_Quick_Reference.md)**
**Condensed cheat sheet for Flow usage**

**What You'll Learn:**
- Quick syntax examples for common patterns
- Task decorator templates
- Flow creation shortcuts
- Common troubleshooting tips

**Best For:** Quick lookups during development

---

### Dataset Module

#### **[SchemaBuilder Documentation](./Dataset%20Module/SchemaBuilder_Documentation.md)**
**Declarative API for defining data organization schemas**

**What You'll Learn:**
- Creating logical groups and arrays with explicit dimensions
- Defining file-level vs categorical metadata
- Metadata routing with pattern matching
- Schema templates for common use cases
- Integration with DataStorage for validation

**Key Topics:**
- `SchemaBuilder` class and fluent API
- Group and array definitions
- Metadata routing rules (wildcards, type-based defaults)
- Schema dictionary structure
- Template system (`cad_basic`, `cad_advanced`, etc.)

**Best For:** Defining structured, validated data schemas before encoding

#### **[DataStorage Documentation](./Dataset%20Module/DataStorage_Documentation.md)**
**Unified interface for persisting and retrieving data**

**What You'll Learn:**
- Abstract DataStorage interface and concrete implementations
- OptStorage (Zarr-based) for production use
- MemoryStorage for testing and prototyping
- JsonStorageHandler for human-readable exports
- Schema integration for validation and routing
- Automatic size tracking and compression

**Key Topics:**
- Core abstract methods (`save_data`, `load_data`, `save_metadata`)
- Zarr format with compression and chunking
- Dimension naming for xarray compatibility
- NaN detection and validation
- Metadata routing (`.infoset` vs `.attribset`)

**Best For:** Understanding how data is persisted during encoding

#### **[DatasetMerger Documentation](./Dataset%20Module/DatasetMerger_Documentation.md)**
**Consolidates individual encoded files into unified datasets**

**What You'll Learn:**
- How Flow automatically invokes merging (`AutoDatasetExportTask`)
- Group-based array concatenation with provenance tracking
- Schema-driven vs heuristic group discovery
- Batch merging for memory-constrained environments
- Output file structure (`.dataset`, `.infoset`, `.attribset`)

**Key Topics:**
- DatasetMerger class and initialization
- DatasetInfo for metadata aggregation
- Matrix flattening for face-face relationships
- Dask-based parallel processing
- File ID code mapping for provenance

**Best For:** Understanding how individual `.data` files become ML-ready datasets

#### **[DatasetExplorer & DatasetLoader Documentation](./Dataset%20Module/DatasetExplorer_DatasetLoader_Documentation.md)**
**Query, analyze, and prepare merged datasets for ML training**

**What You'll Learn:**
- Querying arrays by group and metadata filters
- Statistical analysis and distribution creation
- Cross-group queries and joins
- Stratified train/val/test splitting (single and multi-label)
- PyTorch DataLoader integration
- Custom item loaders for preprocessing

**Key Topics:**
- DatasetExplorer: `get_array_data`, `create_distribution`, `filter_files_by_metadata`
- DatasetLoader: `split`, `get_dataset`, `to_torch`
- Multi-label stratification with membership matrices
- CADDataset class and framework adapters
- Performance optimization (Dask configuration, chunking)

**Best For:** Analyzing datasets and preparing data for ML training

---

### ML Module

#### **[FlowModel Architecture Documentation](./ML%20Module/FlowModel_Architecture.md)** ⚠️ EXPERIMENTAL
**Unified interface for ML model integration (training and inference)**

**What You'll Learn:**
- FlowModel abstract interface design
- Why encoding consistency matters between training and inference
- Available implementations (GraphClassification, GraphNodeClassification)
- FlowTrainer for batch dataset training
- FlowInference for single-file deployment
- Third-party model integration (UV-Net, BrepMFR)

**Key Topics:**
- Abstract methods: `encode_cad_data`, `convert_encoded_data_to_graph`, `load_model_input_from_files`
- FlowTrainer initialization and training loop
- FlowInference preprocessing and prediction
- Checkpoint management and metrics storage
- Attribution and MIT license compliance

**Best For:** Understanding the ML model integration layer

#### **[GraphClassification (UV-Net) Documentation](./ML%20Module/GraphClassification_UVNet.md)** ⚠️ EXPERIMENTAL
**Graph-level classification using UV-Net architecture**

**What You'll Learn:**
- UV-Net architecture (2D CNNs on face UV-grids, 1D CNNs on edge grids)
- Initialization and hyperparameters
- CAD encoding process (UV grid sampling)
- Flow integration with `@flowtask` decorators
- Complete FABWAVE example (45 part classes)

**Key Topics:**
- GraphClassification class initialization
- `encode_cad_data` method details
- Graph conversion and feature attachment
- Training with FlowTrainer
- Inference with FlowInference
- Citation information for UV-Net paper

**Best For:** Part classification and shape categorization tasks

#### **[GraphNodeClassification (BrepMFR) Documentation](./ML%20Module/GraphNodeClassification_BrepMFR.md)** ⚠️ EXPERIMENTAL
**Node-level (face) classification using BrepMFR architecture**

**What You'll Learn:**
- BrepMFR Transformer-based GNN architecture
- Rich topological encoding (extended adjacency, edge path matrices)
- Histogram-based shape descriptors
- Flow integration for per-face labeling
- Complete CADSynth-AAG example (162k models, 25 feature classes)

**Key Topics:**
- GraphNodeClassification class initialization
- Advanced encoding: histograms, extended adjacency, face-pair paths
- Transformer hyperparameters (`n_layers_encode`, `n_layers_decode`)
- Training strategies and hyperparameter tuning
- Inference on new CAD files
- Citation information for BrepMFR paper

**Best For:** Machining feature recognition and semantic segmentation tasks

---

### Insight Module

#### **[Insights & Visualization Documentation](./Insight%20Module/Insights_Visualization_Documentation.md)**
**Visualization and analysis tools for CAD ML pipelines**

**What You'll Learn:**
- Dataset statistics and distributions
- CAD feature visualization
- Training metrics and loss curves
- Model prediction visualization

**Best For:** Monitoring and understanding your CAD ML pipeline

---

## Getting Started

### Recommended Learning Path

#### **For First-Time Users:**
1. **Flow Module:** Read [Flow Documentation](./Flow%20Module/Flow_Documentation.md) to understand the orchestration layer
2. **Access & Encode Module:** Review [Module Access & Encoder](./Acces%20and%20Encode%20Modules/Module_Access_and_Encoder.md) to see how features are extracted
3. Try a complete example from [Flow Documentation > Complete Workflow Example](./Flow%20Module/Flow_Documentation.md#complete-workflow-example)

#### **For ML Engineers:**
1. **Dataset Module:** Skim [DatasetMerger Documentation](./Dataset%20Module/DatasetMerger_Documentation.md) to understand dataset structure
2. **Dataset Module:** Deep-dive [DatasetExplorer & DatasetLoader](./Dataset%20Module/DatasetExplorer_DatasetLoader_Documentation.md) for training prep
3. **ML Module:** Choose [GraphClassification](./ML%20Module/GraphClassification_UVNet.md) or [GraphNodeClassification](./ML%20Module/GraphNodeClassification_BrepMFR.md) based on your task

#### **For Advanced Users:**
1. **Dataset Module:** Study [SchemaBuilder](./Dataset%20Module/SchemaBuilder_Documentation.md) for custom data organization
2. **ML Module:** Read [FlowModel Architecture](./ML%20Module/FlowModel_Architecture.md) to implement custom models
3. **Insight Module:** Explore [Insights & Visualization](./Insight%20Module/Insights_Visualization_Documentation.md) for analysis tools
4. Explore source code in `src/hoops_ai/` for extending functionality

---

## Common Workflows

### Workflow 1: CAD File → Training Dataset

```python
# Step 1: Define schema (optional but recommended)
from hoops_ai.storage.datasetstorage import SchemaBuilder

builder = SchemaBuilder(domain="CAD_analysis", version="1.0")
faces_group = builder.create_group("faces", "face", "Face data")
faces_group.create_array("face_areas", ["face"], "float32")
schema = builder.build()

# Step 2: Define tasks with decorators
from hoops_ai.flowmanager import flowtask
from hoops_ai.cadencoder import BrepEncoder

@flowtask.extract(
    name="gather_files",
    inputs=["cad_datasources"],
    outputs=["cad_dataset"]
)
def gather_files(source: str):
    return glob.glob(f"{source}/**/*.step", recursive=True)

@flowtask.transform(
    name="encode_cad",
    inputs=["cad_file", "cad_loader", "storage"],
    outputs=["encoded_path"]
)
def encode_cad(cad_file, cad_loader, storage):
    storage.set_schema(schema)
    model = cad_loader.create_from_file(cad_file)
    encoder = BrepEncoder(model.get_brep(), storage)
    encoder.push_face_indices()
    encoder.push_face_attributes()
    storage.compress_store()
    return storage.get_file_path("")

# Step 3: Create and run flow
import hoops_ai

flow = hoops_ai.create_flow(
    name="my_pipeline",
    tasks=[gather_files, encode_cad],
    flows_outputdir="./output",
    max_workers=8,
    auto_dataset_export=True
)

flow_output, summary, flow_file = flow.process(
    inputs={'cad_datasources': ["/path/to/cad/files"]}
)
```

### Workflow 2: Dataset → Trained Model

```python
# Step 1: Explore dataset
from hoops_ai.dataset import DatasetExplorer

explorer = DatasetExplorer(flow_output_file=flow_file)
explorer.print_table_of_contents()
explorer.close()

# Step 2: Prepare for training
from hoops_ai.dataset import DatasetLoader

loader = DatasetLoader(
    merged_store_path=summary['flow_data'],
    parquet_file_path=summary['flow_info']
)

loader.split(key="part_category", train=0.7, validation=0.15, test=0.15)

# Step 3: Train model
from hoops_ai.ml.EXPERIMENTAL import GraphClassification, FlowTrainer

flow_model = GraphClassification(num_classes=10)
trainer = FlowTrainer(
    flowmodel=flow_model,
    datasetLoader=loader,
    batch_size=32,
    max_epochs=100
)

best_checkpoint = trainer.train()
```

### Workflow 3: Trained Model → Single-File Inference

```python
# Step 1: Initialize inference pipeline
from hoops_ai.ml.EXPERIMENTAL import FlowInference
from hoops_ai.cadaccess import HOOPSLoader

cad_loader = HOOPSLoader()
flow_model = GraphClassification(num_classes=10)
inference = FlowInference(cad_loader, flow_model)
inference.load_from_checkpoint(best_checkpoint)

# Step 2: Run inference on new file
batch = inference.preprocess("new_part.step")
predictions = inference.predict_and_postprocess(batch)
print(f"Predicted class: {predictions['predictions'][0]}")
```

---

## Key Concepts

### Flow-Based Architecture
All processing in HOOPS AI is organized into **Flows** – sequences of tasks that transform data. Tasks are defined using decorators and automatically parallelized.

### Schema-Driven Data
Define your data structure once with `SchemaBuilder`, and it governs:
- Storage validation
- Metadata routing
- Dataset merging
- Query operations

### Provenance Tracking
Every merged dataset includes `file_id` arrays that link each data point back to its source CAD file, enabling:
- Debugging encoding issues
- Filtering by file properties
- Stratified splitting by file-level metadata

### Encoding Consistency
The `FlowModel` interface ensures that CAD encoding logic is identical between:
- Training (batch processing thousands of files)
- Inference (real-time single-file processing)

---

## Experimental Features

The following components are marked as **⚠️ EXPERIMENTAL** and may change in future releases:

- **FlowModel Architecture** (abstract interface for ML models)
- **GraphClassification** (UV-Net wrapper)
- **GraphNodeClassification** (BrepMFR wrapper)
- **FlowTrainer** (training pipeline)
- **FlowInference** (deployment pipeline)

These features are production-ready but the APIs may evolve based on user feedback and new ML framework requirements.

---

### Example Notebooks
Located in `notebooks/`:
- `1_access_a_cad_file.ipynb`: Load and inspect CAD files
- `2_encode_cad_file.ipynb`: Extract features from a single file
- `3a_ETL_pipeline_using_flow.ipynb`: Complete Flow example
- `4_train_a_ml_model_to_classify_parts.ipynb`: End-to-end ML training
- `5_infer_features_using_cad_as_input.ipynb`: Deployment example

### Community and Contact
- **GitHub:** https://github.com/techsoft3d/hoops-ai
- **Documentation:** https://docs.techsoft3d.com/hoops-ai
- **Support:** support@techsoft3d.com
- **Forums:** https://forum.techsoft3d.com

---

## Attribution

### Third-Party Models

HOOPS AI integrates the following open-source ML architectures under MIT License:

**UV-Net** (Autodesk AI Lab, CVPR 2021)
- Paper: *UV-Net: Learning from Boundary Representations*
- Authors: Jayaraman et al.
- Source: https://github.com/AutodeskAILab/UV-Net

**BrepMFR** (Zhang et al., CAGD 2024)
- Paper: *BrepMFR: Enhancing machining feature recognition in B-rep models through deep learning and domain adaptation*
- Authors: Zhang, Guan, Jiang, Wang, Tan
- Source: https://github.com/zhangshuming0668/BrepMFR

Tech Soft 3D provides wrappers and integration but **does not claim authorship** of these architectures. Users should cite the original papers when publishing results.

---

## License

**HOOPS AI Framework:** Copyright © 2025 Tech Soft 3D, Inc. All rights reserved.

**Third-Party Components:** See individual LICENSE files in `src/hoops_ai/ml/_thirdparty/` for MIT License details.

---

## Version Information

**Current Version:** 1.0  
**Last Updated:** January 2025  
**Documentation Status:** Complete (11 technical documents)  
**API Stability:** Stable (Core modules), Experimental (ML integration)

---

