# DatasetExplorer and DatasetLoader Documentation

## Overview

The **DatasetExplorer** and **DatasetLoader** are complementary tools that work with merged datasets (`.dataset`, `.infoset`, `.attribset` files) produced by the DatasetMerger during Flow execution. Together, they provide comprehensive capabilities for dataset analysis, querying, and ML training preparation.

**Key Purposes:**
- **DatasetExplorer**: Query, analyze, and visualize merged CAD datasets
- **DatasetLoader**: Prepare datasets for ML training with stratified train/val/test splitting

These classes form the **analysis and ML preparation layer** of the HOOPS AI pipeline, consuming the unified datasets produced by the automatic merging process.

```
DatasetMerger Output → DatasetExplorer (Analysis) → DatasetLoader (ML Prep) → Training
(.dataset/.infoset/.attribset)
```

---

## Table of Contents

1. [Architecture & Integration](#architecture--integration)
   - [Position in Pipeline](#position-in-pipeline)
   - [Input Files](#input-files)
   - [Relationship to DatasetMerger](#relationship-to-datasetmerger)
2. [DatasetExplorer](#datasetexplorer)
   - [Initialization](#datasetexplorer-initialization)
   - [Query Operations](#query-operations)
   - [Distribution Analysis](#distribution-analysis)
   - [Metadata Queries](#metadata-queries)
   - [Cross-Group Queries](#cross-group-queries)
   - [Advanced Features](#advanced-features)
3. [DatasetLoader](#datasetloader)
   - [Initialization](#datasetloader-initialization)
   - [Stratified Splitting](#stratified-splitting)
   - [Dataset Access](#dataset-access)
   - [ML Framework Integration](#ml-framework-integration)
   - [Custom Item Loaders](#custom-item-loaders)
4. [Complete Workflow Examples](#complete-workflow-examples)
5. [Best Practices](#best-practices)
6. [Performance Considerations](#performance-considerations)

---

## Architecture & Integration

### Position in Pipeline

DatasetExplorer and DatasetLoader operate in the **Analysis & ML Preparation Phase**:

```
┌──────────────────────────────────────────────────────────────────────┐
│                    HOOPS AI Complete Pipeline                        │
└──────────────────────────────────────────────────────────────────────┘

1. ENCODING PHASE (Per-File)
   ┌─────────────────────────────────────────────────────┐
   │  @flowtask.transform                                │
   │  CAD File → Encoder → Storage → .data file          │
   └─────────────────────────────────────────────────────┘
                            ↓
2. MERGING PHASE (Automatic)
   ┌─────────────────────────────────────────────────────┐
   │  AutoDatasetExportTask (auto_dataset_export=True)   │
   │  Multiple .data → DatasetMerger → .dataset          │
   │  Multiple .json → DatasetInfo → .infoset/.attribset │
   └─────────────────────────────────────────────────────┘
                            ↓
3. ANALYSIS PHASE (DatasetExplorer) ← YOU ARE HERE
   ┌─────────────────────────────────────────────────────┐
   │  .dataset + .infoset + .attribset                   │
   │      ↓                                              │
   │  DatasetExplorer                                    │
   │   - Query arrays by group/key                       │
   │   - Analyze distributions                           │
   │   - Filter by metadata                              │
   │   - Statistical summaries                           │
   │   - Cross-group queries                             │
   └─────────────────────────────────────────────────────┘
                            ↓
4. ML PREPARATION PHASE (DatasetLoader)
   ┌─────────────────────────────────────────────────────┐
   │  DatasetLoader                                      │
   │   - Stratified train/val/test split                 │
   │   - Multi-label support                             │
   │   - Framework-agnostic CADDataset                   │
   │   - PyTorch adapter (.to_torch())                   │
   │   - Custom item loaders                             │
   └─────────────────────────────────────────────────────┘
                            ↓
5. ML TRAINING PHASE
   ┌─────────────────────────────────────────────────────┐
   │  PyTorch DataLoader → Training Loop → Model         │
   └─────────────────────────────────────────────────────┘
```

---

### Input Files

Both classes consume the output of the DatasetMerger:

#### Required Files

1. **`.dataset` file** (Compressed Zarr)
   - Contains all merged array data organized by groups
   - Structure: `{flow_name}.dataset`
   - Format: ZipStore with Zstd compression
   - Accessed via: xarray and Dask for parallel operations

2. **`.infoset` file** (Parquet)
   - Contains file-level metadata (one row per file)
   - Structure: Columnar storage with `id`, `name`, `description`, custom fields
   - Accessed via: pandas DataFrame operations

3. **`.attribset` file** (Parquet) - Optional
   - Contains categorical metadata and label descriptions
   - Structure: `table_name`, `id`, `name`, `description`
   - Accessed via: pandas DataFrame operations

#### File Location

Files are generated by Flow execution in:
```
flow_output/flows/{flow_name}/
├── {flow_name}.dataset      ← Merged data arrays
├── {flow_name}.infoset      ← File-level metadata
├── {flow_name}.attribset    ← Categorical metadata
└── {flow_name}.flow         ← Flow specification (JSON)
```

---

### Relationship to DatasetMerger

**DatasetMerger** (automatic during Flow):
- **Input**: Individual `.data` and `.json` files (per CAD file)
- **Process**: Concatenate arrays, route metadata, add provenance tracking
- **Output**: Unified `.dataset`, `.infoset`, `.attribset` files

**DatasetExplorer** (user-driven analysis):
- **Input**: Output files from DatasetMerger
- **Process**: Query, filter, analyze, visualize
- **Output**: Statistics, distributions, filtered file lists

**DatasetLoader** (ML preparation):
- **Input**: Output files from DatasetMerger
- **Process**: Stratified splitting, dataset creation
- **Output**: Train/val/test CADDataset objects

**Key Distinction:**
- DatasetMerger: **Write-heavy** (consolidate many files into one)
- DatasetExplorer: **Read-heavy** (query and analyze unified data)
- DatasetLoader: **Read + Index** (split and serve data for training)

---

## DatasetExplorer

### DatasetExplorer Initialization

```python
from hoops_ai.dataset import DatasetExplorer

# Method 1: Using flow output JSON file (Recommended)
explorer = DatasetExplorer(flow_output_file="path/to/flow_name.flow")

# Method 2: Explicit file paths
explorer = DatasetExplorer(
    merged_store_path="path/to/flow_name.dataset",
    parquet_file_path="path/to/flow_name.infoset",
    parquet_file_attribs="path/to/flow_name.attribset"  # Optional
)

# Method 3: With custom Dask configuration
explorer = DatasetExplorer(
    flow_output_file="path/to/flow_name.flow",
    dask_client_params={
        'n_workers': 8,
        'threads_per_worker': 4,
        'memory_limit': '8GB'
    }
)
```

**Parameters:**
- `flow_output_file` (str, optional): Path to `.flow` JSON (contains all file paths)
- `merged_store_path` (str, optional): Path to `.dataset` file
- `parquet_file_path` (str, optional): Path to `.infoset` file
- `parquet_file_attribs` (str, optional): Path to `.attribset` file
- `dask_client_params` (dict, optional): Dask configuration for parallel operations

**Automatic Discovery:**
```python
# Discover available groups and arrays
available_groups = explorer.available_groups()
print(f"Groups: {available_groups}")
# Output: {'faces', 'edges', 'graph', 'machining'}

available_arrays = explorer.available_arrays("faces")
print(f"Face arrays: {available_arrays}")
# Output: {'face_indices', 'face_areas', 'face_types', 'face_uv_grids', 'file_id_code_faces'}
```

---

### Query Operations

#### Get Array Data

```python
# Get complete array data for a group
face_areas = explorer.get_array_data(group_name="faces", array_name="face_areas")
# Returns: xr.DataArray with shape [N_total_faces]

# Access underlying NumPy array
face_areas_np = face_areas.values
print(f"Total faces: {len(face_areas_np)}")
print(f"Mean area: {face_areas_np.mean():.2f}")
```

#### Get Group Data

```python
# Get entire dataset for a group
faces_ds = explorer.get_group_data("faces")
print(faces_ds)
# Output:
# <xarray.Dataset>
# Dimensions:        (face: 48530)
# Coordinates:
#   * face           (face) int64 0 1 2 3 ... 48527 48528 48529
# Data variables:
#     face_indices   (face) int32 ...
#     face_areas     (face) float32 ...
#     face_types     (face) int32 ...
#     file_id_code_faces (face) int32 ...

# Access multiple arrays
face_areas = faces_ds['face_areas']
face_types = faces_ds['face_types']
```

#### Filter by Condition

```python
# Get files matching a boolean condition
def high_complexity_filter(ds):
    """Filter for files with many faces"""
    # Count faces per file using file_id_code
    return ds['face_areas'] > 100  # Example: faces with area > 100

file_codes = explorer.get_file_list(
    group="faces",
    where=high_complexity_filter
)
print(f"Found {len(file_codes)} files with large faces")

# Convert file codes to file names
file_names = [explorer.decode_file_id_code(code) for code in file_codes]
```

#### Get File-Specific Data

```python
# Get data for a specific file
file_id_code = 5
face_subset = explorer.file_dataset(file_id_code=file_id_code, group="faces")
print(f"File {file_id_code} has {len(face_subset.face)} faces")

# Access arrays for this file only
file_face_areas = face_subset['face_areas'].values
print(f"Face areas for file {file_id_code}: {file_face_areas}")
```

---

### Distribution Analysis

#### Create Histogram Distribution

```python
# Create distribution with automatic binning
distribution = explorer.create_distribution(
    key="face_areas",
    group="faces",
    bins=20
)

# Access distribution components
print(f"Bin edges: {distribution['bin_edges']}")
print(f"Histogram counts: {distribution['hist']}")
print(f"Files per bin: {distribution['file_ids_in_bins']}")

# Example output:
# bin_edges: [0.5, 1.5, 2.5, ..., 20.5]
# hist: [145, 302, 567, ..., 89]
# file_ids_in_bins: [['part_001', 'part_003'], ['part_002', 'part_005'], ...]
```

#### Visualize Distribution

```python
import matplotlib.pyplot as plt
import numpy as np

dist = explorer.create_distribution(key="face_areas", group="faces", bins=30)

# Plot histogram
bin_centers = 0.5 * (dist['bin_edges'][1:] + dist['bin_edges'][:-1])
plt.bar(bin_centers, dist['hist'], width=(dist['bin_edges'][1] - dist['bin_edges'][0]))
plt.xlabel('Face Area')
plt.ylabel('Count')
plt.title('Face Area Distribution')
plt.show()
```

---

### Metadata Queries

#### File-Level Metadata

```python
# Get metadata for all files
all_file_info = explorer.get_file_info_all()
print(all_file_info.head())
# Output:
#    id         name  size_cadfile  processing_time  complexity_level  subset
# 0   0   part_001      1024000             12.5                     3   train
# 1   1   part_002      2048000             18.3                     4   train
# 2   2   part_003       512000              8.1                     2    test

# Get metadata for specific file
file_info = explorer.get_parquet_info_by_code(file_id_code=5)
print(file_info)
```

#### Categorical Metadata (Labels/Descriptions)

```python
# Get label descriptions from .attribset
complexity_labels = explorer.get_descriptions(table_name="complexity_level")
print(complexity_labels)
# Output:
#   id     name           description
# 0  1   Simple      Basic geometry
# 1  2   Medium      Moderate complexity
# 2  3   Complex     High complexity
# 3  4   Very Complex   Advanced features

# Get specific label description
label_3 = explorer.get_descriptions(table_name="complexity_level", key_id=3)
print(label_3['name'].values[0])  # Output: "Complex"
```

#### Stream Cache Paths (Visualizations)

```python
# Get paths to PNG and 3D stream cache files
stream_paths = explorer.get_stream_cache_paths()
print(stream_paths[['id', 'name', 'stream_cache_png', 'stream_cache_3d']])

# Get stream cache for specific file
file_stream = explorer.get_stream_cache_paths(file_id_code=10)
png_path = file_stream['stream_cache_png'].values[0]
scs_path = file_stream['stream_cache_3d'].values[0]
```

---

### Advanced Features

#### Membership Matrix

```python
# Create membership matrix for multi-label analysis
matrix, file_codes, categories = explorer.build_membership_matrix(
    group="faces",
    key="face_types",
    bins_or_categories=None,  # Auto-discover categories
    as_counts=False  # Boolean membership (True) or counts (False)
)

print(f"Matrix shape: {matrix.shape}")  # (N_files, N_categories)
print(f"File codes: {file_codes[:10]}")
print(f"Categories: {categories}")

# Use for stratification analysis
import pandas as pd
df = pd.DataFrame(matrix, columns=categories)
df['file_code'] = file_codes
print(df.head())
```

#### Print Dataset Structure

```python
# Get overview of entire dataset
explorer.print_table_of_contents()
# Output:
# ========================================
# DATASET TABLE OF CONTENTS
# ========================================
# 
# Available Groups:
# --------------------------------------------------
# 
# Group: faces
#   Arrays:
#     - face_indices: (48530,) int32
#     - face_areas: (48530,) float32
#     - face_types: (48530,) int32
#     - face_uv_grids: (48530, 20, 20, 7) float32
#     - file_id_code_faces: (48530,) int32
# 
# Group: edges
#   Arrays:
#     - edge_indices: (72845,) int32
#     - edge_lengths: (72845,) float32
#     - edge_types: (72845,) int32
#     - file_id_code_edges: (72845,) int32
# 
# Group: machining
#   Arrays:
#     - machining_category: (100,) int32
#     - material_type: (100,) int32
#     - file_id_code_machining: (100,) int32
# 
# Metadata Files:
#   - Info: cad_pipeline.infoset (file-level metadata)
#   - Attributes: cad_pipeline.attribset (categorical metadata)
# 
# Total Files: 100
```

---

### Resource Management

```python
# Close resources when done
explorer.close(close_dask=True)
```

---

## DatasetLoader

### DatasetLoader Initialization

```python
from hoops_ai.dataset import DatasetLoader

# Method 1: Basic initialization
loader = DatasetLoader(
    merged_store_path="path/to/flow_name.dataset",
    parquet_file_path="path/to/flow_name.infoset"
)

# Method 2: With custom item loader (For future versions - not yet fully implemented)
def custom_loader(graph_file, label_file, data_id):
    """Custom function to load and process items"""
    import dgl
    import numpy as np
    
    # Load graph
    graph = dgl.load_graphs(graph_file)[0][0]
    
    # Load label
    label = np.load(label_file)
    
    # Return as dictionary
    return {
        'graph': graph,
        'label': label,
        'id': data_id,
        'num_nodes': graph.number_of_nodes(),
        'num_edges': graph.number_of_edges()
    }

loader = DatasetLoader(
    merged_store_path="path/to/flow_name.dataset",
    parquet_file_path="path/to/flow_name.infoset",
    item_loader_func=custom_loader
)
```

**Parameters:**
- `merged_store_path` (str): Path to `.dataset` file
- `parquet_file_path` (str): Path to `.infoset` file
- `item_loader_func` (callable, optional): Custom function to load items
  - Signature: `func(graph_file, label_file, data_id) -> item`
  - If None, returns raw file paths

---

### Stratified Splitting

#### Basic Stratified Split

```python
# Perform stratified split by a categorical key
train_size, val_size, test_size = loader.split(
    key="complexity_level",  # Metadata key to stratify on
    group="faces",           # Group containing the key
    train=0.7,               # 70% training
    validation=0.15,         # 15% validation
    test=0.15,               # 15% testing
    random_state=42          # For reproducibility
)

print(f"Dataset split:")
print(f"  Train: {train_size} files")
print(f"  Validation: {val_size} files")
print(f"  Test: {test_size} files")
```

**Mathematical Formulation:**

For stratified splitting with key $K$ having $C$ categories, the split aims to preserve the distribution:

$$
P(k_i | \text{train}) \approx P(k_i | \text{validation}) \approx P(k_i | \text{test}) \approx P(k_i)
$$

where $k_i \in K$ is a category and $P(k_i)$ is its proportion in the full dataset.

**Multi-Label Stratification:**

For files with multiple labels (e.g., multiple face types per file), the loader uses `MultilabelStratifiedShuffleSplit`:

$$
\mathbf{M} \in \{0, 1\}^{N \times C}
$$

where:
- $N$ = number of files
- $C$ = number of categories
- $M_{ij} = 1$ if file $i$ has category $j$, else 0

The split preserves label co-occurrence patterns.

---


---

---

### Dataset Access

#### Get Subset Datasets

```python
# Get framework-agnostic datasets
train_dataset = loader.get_dataset("train")
val_dataset = loader.get_dataset("validation")
test_dataset = loader.get_dataset("test")

print(f"Train: {len(train_dataset)} samples")
print(f"Val: {len(val_dataset)} samples")
print(f"Test: {len(test_dataset)} samples")

# Access individual items
item = train_dataset.get_item(0)
print(f"Item: {item}")
```

#### CADDataset Class

The `CADDataset` is a framework-agnostic wrapper:

```python
# Properties
train_dataset.indices          # Indices into parent dataset
train_dataset.parent_dataset   # Reference to DatasetLoader

# Methods
item = train_dataset.get_item(i)       # Get item by local index
raw = train_dataset.get_raw_data(i)    # Get file paths without loading

# Framework conversion (EXPERIMENTAL - not fully implemented)
torch_dataset = train_dataset.to_torch()  # Convert to PyTorch torch.utils.data.Dataset object
```

---

### ML Framework Integration

#### PyTorch Integration

```python
from torch.utils.data import DataLoader

# Get training dataset
train_dataset = loader.get_dataset("train")

# Convert to PyTorch
train_torch = train_dataset.to_torch()

# Create DataLoader
train_loader = DataLoader(
    train_torch,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True  # For GPU training
)

# Training loop
for epoch in range(num_epochs):
    for batch in train_loader:
        # Unpack batch
        graphs = batch['graph']
        labels = batch['label']
        file_ids = batch['id']
        
        # Your training code
        outputs = model(graphs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

---

## Complete Workflow Examples

### Example 1: Basic Analysis and ML Preparation

```python
import hoops_ai
from hoops_ai.flowmanager import flowtask
from hoops_ai.dataset import DatasetExplorer, DatasetLoader
import pathlib

# Assume flow already executed and created:
# - cad_pipeline.dataset
# - cad_pipeline.infoset
# - cad_pipeline.attribset
# - cad_pipeline.flow

flow_file = pathlib.Path("flow_output/flows/cad_pipeline/cad_pipeline.flow")

# ===== STEP 1: Explore Dataset =====
print("="*70)
print("STEP 1: DATASET EXPLORATION")
print("="*70)

explorer = DatasetExplorer(flow_output_file=str(flow_file))

# Print overview
explorer.print_table_of_contents()

# Analyze face area distribution
face_dist = explorer.create_distribution(key="face_areas", group="faces", bins=20)
print(f"\nFace area distribution:")
print(f"  Range: [{face_dist['bin_edges'][0]:.2f}, {face_dist['bin_edges'][-1]:.2f}]")
print(f"  Total faces: {face_dist['hist'].sum()}")
print(f"  Mean bin count: {face_dist['hist'].mean():.1f}")

# Filter files by complexity
high_complexity_filter = lambda ds: ds['complexity_level'] >= 4
complex_files = explorer.get_file_list(group="faces", where=high_complexity_filter)
print(f"\nHigh complexity files: {len(complex_files)}")

# Close explorer
explorer.close()

# ===== STEP 2: Prepare ML Dataset =====
print("\n" + "="*70)
print("STEP 2: ML DATASET PREPARATION")
print("="*70)

# Initialize loader
flow_path = pathlib.Path(flow_file)
loader = DatasetLoader(
    merged_store_path=str(flow_path.parent / f"{flow_path.stem}.dataset"),
    parquet_file_path=str(flow_path.parent / f"{flow_path.stem}.infoset")
)

# Stratified split
train_size, val_size, test_size = loader.split(
    key="complexity_level",
    group="faces",
    train=0.7,
    validation=0.15,
    test=0.15,
    random_state=42
)

print(f"\nDataset split:")
print(f"  Train: {train_size} files")
print(f"  Validation: {val_size} files")
print(f"  Test: {test_size} files")

# Get datasets
train_dataset = loader.get_dataset("train")
val_dataset = loader.get_dataset("validation")
test_dataset = loader.get_dataset("test")

# ===== STEP 3: Prepare for Training =====
print("\n" + "="*70)
print("STEP 3: PYTORCH INTEGRATION")
print("="*70)

from torch.utils.data import DataLoader

# Convert to PyTorch
train_torch = train_dataset.to_torch()
val_torch = val_dataset.to_torch()

# Create data loaders
train_loader = DataLoader(train_torch, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_torch, batch_size=32, shuffle=False, num_workers=4)

print(f"\nDataLoaders created:")
print(f"  Train batches: {len(train_loader)}")
print(f"  Val batches: {len(val_loader)}")

# Test iteration
batch = next(iter(train_loader))
print(f"\nSample batch keys: {list(batch.keys())}")

# ===== STEP 4: Training Loop (Skeleton) =====
print("\n" + "="*70)
print("STEP 4: TRAINING (SKELETON)")
print("="*70)

num_epochs = 10
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    
    # Training phase
    for batch_idx, batch in enumerate(train_loader):
        # Your training code here
        pass
    
    # Validation phase
    for batch in val_loader:
        # Your validation code here
        pass

print("\nWorkflow complete!")
loader.close_resources()
```

---

### Example 2: Advanced Analysis with Visualization

```python
from hoops_ai.dataset import DatasetExplorer
from hoops_ai.insights import DatasetViewer
import matplotlib.pyplot as plt
import numpy as np

# Initialize explorer
explorer = DatasetExplorer(flow_output_file="cad_pipeline.flow")

# ===== Multi-Dimensional Analysis =====

# 1. Face area distribution
face_dist = explorer.create_distribution(key="face_areas", group="faces", bins=30)

# 2. Edge length distribution
edge_dist = explorer.create_distribution(key="edge_lengths", group="edges", bins=30)

# 3. Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot face area histogram
ax1 = axes[0, 0]
bin_centers = 0.5 * (face_dist['bin_edges'][1:] + face_dist['bin_edges'][:-1])
ax1.bar(bin_centers, face_dist['hist'], width=(face_dist['bin_edges'][1] - face_dist['bin_edges'][0]))
ax1.set_xlabel('Face Area')
ax1.set_ylabel('Count')
ax1.set_title('Face Area Distribution')

# Plot edge length histogram
ax2 = axes[0, 1]
bin_centers = 0.5 * (edge_dist['bin_edges'][1:] + edge_dist['bin_edges'][:-1])
ax2.bar(bin_centers, edge_dist['hist'], width=(edge_dist['bin_edges'][1] - edge_dist['bin_edges'][0]))
ax2.set_xlabel('Edge Length')
ax2.set_ylabel('Count')
ax2.set_title('Edge Length Distribution')

# Plot file count per bin
ax3 = axes[1, 0]
file_counts = [len(files) for files in face_dist['file_ids_in_bins']]
ax3.plot(bin_centers, file_counts, marker='o')
ax3.set_xlabel('Face Area')
ax3.set_ylabel('Number of Files')
ax3.set_title('Files per Face Area Bin')

# Plot complexity distribution
complexity_stats = explorer.get_array_statistics(group_name="faces", array_name="complexity_level")
ax4 = axes[1, 1]
ax4.text(0.1, 0.9, f"Mean: {complexity_stats['mean']:.2f}", transform=ax4.transAxes)
ax4.text(0.1, 0.8, f"Std: {complexity_stats['std']:.2f}", transform=ax4.transAxes)
ax4.text(0.1, 0.7, f"Min: {complexity_stats['min']:.2f}", transform=ax4.transAxes)
ax4.text(0.1, 0.6, f"Max: {complexity_stats['max']:.2f}", transform=ax4.transAxes)
ax4.set_title('Dataset Statistics')
ax4.axis('off')

plt.tight_layout()
plt.savefig('dataset_analysis.png', dpi=300)
plt.show()

# ===== Visual Inspection =====

# Get high complexity files for visual inspection
high_complexity_filter = lambda ds: ds['complexity_level'] >= 4
complex_file_codes = explorer.get_file_list(group="faces", where=high_complexity_filter)

# Use DatasetViewer for visual inspection
viewer = DatasetViewer.from_explorer(explorer)
fig = viewer.show_preview_as_image(
    complex_file_codes[:25],  # First 25 complex files
    k=25,
    grid_cols=5,
    label_format='id',
    figsize=(15, 12)
)
plt.savefig('complex_files_preview.png', dpi=300)
plt.show()

explorer.close()
```

---

## Best Practices

### For DatasetExplorer

1. **Use flow_output_file parameter**: Simplifies initialization and ensures correct file paths
   ```python
   explorer = DatasetExplorer(flow_output_file="path/to/flow.flow")
   ```

2. **Close resources**: Always close when done to free memory and Dask workers
   ```python
   explorer.close(close_dask=True)
   ```

3. **Check available groups first**: Use `available_groups()` and `available_arrays()` before querying
   ```python
   groups = explorer.available_groups()
   if 'faces' in groups:
       face_data = explorer.get_group_data('faces')
   ```

4. **Print table of contents early**: Understand dataset structure before analysis
   ```python
   explorer.print_table_of_contents()
   ```

### For DatasetLoader


1. **Set random_state**: Ensure reproducible splits
   ```python
   loader.split(key="label", random_state=42)
   ```

2. **Clean up resources**: Close explorer and clear caches
   ```python
   loader.close_resources(clear_split_history=True)
   ```

---

## Performance Considerations

### Memory Management

**DatasetExplorer:**
- Uses Dask for out-of-core processing (data larger than RAM)
- Zarr chunking enables partial array loading
- Configure Dask workers based on available memory:
  ```python
  dask_params = {
      'n_workers': 4,
      'threads_per_worker': 2,
      'memory_limit': '8GB'  # Per worker
  }
  ```

**DatasetLoader:**
- Keeps only indices in memory, not full data
- Custom loaders should be memory-efficient
- Use PyTorch DataLoader `num_workers` for parallel loading


### Parallel Processing

**DatasetExplorer Parallelism:**
- Distribution computation: Dask parallel histogram
- Cross-group queries: Parallel joins
- Subgraph search: Parallel pattern matching

**DatasetLoader Parallelism:**
- PyTorch DataLoader `num_workers`: Controls loading parallelism
- Set based on CPU cores: `num_workers = min(4, cpu_count())`
- Use `pin_memory=True` for GPU training

---

## Summary

**DatasetExplorer and DatasetLoader** provide a complete solution for dataset analysis and ML preparation:

### DatasetExplorer: Analysis & Exploration
- ✅ Query arrays by group and key
- ✅ Analyze distributions with histograms
- ✅ Filter files by metadata conditions
- ✅ Statistical analysis and visualization
- ✅ Cross-group queries and joins
- ✅ Subgraph pattern matching (GNNs)

### DatasetLoader: ML Preparation
- ✅ Stratified train/val/test splitting
- ✅ Multi-label stratification support
- ✅ Framework-agnostic CADDataset
- ✅ PyTorch integration with `.to_torch()`
- ✅ Custom item loaders for preprocessing
- ✅ Data cleaning and validation

### Integration with HOOPS AI Pipeline
- ✅ Automatic consumption of DatasetMerger outputs
- ✅ Schema-driven group and array discovery
- ✅ Seamless connection to Flow-based workflows
- ✅ Support for visualization assets (PNG, 3D cache)

These tools complete the HOOPS AI data pipeline, enabling users to go from raw CAD files to trained ML models with minimal custom code.
