# DatasetMerger Documentation (Automatic executed during FLOW with auto_dataset_export=True)

## Overview

The **DatasetMerger** is a critical component in HOOPS AI's data pipeline that consolidates multiple individual encoded CAD files (`.data` files) into a single, unified, compressed dataset (`.dataset` file). It performs parallel array concatenation, metadata aggregation, and schema-driven organization to create ML-ready datasets.

**Key Purpose**: The DatasetMerger transforms a collection of per-file encoded data into a unified, queryable, analysis-ready dataset suitable for machine learning training and exploration.

The system follows an automated pipeline pattern:
```
Individual .data files → DatasetMerger → Unified .dataset → DatasetExplorer/DatasetLoader → ML Pipeline
```

**Note**: This document focuses on the **merging process** (consolidation of individual files). For information on **using** the merged datasets (analysis, querying, ML preparation), see:
- **[DatasetExplorer_DatasetLoader_Documentation.md](./DatasetExplorer_DatasetLoader_Documentation.md)** - Comprehensive guide for dataset exploration and ML training preparation

---

## Table of Contents

1. [Architecture & Role](#architecture--role)
   - [Position in HOOPS AI Pipeline](#position-in-hoops-ai-pipeline)
   - [Integration with Flow Module](#integration-with-flow-module)
2. [Core Concepts](#core-concepts)
   - [Data Organization](#data-organization)
   - [Group-Based Merging](#group-based-merging)
   - [Schema-Driven vs Heuristic Discovery](#schema-driven-vs-heuristic-discovery)
3. [DatasetMerger Class](#datasetmerger-class)
   - [Initialization](#initialization)
   - [Group Discovery](#group-discovery)
   - [Merging Logic](#merging-logic)
4. [DatasetInfo Class](#datasetinfo-class)
   - [Metadata Processing](#metadata-processing)
   - [Schema-Driven Routing](#schema-driven-routing)
5. [Output Files](#output-files)
   - [.dataset Files](#dataset-files)
   - [.infoset Files](#infoset-files)
   - [.attribset Files](#attribset-files)
6. [Usage with DatasetExplorer and DatasetLoader](#usage-with-datasetexplorer-and-datasetloader)
7. [Complete Workflow Example](#complete-workflow-example)
8. [Performance Considerations](#performance-considerations)

---

## Architecture & Role

### Position in HOOPS AI Pipeline

The DatasetMerger operates in the **ETL (Extract-Transform-Load) phase** of the HOOPS AI pipeline:

```
┌──────────────────────────────────────────────────────────────────────┐
│                     HOOPS AI Data Pipeline                           │
└──────────────────────────────────────────────────────────────────────┘

1. ENCODING PHASE (Per-File)
   ┌─────────────────────────────────────────────────────┐
   │  CAD File → Encoder → DataStorage → .data file      │
   │  (Repeated for each file in parallel)               │
   └─────────────────────────────────────────────────────┘
                            ↓
2. MERGING PHASE (DatasetMerger) ← YOU ARE HERE
   ┌─────────────────────────────────────────────────────┐
   │  Multiple .data files → DatasetMerger → .dataset    │
   │  Multiple .json files → DatasetInfo → .infoset      │
   │                                      → .attribset    │
   └─────────────────────────────────────────────────────┘
                            ↓
3. ANALYSIS/ML PHASE
   ┌─────────────────────────────────────────────────────┐
   │  .dataset → DatasetExplorer → Query/Filter/Analyze  │
   │  .dataset → DatasetLoader → Train/Val/Test Splits   │
   │                           → ML Model Training        │
   └─────────────────────────────────────────────────────┘
```

**Why Merging is Essential:**
- **Unified Access**: ML models need a single dataset, not thousands of individual files
- **Efficient Queries**: Merged data enables fast filtering and statistical analysis
- **Memory Efficiency**: Dask-based chunked operations handle datasets larger than RAM
- **Metadata Correlation**: File-level and categorical metadata linked to data arrays

---

### Integration with Flow Module

The DatasetMerger is **automatically invoked** by the Flow module through the `AutoDatasetExportTask`:

```python
# User-facing Flow with decorator pattern (automatic merging)
import hoops_ai
from hoops_ai.flowmanager import flowtask
from hoops_ai.storage import DataStorage

# Define tasks using decorators
@flowtask.extract(
    name="gather cad files",
    inputs=["cad_datasources"],
    outputs=["cad_dataset"],
    parallel_execution=True
)
def gather_files(source: str) -> List[str]:
    # Your file gathering logic
    return file_list

@flowtask.transform(
    name="encode cad data",
    inputs=["cad_dataset"],
    outputs=["cad_files_encoded"],
    parallel_execution=True
)
def encode_cad_data(cad_file: str, cad_loader: HOOPSLoader, storage: DataStorage) -> str:
    # Your encoding logic
    storage.set_schema(cad_schema)
    # ... encode data ...
    storage.compress_store()
    return storage.get_file_path("")

# Create and execute flow
flow = hoops_ai.create_flow(
    name="my_cad_pipeline",
    tasks=[gather_files, encode_cad_data],
    max_workers=8,
    flows_outputdir="./output",
    auto_dataset_export=True  # Enable automatic dataset merging
)

# AutoDatasetExportTask is automatically injected here
# - No explicit task needed by user
# - Runs after all encoding completes
# - Produces .dataset, .infoset, .attribset files

flow_output, output_dict, flow_file = flow.process(
    inputs={'cad_datasources': ['./cad_files']}
)
```

**Automatic Injection Logic:**

The Flow manager detects transform tasks (decorated with `@flowtask.transform`) and automatically injects `AutoDatasetExportTask` when `auto_dataset_export=True`:

```python
# Inside hoops_ai.create_flow() (automatic behavior)
# When auto_dataset_export=True and transform tasks are present:
#   1. Flow detects @flowtask.transform decorated tasks
#   2. Automatically injects AutoDatasetExportTask after all transforms
#   3. No manual task creation needed by user

flow = hoops_ai.create_flow(
    name="my_pipeline",
    tasks=[gather_files, encode_cad_data],
    auto_dataset_export=True  # Triggers automatic injection
)
```

**What AutoDatasetExportTask Does:**
1. Collects all `.data` files from transform tasks
2. Collects all `.json` metadata files
3. Creates a `DatasetInfo` instance to process metadata
4. Creates a `DatasetMerger` instance to merge data arrays
5. Produces output files: `{flow_name}.dataset`, `{flow_name}.infoset`, `{flow_name}.attribset`

**Output Dictionary Keys:**
After flow execution, the output dictionary contains:
- `'flow_data'`: Path to merged `.dataset` file
- `'flow_info'`: Path to `.infoset` metadata file
- `'flow_attributes'`: Path to `.attribset` categorical data file
- `'file_count'`: Number of files processed

---

## Core Concepts

### Data Organization

Individual encoded files have a group-based structure:

```
part_001.data (ZipStore Zarr)
├── faces/
│   ├── face_indices (array)
│   ├── face_areas (array)
│   ├── face_types (array)
│   └── face_uv_grids (array)
├── edges/
│   ├── edge_indices (array)
│   ├── edge_lengths (array)
│   └── edge_types (array)
├── graph/
│   ├── edges_source (array)
│   ├── edges_destination (array)
│   └── num_nodes (array)
└── metadata.json
```

After merging, the dataset consolidates all files:

```
my_flow.dataset (ZipStore Zarr)
├── faces/
│   ├── face_indices (concatenated from all files)
│   ├── face_areas (concatenated from all files)
│   └── file_id (provenance tracking: which face came from which file)
├── edges/
│   └── ... (similarly concatenated)
└── graph/
    └── ... (similarly concatenated)
```

---

### Group-Based Merging

The DatasetMerger organizes arrays into **logical groups** for concatenation:

**Mathematical Representation:**

For a group $G$ with primary dimension $d$ (e.g., "face"), given $N$ files with arrays:

$$
A_i^G = \{a_{i,1}, a_{i,2}, \ldots, a_{i,n_i}\}
$$

where $a_{i,j}$ is the $j$-th element of array $A$ from file $i$, and $n_i$ is the count of dimension $d$ in file $i$.

The merged array is:

$$
A_{\text{merged}}^G = A_1^G \oplus A_2^G \oplus \cdots \oplus A_N^G
$$

where $\oplus$ denotes concatenation along dimension $d$:

$$
A_{\text{merged}}^G = \{a_{1,1}, \ldots, a_{1,n_1}, a_{2,1}, \ldots, a_{2,n_2}, \ldots, a_{N,1}, \ldots, a_{N,n_N}\}
$$

**Example:**

```
File 1: face_areas = [1.5, 2.3, 4.1]  (3 faces)
File 2: face_areas = [3.7, 5.2]        (2 faces)
File 3: face_areas = [2.1, 1.8, 6.3]  (3 faces)

Merged: face_areas = [1.5, 2.3, 4.1, 3.7, 5.2, 2.1, 1.8, 6.3]  (8 faces total)
```

**Provenance Tracking:**

A `file_id` array is added to track origin:

```
file_id = [0, 0, 0, 1, 1, 2, 2, 2]
         └─File 1─┘ └File 2┘ └─File 3─┘
```

---

### Schema-Driven vs Heuristic Discovery

The DatasetMerger supports two modes for discovering data structure:

#### 1. Schema-Driven Discovery (Preferred)

When a schema is available (from `SchemaBuilder.set_schema()`), the merger uses explicit group definitions:

```python
schema = {
    "groups": {
        "faces": {
            "primary_dimension": "face",
            "arrays": {
                "face_indices": {"dims": ["face"], "dtype": "int32"},
                "face_areas": {"dims": ["face"], "dtype": "float32"}
            }
        },
        "edges": {
            "primary_dimension": "edge",
            "arrays": {
                "edge_lengths": {"dims": ["edge"], "dtype": "float32"}
            }
        }
    }
}

merger.set_schema(schema)  # Use explicit structure
```

**Benefits:**
- **Predictable**: Structure is explicit and documented
- **Validated**: Arrays are checked against schema dimensions
- **Extensible**: New groups/arrays follow defined patterns

#### 2. Heuristic Discovery (Fallback)

Without a schema, the merger scans files and uses naming patterns to infer structure:

```python
# Heuristic rules:
# - Arrays with "face" in name → "faces" group
# - Arrays with "edge" in name → "edges" group
# - Arrays with "graph" in name → "graph" group
# - Arrays in "graph/edges/" subgroup → flattened to "graph/edges_source"

discovered_groups = merger.discover_groups_from_files(max_files_to_scan=5)
```

**Pattern Matching:**
```python
if "face" in array_name.lower():
    group = "faces"
elif "edge" in array_name.lower():
    group = "edges"
elif "duration" in array_name.lower() or "size" in array_name.lower():
    group = "file"  # File-level metadata
```

---

## DatasetMerger Class

### Initialization

```python
from hoops_ai.storage.datasetstorage import DatasetMerger

merger = DatasetMerger(
    zip_files=["path/to/part_001.data", "path/to/part_002.data", ...],
    merged_store_path="merged_dataset.dataset",
    file_id_codes={"part_001": 0, "part_002": 1, ...},  # From DatasetInfo
    dask_client_params={'n_workers': 4, 'threads_per_worker': 4},
    delete_source_files=True  # Clean up after merging
)
```

**Parameters:**
- `zip_files` (List[str]): Paths to individual `.data` files to merge
- `merged_store_path` (str): Output path for merged `.dataset` file
- `file_id_codes` (Dict[str, int]): Mapping from file stem to integer ID (provenance)
- `dask_client_params` (Dict): Configuration for parallel processing
- `delete_source_files` (bool): Whether to delete source files after successful merge

---

### Group Discovery

#### Automatic Discovery

```python
# Discover groups from input files
discovered_groups = merger.discover_groups_from_files(max_files_to_scan=5)

# Print discovered structure
merger.print_discovered_structure()
# Output:
# Discovered Data Structure:
# ==================================================
# Group: faces
#   Arrays: ['face_indices', 'face_areas', 'face_types', 'face_uv_grids']
# Group: edges
#   Arrays: ['edge_indices', 'edge_lengths', 'edge_types']
# Group: graph
#   Arrays: ['edges_source', 'edges_destination', 'num_nodes']
```

#### Manual Schema Setting

```python
from hoops_ai.storage.datasetstorage import SchemaBuilder

# Build schema
builder = SchemaBuilder(domain="CAD_analysis")
faces_group = builder.create_group("faces", "face", "Face data")
faces_group.create_array("face_areas", ["face"], "float32")
schema = builder.build()

# Apply schema to merger
merger.set_schema(schema)
```

---

### Merging Logic

#### Single-Pass Merge

For small datasets (all files fit in memory):

```python
merger.merge_data(
    face_chunk=500_000,           # Chunk size for face arrays
    edge_chunk=500_000,           # Chunk size for edge arrays
    faceface_flat_chunk=100_000,  # Chunk size for face-face relationships
    batch_size=None,              # None = merge all at once
    consolidate_metadata=True,    # Consolidate Zarr metadata
    force_compression=True        # Output as compressed .dataset
)
```

**Process:**
1. Load all `.data` files into xarray Datasets organized by group
2. Concatenate arrays within each group along primary dimension
3. Add `file_id` provenance tracking
4. Write merged data to `.dataset` file

---

#### Batch Merge

For large datasets (memory-constrained):

```python
merger.merge_data(
    face_chunk=500_000,
    edge_chunk=500_000,
    faceface_flat_chunk=100_000,
    batch_size=200,  # Process 200 files at a time
    consolidate_metadata=True,
    force_compression=True
)
```

**Process:**
1. **Partial Merges**: Process files in batches of `batch_size`, creating partial `.dataset` files
2. **Progressive Cleanup**: Delete source files after each batch is successfully merged
3. **Final Merge**: Merge all partial datasets into final `.dataset`
4. **Cleanup**: Remove partial datasets

**Mathematical Formulation:**

For $N$ total files with batch size $B$:

$$
\text{Number of batches} = \lceil N / B \rceil
$$

Batch $i$ merges files:

$$
\text{Batch}_i = \{f_{iB+1}, f_{iB+2}, \ldots, f_{\min((i+1)B, N)}\}
$$

Final merge concatenates partial results:

$$
D_{\text{final}} = D_{\text{batch}_1} \oplus D_{\text{batch}_2} \oplus \cdots \oplus D_{\text{batch}_k}
$$

---

#### Special Processing: Matrix Flattening

For face-face relationship arrays (2D matrices per file), the merger flattens them:

**Per-File Structure:**
```
File 1: a3_distance [3 faces × 3 faces × 64 bins]
File 2: a3_distance [2 faces × 2 faces × 64 bins]
```

**Flattening Process:**

$$
\text{Flattened}_{i} = \text{reshape}(A_i, [n_i \times n_i, \text{bins}])
$$

**Merged Structure:**
```
Merged: a3_distance [13 face-pairs × 64 bins]
                     └─ (3×3=9) + (2×2=4) = 13 pairs
```

This enables efficient concatenation and provenance tracking.

---

## DatasetInfo Class

The `DatasetInfo` class handles metadata extraction and routing, working alongside `DatasetMerger`.

### Initialization

```python
from hoops_ai.storage.datasetstorage import DatasetInfo

ds_info = DatasetInfo(
    info_files=["path/to/part_001.json", "path/to/part_002.json", ...],
    merged_store_path="merged_dataset.infoset",      # Output for file-level metadata
    attribute_file_path="merged_dataset.attribset",  # Output for categorical metadata
    schema=schema_dict  # Optional: schema for metadata routing
)
```

---

### Metadata Processing

#### Parse and Route Metadata

```python
# Parse all JSON metadata files
ds_info.parse_info_files()

# Build file ID mappings for provenance
file_id_codes = ds_info.build_code_mappings(zip_files)
# Returns: {"part_001": 0, "part_002": 1, ...}

# Store metadata to Parquet files
ds_info.store_info_to_parquet(table_name="file_info")
```

**What `parse_info_files()` Does:**

1. **Load JSON files**: Read all metadata files
2. **Route metadata**: Classify fields as file-level or categorical
3. **Extract descriptions**: Merge nested description structures
4. **Validate**: Check against schema if provided
5. **Aggregate**: Combine metadata from all files
6. **Store**: Write to `.infoset` and `.attribset` Parquet files
7. **Cleanup**: Delete processed JSON files

---

### Schema-Driven Routing

The DatasetInfo uses schema rules to route metadata fields:

```python
# Schema defines routing rules
schema = {
    "metadata": {
        "file_level": {
            "size_cadfile": {"dtype": "int64", "required": False},
            "processing_time": {"dtype": "float32", "required": False}
        },
        "categorical": {
            "file_label": {"dtype": "int32", "values": [0, 1, 2, 3, 4]}
        },
        "routing_rules": {
            "file_level_patterns": ["size_*", "duration_*", "processing_*"],
            "categorical_patterns": ["*_label", "category", "type"],
            "default_numeric": "file_level",
            "default_categorical": "categorical"
        }
    }
}

ds_info = DatasetInfo(..., schema=schema)
```

**Routing Logic:**

1. **Explicit Check**: Is field defined in `file_level` or `categorical`?
2. **Pattern Match**: Does field name match routing patterns (with wildcards)?
3. **Type-Based Default**: Numeric → file_level, String/Bool → categorical

**Example:**
```python
# Field: "size_cadfile" with value 1024000
# 1. Explicit check: Found in file_level definitions ✓
# 2. Routing: → .infoset file

# Field: "file_label" with value 3
# 1. Explicit check: Found in categorical definitions ✓
# 2. Routing: → .attribset file

# Field: "custom_metric" with value 42.5 (not in schema)
# 1. Explicit check: Not found
# 2. Pattern match: No match
# 3. Type-based default: Numeric → file_level
# 4. Routing: → .infoset file
```

---

## Output Files

The DatasetMerger and DatasetInfo produce three primary output files:

### .dataset Files

**Format**: Compressed Zarr (ZipStore) with `.dataset` extension

**Contents**: All numerical array data organized by groups

**Structure:**
```
my_flow.dataset (ZipStore)
├── faces/
│   ├── face_indices [N_faces]
│   ├── face_areas [N_faces]
│   ├── face_types [N_faces]
│   ├── face_uv_grids [N_faces × U × V × 7]
│   └── file_id [N_faces]  ← Provenance tracking
├── edges/
│   ├── edge_indices [N_edges]
│   ├── edge_lengths [N_edges]
│   ├── edge_types [N_edges]
│   └── file_id [N_edges]  ← Provenance tracking
├── graph/
│   ├── edges_source [N_graph_edges]
│   ├── edges_destination [N_graph_edges]
│   ├── num_nodes [N_files]
│   └── file_id [N_graph_edges]  ← Provenance tracking
└── faceface/
    ├── a3_distance [N_face_pairs × bins]
    ├── d2_distance [N_face_pairs × bins]
    ├── extended_adjacency [N_face_pairs]
    └── file_id [N_face_pairs]  ← Provenance tracking
```

**Access Pattern:**
Use the DatasetExplorer or DatasetLoader to query and analyze data.

---

### .infoset Files

**Format**: Parquet (columnar storage)

**Contents**: File-level metadata (one row per file)

**Schema:**
```
┌────┬──────────────┬─────────────┬───────────────┬──────────────────┬────────┐
│ id │ name         │ description │ size_cadfile  │ processing_time  │ subset │
├────┼──────────────┼─────────────┼───────────────┼──────────────────┼────────┤
│ 0  │ part_001     │ Bracket     │ 1024000       │ 12.5             │ train  │
│ 1  │ part_002     │ Housing     │ 2048000       │ 18.3             │ train  │
│ 2  │ part_003     │ Cover       │ 512000        │ 8.1              │ test   │
└────┴──────────────┴─────────────┴───────────────┴──────────────────┴────────┘
```

**Additional Columns** (schema-dependent):
- `flow_name`: Name of the flow that processed the file
- `stream_cache_png`: Path to PNG visualization
- `stream_cache_3d`: Path to 3D model (SCS/STL)
- Custom fields defined in schema



---

### .attribset Files

**Format**: Parquet (columnar storage)

**Contents**: Categorical metadata and label descriptions

**Schema:**
```
┌──────────────┬────┬──────────────┬─────────────────────┐
│ table_name   │ id │ name         │ description         │
├──────────────┼────┼──────────────┼─────────────────────┤
│ file_label   │ 0  │ Simple       │ Basic geometry      │
│ file_label   │ 1  │ Medium       │ Moderate complexity │
│ file_label   │ 2  │ Complex      │ High complexity     │
│ face_types   │ 0  │ Plane        │ Planar surface      │
│ face_types   │ 1  │ Cylinder     │ Cylindrical surface │
│ face_types   │ 2  │ Sphere       │ Spherical surface   │
│ edge_types   │ 0  │ Line         │ Linear edge         │
│ edge_types   │ 1  │ Arc          │ Circular arc        │
└──────────────┴────┴──────────────┴─────────────────────┘
```


---

## Usage with DatasetExplorer and DatasetLoader

The merged outputs (`.dataset`, `.infoset`, `.attribset`) are consumed by **DatasetExplorer** for analysis and **DatasetLoader** for ML preparation.

### Quick Start

```python
from hoops_ai.dataset import DatasetExplorer, DatasetLoader

# Analysis with DatasetExplorer
explorer = DatasetExplorer(flow_output_file="cad_pipeline.flow")
explorer.print_table_of_contents()

# Query and analyze
face_dist = explorer.create_distribution(key="face_areas", group="faces", bins=20)
file_codes = explorer.get_file_list(group="faces", where=lambda ds: ds['complexity_level'] >= 4)

explorer.close()

# ML Preparation with DatasetLoader
loader = DatasetLoader(
    merged_store_path="cad_pipeline.dataset",
    parquet_file_path="cad_pipeline.infoset"
)

# Stratified split
loader.split(key="complexity_level", train=0.7, validation=0.15, test=0.15)

# Get datasets
train_dataset = loader.get_dataset("train")
val_dataset = loader.get_dataset("validation")

# PyTorch integration
train_torch = train_dataset.to_torch()
```

**For comprehensive documentation**, see:
- **[DatasetExplorer_DatasetLoader_Documentation.md](./DatasetExplorer_DatasetLoader_Documentation.md)** - Complete guide covering:
  - Query operations and distribution analysis
  - Metadata queries and filtering
  - Stratified splitting for ML training
  - PyTorch and custom framework integration
  - Complete workflow examples
  - Performance optimization and troubleshooting

---

## Complete Workflow Example

This section demonstrates a complete end-to-end workflow using the **decorator pattern** with `@flowtask` decorators and `hoops_ai.create_flow()`:

### Step 1: Define Schema

```python
from hoops_ai.storage.datasetstorage import SchemaBuilder

# Build schema for CAD analysis
builder = SchemaBuilder(domain="CAD_analysis", version="1.0")

# Define groups and arrays
faces_group = builder.create_group("faces", "face", "Face geometry data")
faces_group.create_array("face_areas", ["face"], "float32", "Face areas in mm²")
faces_group.create_array("face_types", ["face"], "int32", "Face type codes")

edges_group = builder.create_group("edges", "edge", "Edge geometry data")
edges_group.create_array("edge_lengths", ["edge"], "float32", "Edge lengths in mm")

# Define metadata routing
builder.define_file_metadata("processing_time", "float32", "Encoding time in seconds")
builder.define_categorical_metadata("complexity_level", "int32", "Part complexity (1-5)")

cad_schema = builder.build()
```

---

### Step 2: Define Tasks with Decorators

```python
import hoops_ai
from hoops_ai.flowmanager import flowtask
from hoops_ai.cadaccess import HOOPSLoader
from hoops_ai.storage import DataStorage, CADFileRetriever, LocalStorageProvider
from typing import List
import pathlib

# Task 1: Gather CAD files
@flowtask.extract(
    name="gather cad files",
    inputs=["cad_datasources"],
    outputs=["cad_dataset"],
    parallel_execution=True
)
def gather_cad_files(source: str) -> List[str]:
    """Gather CAD files from source directory"""
    retriever = CADFileRetriever(
        storage_provider=LocalStorageProvider(directory_path=source),
        formats=[".stp", ".step", ".iges", ".igs"]
    )
    return retriever.get_file_list()

# Task 2: Encode CAD data
@flowtask.transform(
    name="encode cad geometry",
    inputs=["cad_dataset"],
    outputs=["cad_files_encoded"],
    parallel_execution=True
)
def encode_cad_geometry(cad_file: str, cad_loader: HOOPSLoader, storage: DataStorage) -> str:
    """Extract geometric features from CAD file and save to storage"""
    import time
    from hoops_ai.cadencoder import BrepEncoder
    from hoops_ai.cadaccess import HOOPSTools
    
    start_time = time.time()
    
    # Set schema for storage
    storage.set_schema(cad_schema)
    
    # Load CAD model
    cad_model = cad_loader.create_from_file(cad_file)
    
    # Adapt model for B-rep extraction
    hoopstools = HOOPSTools()
    hoopstools.adapt_brep(cad_model, None)
    
    # Extract features using BrepEncoder
    brep_encoder = BrepEncoder(cad_model.get_brep(), storage)
    brep_encoder.push_face_indices()
    brep_encoder.push_face_attributes()
    brep_encoder.push_edge_indices()
    brep_encoder.push_edge_attributes()
    
    # Save metadata
    elapsed = time.time() - start_time
    storage.save_metadata("processing_time", elapsed)
    storage.save_metadata("complexity_level", calculate_complexity(cad_model))
    storage.save_metadata("Item", str(cad_file))
    
    # Compress and return path
    storage.compress_store()
    return storage.get_file_path("")

def calculate_complexity(model) -> int:
    """Calculate part complexity (1-5)"""
    # Your complexity calculation logic
    return 3  # Placeholder
```

---

### Step 3: Create and Execute Flow

```python
# Configuration
datasources_dir = "/path/to/cad/files"
flows_outputdir = pathlib.Path("./flow_output")

# Create flow with automatic dataset export
cad_flow = hoops_ai.create_flow(
    name="cad_geometry_pipeline",
    tasks=[gather_cad_files, encode_cad_geometry],
    max_workers=8,  # Parallel processing with 8 workers
    flows_outputdir=str(flows_outputdir),
    ml_task="CAD Geometry Analysis",
    auto_dataset_export=True  # Enable automatic merging
)

# Execute flow
print("Starting CAD processing pipeline...")
flow_output, output_dict, flow_file = cad_flow.process(
    inputs={'cad_datasources': [datasources_dir]}
)

# Display results
print("\n" + "="*70)
print("FLOW EXECUTION COMPLETED")
print("="*70)
print(f"\nDataset files created:")
print(f"  Main dataset:  {output_dict.get('flow_data')}")
print(f"  Info metadata: {output_dict.get('flow_info')}")
print(f"  Attributes:    {output_dict.get('flow_attributes')}")
print(f"  Flow file:     {flow_file}")
print(f"\nFiles processed: {output_dict.get('file_count', 0)}")
print(f"Total time: {output_dict.get('Duration [seconds]', {}).get('total', 0):.2f}s")
```

**Key Points:**
- `@flowtask.extract`: Gathers input files (parallel execution supported)
- `@flowtask.transform`: Processes files with automatic storage lifecycle management
- `hoops_ai.create_flow()`: Orchestrates tasks and injects `AutoDatasetExportTask`
- `auto_dataset_export=True`: Enables automatic merging after encoding
- Framework handles: threading, progress tracking, error logging, cleanup

---

### Step 4: Explore and Prepare ML Dataset

After Flow execution completes, use **DatasetExplorer** and **DatasetLoader** to analyze and prepare the merged dataset for ML training.

```python
from hoops_ai.dataset import DatasetExplorer, DatasetLoader

# Load merged dataset using flow file
explorer = DatasetExplorer(flow_output_file=str(flow_file))

# Print table of contents
explorer.print_table_of_contents()

# Query and analyze (example)
face_area_dist = explorer.create_distribution(key="face_areas", group="faces", bins=20)
print(f"Face area distribution computed")

explorer.close()

# Prepare for ML training
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

print(f"Dataset split: Train={train_size}, Val={val_size}, Test={test_size}")

# Get datasets ready for training
train_dataset = loader.get_dataset("train")
print(f"Training dataset ready with {len(train_dataset)} samples")

loader.close_resources()
```

**For detailed exploration and ML preparation workflows**, see:
**[DatasetExplorer_DatasetLoader_Documentation.md](./DatasetExplorer_DatasetLoader_Documentation.md)**

This comprehensive guide covers:
- Advanced query operations and filtering
- Distribution analysis and visualization
- Statistical summaries and cross-group queries
- Stratified splitting with multi-label support
- PyTorch and custom framework integration
- Custom item loaders and data augmentation
- Complete ML training pipeline examples
- Performance optimization and troubleshooting

---

### Complete Pipeline Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DECORATOR-BASED CAD PIPELINE                     │
└─────────────────────────────────────────────────────────────────────┘

STEP 1: Define Schema
┌────────────────────────────────────────┐
│ SchemaBuilder                          │
│  - Groups (faces, edges)               │
│  - Arrays (areas, lengths, types)      │
│  - Metadata routing rules              │
│  → cad_schema (Dict)                   │
└────────────────────────────────────────┘
                ↓
STEP 2: Define Tasks with Decorators
┌────────────────────────────────────────┐
│ @flowtask.extract                      │
│   gather_cad_files()                   │
│   → List[file_paths]                   │
└────────────────────────────────────────┘
                ↓
┌────────────────────────────────────────┐
│ @flowtask.transform                    │
│   encode_cad_geometry()                │
│   Uses: cad_loader, storage            │
│   → storage.get_file_path("")          │
│   → .data + .json files                │
└────────────────────────────────────────┘
                ↓
STEP 3: Create and Execute Flow
┌────────────────────────────────────────┐
│ hoops_ai.create_flow(                  │
│   tasks=[...],                         │
│   auto_dataset_export=True             │
│ )                                      │
│ flow.process(inputs={...})             │
└────────────────────────────────────────┘
                ↓
        [AUTO-INJECTED]
┌────────────────────────────────────────┐
│ AutoDatasetExportTask                  │
│                                        │
│  ┌──────────────────────────────────┐ │
│  │ DatasetInfo                      │ │
│  │  - Parse .json files             │ │
│  │  - Route metadata by schema      │ │
│  │  - Build file_id mappings        │ │
│  │  → .infoset (Parquet)            │ │
│  │  → .attribset (Parquet)          │ │
│  └──────────────────────────────────┘ │
│                                        │
│  ┌──────────────────────────────────┐ │
│  │ DatasetMerger                    │ │
│  │  - Discover groups from .data    │ │
│  │  - Concatenate arrays by group   │ │
│  │  - Add file_id provenance        │ │
│  │  - Dask parallel processing      │ │
│  │  → .dataset (Zarr compressed)    │ │
│  └──────────────────────────────────┘ │
└────────────────────────────────────────┘
                ↓
OUTPUT FILES (in flow_output/flows/cad_geometry_pipeline/)
┌────────────────────────────────────────┐
│ cad_geometry_pipeline.dataset          │  ← Merged data arrays
│ cad_geometry_pipeline.infoset          │  ← File-level metadata
│ cad_geometry_pipeline.attribset        │  ← Categorical metadata
│ cad_geometry_pipeline.flow             │  ← Flow specification
└────────────────────────────────────────┘
                ↓
STEP 4: Downstream Analysis & ML
┌────────────────────────────────────────┐
│ DatasetExplorer                        │
│  - Query by metadata filters           │
│  - Analyze distributions               │
│  - Statistical summaries               │
└────────────────────────────────────────┘
                ↓
┌────────────────────────────────────────┐
│ DatasetLoader                          │
│  - Train/validation/test split         │
│  - PyTorch DataLoader creation         │
│  - ML model training preparation       │
└────────────────────────────────────────┘
```

---

### Advanced: Optional Custom Merging Task

If you need custom merging behavior beyond automatic merging:

```python
@flowtask.custom(
    name="custom dataset merge",
    inputs=["cad_files_encoded"],
    outputs=["merged_dataset_path"],
    parallel_execution=False
)
def custom_merge_task(encoded_files: List[str]) -> str:
    """Custom merging logic with specific parameters"""
    from hoops_ai.storage.datasetstorage import DatasetMerger, DatasetInfo
    import pathlib
    
    # Find all .data and .json files
    data_files = [f"{f}.data" for f in encoded_files]
    json_files = [f"{f}.json" for f in encoded_files]
    
    output_dir = pathlib.Path("./custom_merged")
    output_dir.mkdir(exist_ok=True)
    
    # Process metadata
    ds_info = DatasetInfo(
        info_files=json_files,
        merged_store_path=str(output_dir / "custom.infoset"),
        attribute_file_path=str(output_dir / "custom.attribset"),
        schema=cad_schema
    )
    ds_info.parse_info_files()
    file_id_codes = ds_info.build_code_mappings(data_files)
    ds_info.store_info_to_parquet()
    
    # Merge data arrays
    merger = DatasetMerger(
        zip_files=data_files,
        merged_store_path=str(output_dir / "custom.dataset"),
        file_id_codes=file_id_codes,
        dask_client_params={'n_workers': 16, 'threads_per_worker': 2},
        delete_source_files=False
    )
    merger.set_schema(cad_schema)
    merger.merge_data(
        face_chunk=1_000_000,  # Custom chunk sizes
        edge_chunk=1_000_000,
        batch_size=500  # Process in larger batches
    )
    
    return str(output_dir / "custom.dataset")

# Use in flow with custom merging
custom_flow = hoops_ai.create_flow(
    name="custom_pipeline",
    tasks=[gather_cad_files, encode_cad_geometry, custom_merge_task],
    auto_dataset_export=False  # Disable automatic merging
)
```

---

## Performance Considerations

### Memory Management

**Batch Merging for Large Datasets:**
```python
# For 10,000+ files, use batch merging
merger.merge_data(
    batch_size=500,  # Process 500 files at a time
    face_chunk=500_000,
    edge_chunk=500_000
)
```

**Benefits:**
- Prevents out-of-memory errors
- Enables progress tracking
- Allows incremental cleanup of source files

---

### Parallel Processing

**Dask Configuration:**
```python
dask_client_params = {
    'n_workers': 8,              # Number of parallel workers
    'threads_per_worker': 4,     # Threads per worker
    'processes': True,           # Use separate processes
    'memory_limit': '8GB',       # Per-worker memory limit
    'dashboard_address': ':8787' # Monitoring dashboard
}

merger = DatasetMerger(
    zip_files=files,
    merged_store_path="output.dataset",
    dask_client_params=dask_client_params
)
```

**Tuning Guidelines:**
- **Many small files**: More workers, fewer threads (I/O bound)
- **Few large files**: Fewer workers, more threads (CPU bound)
- **Memory constraints**: Reduce `memory_limit` and use smaller `batch_size`

---

### Compression

**Zarr Compression Settings:**
- **Codec**: Zstd level 12 (high compression ratio)
- **Chunks**: Automatic sizing (~1M elements per chunk)
- **Filters**: Delta encoding for integer arrays

**Typical Compression Ratios:**
- Raw array data: 10-20x compression
- Face-face histograms: 5-10x compression
- Graph structures: 3-5x compression

---

## Summary

The **DatasetMerger** is the critical bridge between individual encoded CAD files and ML-ready datasets:

- **Automatic Integration**: Seamlessly invoked by Flow module (no manual setup)
- **Schema-Driven**: Uses SchemaBuilder definitions for predictable, validated merging
- **Parallel Processing**: Leverages Dask for efficient large-scale data consolidation
- **Provenance Tracking**: Maintains file_id arrays for traceability
- **Multiple Outputs**: Produces `.dataset` (arrays), `.infoset` (file metadata), `.attribset` (categorical data)

**Output files serve as input for:**
- **DatasetExplorer**: Query, filter, and analyze merged datasets
- **DatasetLoader**: Prepare train/val/test splits for ML training

**Complete System Architecture:**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         HOOPS AI Data Pipeline                              │
└─────────────────────────────────────────────────────────────────────────────┘

STEP 1: SCHEMA DEFINITION (Optional but Recommended)
┌──────────────────────────────────────────────────────────────┐
│  SchemaBuilder                                               │
│  • Define groups (faces, edges, graph)                       │
│  • Define arrays and dimensions                              │
│  • Set metadata routing rules                                │
│  • Build schema dictionary                                   │
└────────────────────────┬─────────────────────────────────────┘
                         │ schema.json
                         ↓
STEP 2: ENCODING (Per-File Parallel Processing)
┌──────────────────────────────────────────────────────────────┐
│  Flow → CADEncodingTask (via ParallelTask)                   │
│                                                               │
│  CAD File 1 → Encoder → DataStorage → part_001.data ────┐    │
│  CAD File 2 → Encoder → DataStorage → part_002.data ────┤    │
│      ...                                                 │    │
│  CAD File N → Encoder → DataStorage → part_N.data ──────┤    │
│                                                          │    │
│  Each .data file (Zarr format) contains:                │    │
│    • faces/face_areas, face_types, face_uv_grids        │    │
│    • edges/edge_lengths, edge_types                     │    │
│    • graph/edges_source, edges_destination              │    │
│                                                          │    │
│  Metadata files (JSON):                                 │    │
│    part_001.json, part_002.json, ..., part_N.json       │    │
└──────────────────────────────────────────────────────────┼───┘
                                                           │
                         ┌─────────────────────────────────┘
                         │ All .data and .json files
                         ↓
STEP 3: AUTOMATIC MERGING (AutoDatasetExportTask)
┌──────────────────────────────────────────────────────────────┐
│  DatasetInfo                                                  │
│  • Load all .json metadata files                             │
│  • Route metadata using schema rules                         │
│  • Create file_id mappings                                   │
│  • Output: {flow_name}.infoset (file-level metadata)         │
│            {flow_name}.attribset (categorical metadata)      │
└────────────────────────┬─────────────────────────────────────┘
                         │
┌────────────────────────┴─────────────────────────────────────┐
│  DatasetMerger                                                │
│  • Discover groups from schemas or heuristics                │
│  • Load all .data files as xarray Datasets                   │
│  • Concatenate arrays by group (faces, edges, graph)         │
│  • Add file_id provenance tracking                           │
│  • Handle special processing (matrix flattening)             │
│  • Output: {flow_name}.dataset (compressed Zarr)             │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         │ Three output files:
                         │  • {flow_name}.dataset
                         │  • {flow_name}.infoset
                         │  • {flow_name}.attribset
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                    OUTPUT FILES                              │
│                                                              │
│  ┌────────────────────────────────────────────┐             │
│  │ {flow_name}.dataset (Zarr/ZipStore)        │             │
│  │ ─────────────────────────────────────      │             │
│  │ • Numerical array data organized by groups │             │
│  │ • faces/, edges/, graph/, faceface/        │             │
│  │ • Each group has file_id for provenance    │             │
│  │ • Compressed for efficient storage          │             │
│  └────────────────────────────────────────────┘             │
│                                                              │
│  ┌────────────────────────────────────────────┐             │
│  │ {flow_name}.infoset (Parquet)              │             │
│  │ ─────────────────────────────────          │             │
│  │ • File-level metadata (one row per file)   │             │
│  │ • Columns: id, name, description,          │             │
│  │   size_cadfile, processing_time, etc.      │             │
│  │ • Queryable with pandas                     │             │
│  └────────────────────────────────────────────┘             │
│                                                              │
│  ┌────────────────────────────────────────────┐             │
│  │ {flow_name}.attribset (Parquet)            │             │
│  │ ─────────────────────────────────────       │             │
│  │ • Categorical metadata & descriptions      │             │
│  │ • Label mappings (id → name → description) │             │
│  │ • Face types, edge types, etc.             │             │
│  └────────────────────────────────────────────┘             │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      │ Consumed by analysis & ML tools
                      ↓
STEP 4: ANALYSIS & ML TRAINING
┌──────────────────────────────────────────────────────────────┐
│  DatasetExplorer                                              │
│  • Query merged datasets (get_array_data)                    │
│  • Statistical analysis (compute_statistics)                 │
│  • Distribution creation (create_distribution)               │
│  • Metadata filtering (filter_files_by_metadata)             │
│  • Visualization support                                     │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│  DatasetLoader                                                │
│  • Load merged datasets for ML training                      │
│  • Stratified train/val/test splitting                       │
│  • PyTorch/TensorFlow adapter support                        │
│  • Custom item loader functions                              │
│  • Batch loading for training loops                          │
└──────────────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────────┐
│  ML Training Pipeline                                         │
│  • Load train/val/test datasets                              │
│  • Create DataLoaders with batching                          │
│  • Train neural networks (GNNs, CNNs, etc.)                  │
│  • Model evaluation and inference                            │
└──────────────────────────────────────────────────────────────┘
```

**Key Integration Points:**

1. **SchemaBuilder → DataStorage**: Schema defines how individual files are organized
2. **DataStorage → DatasetMerger**: Individual `.data` files are merged using schema structure
3. **SchemaBuilder → DatasetInfo**: Schema routes metadata to correct Parquet files
4. **DatasetMerger Output → DatasetExplorer**: Merged `.dataset` enables exploration
5. **DatasetMerger Output → DatasetLoader**: Merged `.dataset` enables ML training

The system enables seamless progression from raw CAD files to production ML models with minimal manual intervention.
