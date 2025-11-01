# HOOPS AI Flow - Quick Reference

## Architecture Overview

HOOPS AI Flow uses a **decorator-based task orchestration architecture** where data flows through a pipeline of parallelizable tasks. The framework automatically manages:

- **Task Dependencies**: Data flows from task outputs to task inputs
- **Parallel Execution**: Process multiple files concurrently using ProcessPoolExecutor
- **Storage Management**: Automatic creation and cleanup of data stores
- **Schema Routing**: Organize encoded data into structured groups for ML consumption

### Core Concepts

1. **Tasks**: Python functions decorated with `@flowtask` that process data
2. **Flow**: Orchestrator that chains tasks and manages execution
3. **Schema**: Blueprint defining how encoded data should be organized
4. **Storage**: Zarr-based persistence layer for encoded features
5. **Explorer**: Query interface for merged datasets

---

## Quick Start (3 Steps)

### 1. Define Schema and Tasks (cad_tasks.py)

```python
import os
import glob
import hoops_ai
from hoops_ai.flowmanager import flowtask
from hoops_ai.storage.datasetstorage.schema_builder import SchemaBuilder

# Set license at module level for worker processes
hoops_ai.set_license(os.getenv("HOOPS_AI_LICENSE"), validate=False)

# Define schema at module level
builder = SchemaBuilder(domain="CAD_analysis", version="1.0")
group = builder.create_group("faces", "face", "Face data")
group.create_array("areas", ["face"], "float32", "Face areas")
cad_schema = builder.build()

@flowtask.extract(name="gather", inputs=["sources"], outputs=["files"])
def gather_files(sources):
    all_files = []
    for source in sources:
        all_files.extend(glob.glob(f"{source}/**/*.step", recursive=True))
    return all_files

@flowtask.transform(name="encode", inputs=["cad_file", "cad_loader", "storage"], 
                    outputs=["encoded"])
def encode_data(cad_file, cad_loader, storage):
    cad_model = cad_loader.create_from_file(cad_file)
    storage.set_schema(cad_schema)
    # Extract features...
    storage.compress_store()
    return storage.get_file_path("")
```

### 2. Create and Execute Flow (Notebook)

```python
from cad_tasks import gather_files, encode_data
import hoops_ai

flow = hoops_ai.create_flow(
    name="cad_pipeline",
    tasks=[gather_files, encode_data],
    flows_outputdir="./output",
    max_workers=8,
    auto_dataset_export=True
)

output, summary, flow_file = flow.process(inputs={'sources': ["/path/to/cad"]})
```

### 3. Query and Use Dataset

```python
from hoops_ai.dataset import DatasetExplorer

explorer = DatasetExplorer(flow_output_file=flow_file)
explorer.print_table_of_contents()

# Query files by condition
file_list = explorer.get_file_list(
    group="faces",
    where=lambda ds: ds['face_count'] > 100
)
```

---

## API Reference

### Task Decorators

#### @flowtask.extract
**Purpose**: Gather input data (files, database queries, etc.)

```python
@flowtask.extract(
    name="task_name",           # Optional: defaults to function name
    inputs=["sources"],          # Keys from flow.process(inputs={...})
    outputs=["files"],           # Keys passed to next task
    parallel_execution=True      # Default: True
)
def gather_files(sources: List[str]) -> List[str]:
    """
    Args:
        sources: Input data from flow.process()
    Returns:
        List of items to process (e.g., file paths)
    """
    return [...]
```

#### @flowtask.transform
**Purpose**: Process individual items (CAD encoding, feature extraction)

```python
@flowtask.transform(
    name="encode",
    inputs=["cad_file", "cad_loader", "storage"],  # Framework injects loader & storage
    outputs=["encoded_path"],
    parallel_execution=True
)
def encode_cad(cad_file: str, cad_loader, storage) -> str:
    """
    Args:
        cad_file: Single file from previous task output
        cad_loader: HOOPSLoader instance (auto-injected)
        storage: DataStorage instance (auto-injected)
    Returns:
        Path to encoded data file
    """
    return "path/to/encoded.data"
```

#### @flowtask.custom
**Purpose**: Aggregation, validation, or custom logic

```python
@flowtask.custom(
    name="aggregate",
    inputs=["encoded_files"],
    outputs=["summary"],
    parallel_execution=False  # Typically sequential
)
def aggregate_results(encoded_files: List[str]) -> dict:
    """Custom processing logic"""
    return {"summary": "..."}
```

### Flow Configuration

```python
hoops_ai.create_flow(
    name: str,                    # Flow identifier
    tasks: List[callable],        # Decorated task functions
    flows_outputdir: str,         # Output directory
    max_workers: int = None,      # Parallel workers (None = auto-detect)
    debug: bool = False,          # True = sequential execution
    auto_dataset_export: bool = True,  # Auto-merge encoded data
    ml_task: str = ""            # Description for documentation
)
```

**Returns**: `(FlowOutput, dict, str)`
- `FlowOutput`: Detailed execution results
- `dict`: Summary with keys: `file_count`, `flow_data`, `flow_info`, `Duration [seconds]`
- `str`: Path to `.flow` file

### Schema Builder API

```python
from hoops_ai.storage.datasetstorage.schema_builder import SchemaBuilder

# Initialize schema
builder = SchemaBuilder(domain="MyDomain", version="1.0")

# Create data group
group = builder.create_group(
    name="faces",              # Group name
    base_dimension="face",     # Base dimension for arrays
    description="Face data"    # Documentation
)

# Add arrays to group
group.create_array(
    name="areas",              # Array name
    dims=["face"],             # Dimensions
    dtype="float32",           # Data type
    description="Face areas"   # Documentation
)

# Define metadata routing
builder.define_file_metadata("processing_time", "float32", "Processing time")
builder.define_categorical_metadata("category", "int32", "Part category")

# Build schema
schema = builder.build()
```

### DatasetExplorer API

```python
from hoops_ai.dataset import DatasetExplorer

explorer = DatasetExplorer(flow_output_file="path/to/flow.flow")

# View dataset structure
explorer.print_table_of_contents()

# Get available groups
groups = explorer.available_groups()  # Returns: ['faces', 'edges', ...]

# Query files
file_list = explorer.get_file_list(
    group="faces",
    where=lambda ds: ds['face_count'] > 100
)

# Create distribution
dist = explorer.create_distribution(
    key="category",
    group="labels",
    bins=10  # None = auto-bin
)
```

### DatasetLoader API

```python
from hoops_ai.dataset import DatasetLoader

loader = DatasetLoader(
    merged_store_path="path/to/flow.dataset",
    parquet_file_path="path/to/flow.infoset"
)

# Split dataset
train_size, val_size, test_size = loader.split(
    key="category",            # Column to stratify by
    group="labels",            # Group containing the key
    train=0.7,
    validation=0.15,
    test=0.15,
    random_state=42
)

# Get PyTorch dataset
train_dataset = loader.get_dataset("train")

loader.close_resources()
```

---

## Common Usage Patterns

### Pattern 1: CAD Encoding Pipeline

```python
@flowtask.transform(name="encode", inputs=["cad_file", "cad_loader", "storage"])
def encode_cad(cad_file, cad_loader, storage):
    # Load CAD file
    cad_model = cad_loader.create_from_file(cad_file)
    storage.set_schema(cad_schema)
    
    # Extract features
    from hoops_ai.cadencoder import BrepEncoder
    brep_encoder = BrepEncoder(cad_model.get_brep(), storage)
    brep_encoder.push_face_attributes()
    
    # Save custom metadata
    storage.save_metadata("face_count", cad_model.get_face_count())
    
    # Finalize
    storage.compress_store()
    return storage.get_file_path("")
```

### Pattern 2: Multi-Source Gathering

```python
@flowtask.extract(name="gather", inputs=["sources"], outputs=["files"])
def gather_files(sources):
    all_files = []
    for source in sources:
        all_files.extend(glob.glob(f"{source}/**/*.step", recursive=True))
    return all_files
```

### Pattern 3: Filtered Dataset Querying

```python
# Complex query with lambda
high_complexity = lambda ds: (ds['face_count'] > 100) & (ds['category'] == 5)
file_list = explorer.get_file_list(group="labels", where=high_complexity)

# Visualize results
from hoops_ai.insights import DatasetViewer
viewer = DatasetViewer.from_explorer(explorer)
viewer.show_preview_as_image(file_list, grid_cols=5)
```

---

## Output File Structure

```
flows_outputdir/flows/{flow_name}/
├── {flow_name}.flow          # Main output: JSON with all metadata
├── {flow_name}.dataset       # Merged Zarr dataset
├── {flow_name}.infoset       # File-level metadata (Parquet)
├── {flow_name}.attribset     # Categorical metadata (Parquet)
├── error_summary.json        # Errors encountered during processing
├── flow_log.log              # Detailed execution log
├── encoded/                  # Individual .data files (Zarr format)
└── stream_cache/             # PNG previews for visualization
```

---

## Windows ProcessPoolExecutor Requirements

**Critical**: On Windows, parallel execution uses separate processes (not threads). This requires:

### ✅ Required Setup

1. **Define tasks in `.py` files** (e.g., `cad_tasks.py`)
2. **Set license at module level**:
   ```python
   hoops_ai.set_license(os.getenv("HOOPS_AI_LICENSE"), validate=False)
   ```
3. **Define schema at module level**:
   ```python
   cad_schema = builder.build()  # Global variable
   ```
4. **Import tasks in notebook**:
   ```python
   from cad_tasks import gather_files, encode_data
   ```

### ❌ Will Fail

- Defining tasks in notebook cells
- Setting license only in notebook
- Defining schema only in notebook

---

## Debugging

### Sequential Execution Mode

```python
flow = hoops_ai.create_flow(..., debug=True)  # Sequential, easier to debug
```

### Check Execution Logs

```python
import json

# View errors
with open("output/flows/my_flow/error_summary.json") as f:
    errors = json.load(f)
    for err in errors:
        print(f"File: {err['file']}, Error: {err['message']}")

# View execution log
with open("output/flows/my_flow/flow_log.log") as f:
    print(f.read())
```

---

## Best Practices

1. **Start Small**: Test with 10-100 files before scaling
2. **Use Schemas**: Always define schemas for predictable data organization
3. **Handle Errors Gracefully**: Framework collects errors; inspect after execution
4. **Monitor Resources**: Check memory usage during large dataset processing
5. **Version Control**: Track schemas and task definitions in git

---

**See [Flow_Documentation.md](Flow_Documentation.md) for detailed architecture and advanced patterns.**
