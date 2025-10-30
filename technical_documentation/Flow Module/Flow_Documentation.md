# HOOPS AI: FLOW Documentation

## Overview

The **HOOPS AI Flow** module is the orchestration engine at the heart of the HOOPS AI framework. It provides a decorator-based system for building robust, parallel CAD data processing pipelines that automatically handle data extraction, transformation, merging, and preparation for machine learning workflows.

**Key Purpose:**
Transform CAD files into ML-ready datasets through a simple, declarative API while automatically managing:
- Parallel execution with process pools
- HOOPSLoader instances per worker process
- Comprehensive error handling and logging
- Dataset merging and metadata organization
- Progress tracking and performance monitoring

The Flow module eliminates the complexity of parallel CAD processing, allowing data scientists to focus on defining **what** to process rather than **how** to process it in parallel.

---

## Table of Contents

1. [Architecture & Design Philosophy](#architecture--design-philosophy)
2. [Core Components](#core-components)
3. [Task Decorators: The Foundation](#task-decorators-the-foundation)
4. [Flow Creation and Execution](#flow-creation-and-execution)
5. [Automatic Features](#automatic-features)
6. [Task Definition Patterns](#task-definition-patterns)
7. [Windows ProcessPoolExecutor Requirements](#windows-processpoolexecutor-requirements)
8. [Complete Workflow Example](#complete-workflow-example)
9. [Advanced Topics](#advanced-topics)
10. [Best Practices](#best-practices)

---

## Architecture & Design Philosophy

### The ETL Pattern

HOOPS AI Flow implements the **Extract-Transform-Load (ETL)** pattern for CAD data processing:

```
┌──────────────────────────────────────────────────────────────────────┐
│                    HOOPS AI Flow Architecture                        │
└──────────────────────────────────────────────────────────────────────┘

1. EXTRACT Phase (@flowtask.extract)
   ┌─────────────────────────────────────────────────────┐
   │  User Function: gather_files(source)                │
   │  • Scan directories for CAD files                   │
   │  • Filter by file extensions                        │
   │  • Return list of file paths                        │
   └─────────────────────────────────────────────────────┘
                          ↓
2. TRANSFORM Phase (@flowtask.transform)
   ┌────────────────────────────────────────────────────────┐
   │  User Function: encode_data(cad_file, loader, storage) │
   │  • Load CAD model (HOOPSLoader per worker)             │
   │  • Extract features (BrepEncoder or custom)            │
   │  • Save to structured storage (schema-driven)          │
   │  • Return encoded file path                            │
   └────────────────────────────────────────────────────────┘
                          ↓
3. LOAD Phase (Auto-Injected)
   ┌─────────────────────────────────────────────────────┐
   │  AutoDatasetExportTask                              │
   │  • Merge individual .data files                     │
   │  • Route metadata (.infoset / .attribset)           │
   │  • Create unified .dataset + .flow file             │
   │  • Prepare for DatasetExplorer / ML training        │
   └─────────────────────────────────────────────────────┘
```

### Design Principles

1. **Declarative Over Imperative**: Define **what** to do, not **how** to parallelize
2. **Separation of Concerns**: User logic is isolated from framework infrastructure
3. **Type Safety**: Decorators enforce input/output contracts
4. **Fail-Safe Execution**: Errors are collected, not propagated (tasks continue)
5. **Zero Configuration**: Sensible defaults for parallel execution, storage, and logging

---

## Core Components

### 1. Task Base Classes

All tasks inherit from one of these base classes:

#### **ParallelTask**
- **Purpose**: Process collections of items in parallel
- **Execution**: Uses ProcessPoolExecutor with isolated worker processes
- **Use Cases**: CAD file encoding, feature extraction, batch transformations
- **Key Methods**:
  - `process_item(item)`: User-defined logic for a single item
  - `execute(item_index, item)`: Framework wrapper with error handling
  - `finalize()`: Post-processing, error summarization

#### **SequentialTask**
- **Purpose**: Execute single, non-parallelizable operations
- **Execution**: Runs in the main process
- **Use Cases**: Dataset merging, model training, report generation
- **Key Methods**:
  - `process(inputs)`: User-defined logic operating on all inputs
  - `execute(inputs)`: Framework wrapper

### 2. ParallelExecutor

The **ParallelExecutor** manages the parallel execution infrastructure. This is handled automatically by HOOPS AI Flow.

**Key Features:**

1. **Process Isolation**: Each worker process gets its own:
   - Python interpreter instance
   - HOOPSLoader with independent license
   - Memory space (no GIL contention)
   - Storage handler instances



3. **Dynamic Execution Mode**:
   - `debug=True` → Sequential execution (easy debugging)
   - `debug=False` + `max_workers > 1` → ProcessPoolExecutor
   - `debug=False` + `max_workers = 1` → Still uses ProcessPoolExecutor (for consistency)

### 3. Flow Orchestrator

The **Flow** class manages task execution and data flow:

```python
class Flow:
    def __init__(self, name, specifications, tasks):
        """
        Initialize flow with:
        - name: Unique flow identifier
        - specifications: Configuration dictionary
        - tasks: List of task classes to execute
        """
    
    def process(self, inputs) -> (FlowOutput, dict, str):
        """
        Execute the flow pipeline:
        1. Validate task dependencies
        2. Execute tasks in sequence
        3. Pass outputs to next task inputs
        4. Collect results and errors
        5. Generate .flow summary file
        
        Returns:
            FlowOutput: Full execution results
            dict: Summary statistics
            str: Path to .flow file
        """
```

**Automatic Features:**

- **Dependency Resolution**: Validates that each task's required inputs are available
- **Data Routing**: Automatically connects task outputs to next task's inputs
- **Error Aggregation**: Collects errors without stopping the pipeline
- **Logging**: Centralized logging with task-specific contexts
- **Directory Management**: Creates/cleans flow output directories

---

## Task Decorators: The Foundation

### The `@flowtask` Decorator API

HOOPS AI provides three decorator types:

#### 1. `@flowtask.extract` - Data Extraction

**Purpose**: Gather input data (CAD files, databases, APIs)

```python
@flowtask.extract(
    name="gather_cad_files",          # Task name (default: function name)
    inputs=["cad_datasources"],       # Expected input keys
    outputs=["cad_dataset"],          # Output keys this task produces
    parallel_execution=True           # Enable/disable parallelism
)
def gather_files(source: str) -> List[str]:
    """
    User-defined function to gather CAD files from a source directory.
    
    Args:
        source: Directory path to scan
        
    Returns:
        List of CAD file paths
    """
    return glob.glob(f"{source}/**/*.step", recursive=True)
```

**What Happens Behind the Scenes:**

1. Function is wrapped by `GatherCADFiles` task class
2. Registered in `_registered_extract_functions` global registry
3. Metadata attached: `_task_type`, `_task_name`, `_task_inputs`, `_task_outputs`
4. Function is serialized with `cloudpickle` for worker process distribution

#### 2. `@flowtask.transform` - Data Transformation

**Purpose**: Process individual items (encode CAD files, extract features)

```python
@flowtask.transform(
    name="encode_manufacturing_data",
    inputs=["cad_file", "cad_loader", "storage"],  # Injected by framework
    outputs=["face_count", "edge_count"],
    parallel_execution=True
)
def encode_data(cad_file: str, cad_loader: HOOPSLoader, storage: DataStorage) -> str:
    """
    User-defined function to encode a single CAD file.
    
    Args:
        cad_file: Path to CAD file (from extract phase)
        cad_loader: HOOPSLoader instance (one per worker process)
        storage: DataStorage instance for saving encoded data
        
    Returns:
        Path to encoded .data file
    """
    # Load CAD model
    cad_model = cad_loader.create_from_file(cad_file)
    
    # Extract features
    brep_encoder = BrepEncoder(cad_model.get_brep(), storage)
    brep_encoder.push_face_indices()
    brep_encoder.push_face_attributes()
    
    # Save and compress
    storage.save_data("faces/face_areas", face_areas_array)
    storage.compress_store()
    
    return storage.get_file_path("")
```

**What Happens Behind the Scenes:**

1. Function is wrapped by `EncodingTask` task class
2. Framework automatically provides:
   - `cad_file`: From previous task output
   - `cad_loader`: HOOPSLoader instance (one per worker, initialized with license)
   - `storage`: DataStorage with schema configuration
3. Each worker process runs this function independently
4. Errors are caught and logged, processing continues

#### 3. `@flowtask.custom` - Custom Processing Task (Still under testing)

**Purpose**: Flexible tasks for any processing logic

```python
@flowtask.custom(
    name="calculate_statistics",
    inputs=["encoded_files"],
    outputs=["stats_summary"],
    parallel_execution=False  # Run in main process
)
def compute_stats(encoded_files: List[str]) -> dict:
    """
    Custom task example: Compute statistics across all encoded files.
    """
    return {"file_count": len(encoded_files), "total_size": calculate_size(encoded_files)}
```

### Decorator Metadata

When you apply a decorator, these attributes are attached to your function:

| Attribute | Type | Description |
|-----------|------|-------------|
| `_task_type` | str | `"extract"`, `"transform"`, or `"custom"` |
| `_task_name` | str | User-defined or function name |
| `_task_inputs` | List[str] | Required input keys |
| `_task_outputs` | List[str] | Output keys produced |
| `_parallel_execution` | bool | Enable parallel execution |
| `_task_class` | lambda | Factory for task wrapper class |

---

## Flow Creation and Execution

### Creating a Flow with `hoops_ai.create_flow()`

The module-level function `create_flow()` simplifies flow instantiation:

```python
import hoops_ai
from hoops_ai.flowmanager import flowtask

# Define tasks (see decorator section above)
@flowtask.extract(...)
def gather_files(source): ...

@flowtask.transform(...)
def encode_data(cad_file, cad_loader, storage): ...

# Create flow
cad_flow = hoops_ai.create_flow(
    name="manufacturing_pipeline",          # Flow identifier
    tasks=[gather_files, encode_data],      # List of decorated functions
    flows_outputdir="./output",             # Output directory
    max_workers=8,                          # Parallel workers (None = auto-detect CPU count)
    ml_task="Part Classification",         # Optional ML task description
    debug=False,                            # False = parallel, True = sequential
    auto_dataset_export=True                # Auto-inject dataset merging task
)

# Execute flow
flow_output, summary, flow_file = cad_flow.process(
    inputs={'cad_datasources': ["/path/to/cad/files"]}
)

# Inspect results
print(f"Processed {summary['file_count']} files in {summary['Duration [seconds]']['total']:.2f}s")
print(f"Dataset: {summary['flow_data']}")
print(f"Info: {summary['flow_info']}")
print(f"Flow file: {flow_file}")
```

### Flow Execution Lifecycle

```
┌──────────────────────────────────────────────────────────────────────┐
│                    Flow Execution Lifecycle                          │
└──────────────────────────────────────────────────────────────────────┘

1. Flow Initialization
   • Create flow instance with name and specifications
   • Validate task list (check inputs/outputs)
   • Setup logging infrastructure
   • Initialize ParallelExecutor

2. Flow Processing (cad_flow.process())
   ├─ Pre-Execution
   │  • Clean/create flow output directory
   │  • Log flow task registry
   │  • Initialize available_data dict with user inputs
   │
   ├─ Task Execution Loop (for each task)
   │  ├─ Validate Dependencies
   │  │  • Check all task_inputs are in available_data
   │  │  • Raise error if dependencies missing
   │  │
   │  ├─ Initialize Task Instance
   │  │  • Create task with logger, specifications, flow_name
   │  │  • Inject user function into task wrapper
   │  │
   │  ├─ Execute Task (ParallelTask)
   │  │  • Prepare items from available_data
   │  │  • Submit to ParallelExecutor
   │  │  • Worker initialization (HOOPSLoader setup)
   │  │  • Process items in parallel
   │  │  • Collect results and errors
   │  │  • Update progress bar (tqdm)
   │  │
   │  ├─ Execute Task (SequentialTask)
   │  │  • Gather inputs from available_data
   │  │  • Call task.process(inputs)
   │  │  • Collect outputs
   │  │
   │  ├─ Update Available Data
   │  │  • Add task outputs to available_data
   │  │  • Make outputs available to next tasks
   │  │
   │  └─ Error Handling
   │     • Catch exceptions per item (ParallelTask)
   │     • Log errors, continue processing
   │     • Generate error_summary.json
   │
   └─ Post-Execution
      • Finalize all tasks (error summaries)
      • Generate FlowOutput object
      • Create .flow summary file
      • Close logging handlers
      • Return (FlowOutput, summary_dict, flow_file_path)
```

### Flow Output Structure

After execution, the flow creates this directory structure:

```
flows_outputdir/flows/{flow_name}/
├── {flow_name}.flow              # JSON summary of execution
├── {flow_name}.dataset           # Merged Zarr dataset (compressed)
├── {flow_name}.infoset           # Parquet file with file-level metadata
├── {flow_name}.attribset         # Parquet file with categorical metadata
├── error_summary.json            # Errors encountered during processing
├── flow_log.log                  # Detailed execution log
├── encoded/                      # Individual encoded files (.data)
│   ├── part_001.data
│   ├── part_002.data
│   └── ...
├── files_summary/                # Per-file JSON metadata
│   ├── part_001.json
│   ├── part_002.json
│   └── ...
└── stream_cache/                 # Visualization assets
    ├── png/                      # PNG previews
    └── scs/                      # 3D stream cache files
```

### The `.flow` File

The `.flow` file is a JSON summary containing:

```json
{
    "flow_name": "manufacturing_pipeline",
    "flow_data": "/path/to/manufacturing_pipeline.dataset",
    "flow_info": "/path/to/manufacturing_pipeline.infoset",
    "flow_attributes": "/path/to/manufacturing_pipeline.attribset",
    "Duration [seconds]": {
        "total": 123.45,
        "GatherCADFiles": 5.2,
        "EncodingTask": 110.8,
        "AutoDatasetExportTask": 7.45
    },
    "file_count": 100,
    "error_distribution": {
        "FileNotFoundError": 2,
        "MemoryError": 1
    },
    "Flow Specifications": {
        "max_workers": 8,
        "sequential_mode": false,
        "storage_type": "optimized",
        "ml_task": "Part Classification"
    }
}
```

**Key Purpose**: This file is the **single source of truth** for downstream analysis. It links to:
- `.dataset`: The merged CAD data
- `.infoset`: File-level metadata (names, timestamps, subsets)
- `.attribset`: Categorical metadata (labels, descriptions)

---

## Automatic Features

### 1. Automatic HOOPSLoader Management

**Challenge**: HOOPSLoader is **not thread-safe** and requires one license per concurrent instance.

**Solution**: The framework automatically creates and manages one HOOPSLoader per worker process.



**User Function Receives**:
```python
@flowtask.transform(...)
def encode_data(cad_file, cad_loader, storage):
    # cad_loader = self.cad_access (from worker's Task instance)
    cad_model = cad_loader.create_from_file(cad_file)  # ✓ Thread-safe
```

### 2. Automatic Error Handling

**Philosophy**: One bad CAD file should not crash the entire pipeline.

**Mechanism**: Errors are caught at the item level, logged, and processing continues.

```python
# In ParallelTask.execute()
def execute(self, item_index, item):
    try:
        result = self.process_item(item)
        return {'result': result, 'error': None}
    except Exception as e:
        error_trace = traceback.format_exc()
        return {'result': None, 'error': str(e), 'logs': [error_trace]}
```

**Error Summary**: Automatically generated as `error_summary.json`:

```json
[
    {
        "item_index": 42,
        "item": "/path/to/corrupted_file.step",
        "worker_pid": 12345,
        "error": "Failed to load CAD model: Invalid STEP format",
        "logs": ["Traceback (most recent call last):", "..."]
    }
]
```

### 3. Automatic Progress Tracking

**tqdm Integration**: All parallel tasks show real-time progress:

```
Manufacturing data encoding: 87%|████████▋ | 87/100 [02:15<00:19, 0.67it/s, errors=2]
```

**Progress Bar Features**:
- Current progress (87/100)
- Time elapsed (02:15)
- Time remaining estimate (00:19)
- Processing rate (0.67 items/second)
- Error count (errors=2)

### 4. Automatic Dataset Merging

**Auto-Injection Logic**: When `auto_dataset_export=True`, the framework:

1. **Detects Encoding Tasks**: Scans task list for `@flowtask.transform` tasks
2. **Injects AutoDatasetExportTask**: Adds it after the last encoding task
3. **Merges Data**: Combines all `.data` files into a unified `.dataset`
4. **Routes Metadata**: Splits metadata into `.infoset` (file-level) and `.attribset` (categorical)


**User Benefit**: No need to manually write dataset merging code!

### 5. Automatic Logging

**Logging Hierarchy**:

```
flow_log.log (INFO level)
├─ Flow start/end markers
├─ Task execution summaries
├─ Dependency validation
└─ Task-specific logs
    ├─ Worker process logs (WARNING+)
    ├─ Error traces (ERROR)
    └─ Debug messages (DEBUG, if debug=True)
```

**Example Log Output**:

```
2025-01-15 14:30:00 - INFO - ######### Flow 'manufacturing_pipeline' start #######
2025-01-15 14:30:00 - INFO - Flow Execution Summary
2025-01-15 14:30:00 - INFO - Task 1: gather_cad_files
2025-01-15 14:30:00 - INFO -     Inputs : cad_datasources
2025-01-15 14:30:00 - INFO -     Outputs: cad_dataset
2025-01-15 14:30:05 - INFO - Executing ParallelTask 'gather_cad_files' with 100 items.
2025-01-15 14:30:10 - INFO - Task 2: encode_manufacturing_data
2025-01-15 14:30:10 - INFO - Using ProcessPoolExecutor with 8 worker processes
2025-01-15 14:32:25 - INFO - Time taken: 145.32 seconds
2025-01-15 14:32:25 - INFO - ######### Flow 'manufacturing_pipeline' end ######
```

---

## Task Definition Patterns

### Pattern 1: Simple File Gathering

```python
@flowtask.extract(
    name="gather_cad_files",
    inputs=["cad_datasources"],
    outputs=["cad_dataset"]
)
def gather_files(source: str) -> List[str]:
    """Gather all STEP files from a directory"""
    return glob.glob(f"{source}/**/*.step", recursive=True)
```

### Pattern 2: CAD Encoding with Schema

```python
@flowtask.transform(
    name="encode_brep_features",
    inputs=["cad_file", "cad_loader", "storage"],
    outputs=["encoded_path"]
)
def encode_brep(cad_file: str, cad_loader: HOOPSLoader, storage: DataStorage) -> str:
    """Extract B-Rep features and save to structured storage"""
    # 1. Load CAD model
    cad_model = cad_loader.create_from_file(cad_file)
    
    # 2. Set schema for structured data organization
    storage.set_schema(cad_schema)  # Defined at module level
    
    # 3. Extract features
    hoops_tools = HOOPSTools()
    hoops_tools.adapt_brep(cad_model, None)
    
    brep_encoder = BrepEncoder(cad_model.get_brep(), storage)
    brep_encoder.push_face_indices()
    brep_encoder.push_face_attributes()
    brep_encoder.push_edge_attributes()
    
    # 4. Save custom metadata
    storage.save_metadata("processing_date", datetime.now().isoformat())
    storage.save_metadata("encoder_version", "1.2.3")
    
    # 5. Compress and return path
    storage.compress_store()
    return storage.get_file_path("")
```

### Pattern 3: Multi-Source Data Gathering

```python
@flowtask.extract(
    name="gather_from_multiple_sources",
    inputs=["cad_datasources"],
    outputs=["cad_dataset"]
)
def gather_multi_source(sources: List[str]) -> List[str]:
    """Gather CAD files from multiple directories"""
    all_files = []
    for source in sources:
        all_files.extend(glob.glob(f"{source}/**/*.step", recursive=True))
        all_files.extend(glob.glob(f"{source}/**/*.stp", recursive=True))
    return all_files
```

### Pattern 4: Conditional Processing

```python
@flowtask.transform(
    name="selective_encoding",
    inputs=["cad_file", "cad_loader", "storage"],
    outputs=["encoded_path"]
)
def encode_selective(cad_file: str, cad_loader: HOOPSLoader, storage: DataStorage) -> str:
    """Only encode files meeting certain criteria"""
    cad_model = cad_loader.create_from_file(cad_file)
    
    # Check file size or complexity
    if cad_model.get_face_count() < 10:
        # Skip simple models
        return None
    
    # Continue with encoding
    brep_encoder = BrepEncoder(cad_model.get_brep(), storage)
    brep_encoder.push_face_attributes()
    
    storage.compress_store()
    return storage.get_file_path("")
```

---

## Windows ProcessPoolExecutor Requirements

### The Windows Multiprocessing Challenge

**Problem**: On Windows, the `multiprocessing` module uses **spawn** (not fork), meaning:
- Each worker process starts fresh (no memory inheritance)
- Worker processes must import all code from `.py` files
- Functions defined in notebooks **cannot be pickled** on Windows

**Solution**: Define tasks in separate `.py` files.

### ✅ Correct Pattern: External Task File

**File Structure**:
```
notebooks/
├── 3a_ETL_pipeline_using_flow.ipynb   # Main notebook
└── cad_tasks.py                        # Task definitions (REQUIRED)
```

**cad_tasks.py** (separate file):
```python
"""
CAD Processing Tasks for Manufacturing Analysis

CRITICAL for Windows ProcessPoolExecutor:
1. License: Set at module level from environment variable
2. Schema: Define at module level (not in notebook)
3. Tasks: Define in .py files (not in notebooks)
"""

import os
import glob
import numpy as np
import hoops_ai
from hoops_ai.flowmanager import flowtask
from hoops_ai.cadaccess import HOOPSLoader, HOOPSTools
from hoops_ai.cadencoder import BrepEncoder
from hoops_ai.storage import DataStorage
from hoops_ai.storage.datasetstorage.schema_builder import SchemaBuilder

# ============================================================================
# LICENSE SETUP - Module Level (REQUIRED for worker processes)
# ============================================================================
license_key = os.getenv("HOOPS_AI_LICENSE")
if license_key:
    hoops_ai.set_license(license_key, validate=False)
else:
    print("WARNING: HOOPS_AI_LICENSE environment variable not set")
# ============================================================================

# ============================================================================
# SCHEMA DEFINITION - Module Level (REQUIRED for worker processes)
# ============================================================================
builder = SchemaBuilder(domain="Manufacturing_Analysis", version="1.0")
machining_group = builder.create_group("machining", "part", "Manufacturing data")
machining_group.create_array("machining_category", ["part"], "int32", "Category")
machining_group.create_array("material_type", ["part"], "int32", "Material")
cad_schema = builder.build()
# ============================================================================

@flowtask.extract(name="gather_files", inputs=["cad_datasources"], outputs=["cad_dataset"])
def gather_files(source: str) -> List[str]:
    """Gather CAD files from source directory"""
    return glob.glob(f"{source}/**/*.step", recursive=True)

@flowtask.transform(name="encode_data", inputs=["cad_file", "cad_loader", "storage"], 
                    outputs=["encoded_path"])
def encode_manufacturing_data(cad_file: str, cad_loader: HOOPSLoader, 
                               storage: DataStorage) -> str:
    """Encode CAD file with manufacturing features"""
    # Load model
    cad_model = cad_loader.create_from_file(cad_file)
    storage.set_schema(cad_schema)  # Schema from module level
    
    # Extract features
    hoops_tools = HOOPSTools()
    hoops_tools.adapt_brep(cad_model, None)
    brep_encoder = BrepEncoder(cad_model.get_brep(), storage)
    brep_encoder.push_face_attributes()
    
    # Save manufacturing data
    storage.save_data("machining/machining_category", np.array([1]))
    storage.compress_store()
    return storage.get_file_path("")
```

**Notebook** (imports tasks):
```python
# Cell 1: Set environment variable (before importing)
import os
os.environ["HOOPS_AI_LICENSE"] = "your-license-key"

# Cell 2: Import tasks from external file
from cad_tasks import gather_files, encode_manufacturing_data, cad_schema

# Cell 3: Create and run flow (now works with parallel execution!)
import hoops_ai

cad_flow = hoops_ai.create_flow(
    name="manufacturing_pipeline",
    tasks=[gather_files, encode_manufacturing_data],
    flows_outputdir="./output",
    max_workers=8,  # ✓ Parallel execution now works on Windows!
    debug=False
)

flow_output, summary, flow_file = cad_flow.process(
    inputs={'cad_datasources': ["/path/to/cad"]}
)
```

### ❌ Incorrect Pattern: Notebook-Defined Tasks

```python
# This will FAIL on Windows with ProcessPoolExecutor!
@flowtask.transform(...)
def encode_data(cad_file, cad_loader, storage):  # Defined in notebook
    ...

cad_flow = hoops_ai.create_flow(
    tasks=[encode_data],  # ❌ Cannot pickle notebook-defined function
    max_workers=8  # ❌ Will crash with PicklingError
)
```

**Error Message**:
```
AttributeError: Can't pickle local object '<lambda>'
```

### Why This Pattern is Required

**Worker Process Initialization**:
```
Main Process (Notebook)              Worker Process (Fresh Python)
┌─────────────────────┐              ┌─────────────────────┐
│ Launch worker       │─────spawn───>│ Start Python        │
│                     │              │ Import cad_tasks.py │ ← Reads file from disk
│                     │              │ Load license        │ ← From module level
│                     │              │ Load schema         │ ← From module level
│ Submit task         │──serialize──>│ Execute function    │ ← Uses imported function
│                     │              │                     │
└─────────────────────┘              └─────────────────────┘
```

**Key Points**:
1. Worker processes **cannot see notebook variables**
2. Worker processes **must import from .py files**
3. License/schema **must be set during module import**
4. Functions **must be module-level** (not nested in notebooks)

---

## Complete Workflow Example

Let's walk through a complete example from start to finish.

### Step 1: Prepare Environment

**PowerShell**:
```powershell
# Set license environment variable
$env:HOOPS_AI_LICENSE = "your-license-key-here"

# Launch Jupyter
jupyter lab
```

### Step 2: Create Task Definition File

**cad_tasks_example.py**:
```python
"""Complete example of CAD processing tasks for part classification"""

import os
import glob
import random
import numpy as np
import hoops_ai
from hoops_ai.flowmanager import flowtask
from hoops_ai.cadaccess import HOOPSLoader, HOOPSTools
from hoops_ai.cadencoder import BrepEncoder
from hoops_ai.storage import DataStorage
from hoops_ai.storage.datasetstorage.schema_builder import SchemaBuilder

# License setup (module level)
license_key = os.getenv("HOOPS_AI_LICENSE")
if license_key:
    hoops_ai.set_license(license_key, validate=False)

# Schema definition (module level)
builder = SchemaBuilder(domain="Part_Classification", version="1.0")

# Faces group
faces_group = builder.create_group("faces", "face", "Face-level geometric data")
faces_group.create_array("face_areas", ["face"], "float32", "Face surface areas")
faces_group.create_array("face_types", ["face"], "int32", "Face type codes")

# Labels group
labels_group = builder.create_group("labels", "part", "Part classification labels")
labels_group.create_array("part_category", ["part"], "int32", "Part category (0-9)")
labels_group.create_array("complexity_score", ["part"], "float32", "Complexity rating")

# Metadata routing
builder.define_file_metadata('processing_time', 'float32', 'Encoding duration in seconds')
builder.define_categorical_metadata('category_name', 'str', 'Human-readable category')

cad_schema = builder.build()

@flowtask.extract(
    name="gather_part_files",
    inputs=["cad_datasources"],
    outputs=["cad_dataset"]
)
def gather_parts(source: str):
    """Gather STEP files from source directory"""
    patterns = ["*.step", "*.stp"]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(f"{source}/**/{pattern}", recursive=True))
    print(f"Found {len(files)} CAD files")
    return files

@flowtask.transform(
    name="encode_part_features",
    inputs=["cad_file", "cad_loader", "storage"],
    outputs=["encoded_path"]
)
def encode_part(cad_file: str, cad_loader: HOOPSLoader, storage: DataStorage):
    """Encode part geometry and assign classification label"""
    import time
    start_time = time.time()
    
    # Load CAD model
    cad_model = cad_loader.create_from_file(cad_file)
    storage.set_schema(cad_schema)
    
    # Prepare B-Rep
    hoops_tools = HOOPSTools()
    hoops_tools.adapt_brep(cad_model, None)
    
    # Extract geometric features
    brep_encoder = BrepEncoder(cad_model.get_brep(), storage)
    brep_encoder.push_face_indices()
    brep_encoder.push_face_attributes()  # → saves face_areas, face_types
    
    # Generate classification label (in real scenarios, this comes from labels)
    file_basename = os.path.basename(cad_file)
    random.seed(hash(file_basename))
    category = random.randint(0, 9)
    complexity = random.uniform(1.0, 10.0)
    
    category_names = ["Bracket", "Shaft", "Housing", "Gear", "Fastener", 
                      "Connector", "Panel", "Bushing", "Bearing", "Gasket"]
    
    # Save classification data
    storage.save_data("labels/part_category", np.array([category], dtype=np.int32))
    storage.save_data("labels/complexity_score", np.array([complexity], dtype=np.float32))
    
    # Save metadata
    processing_time = time.time() - start_time
    storage.save_metadata("processing_time", processing_time)
    storage.save_metadata("category_name", category_names[category])
    storage.save_metadata("Item", cad_file)
    
    # Compress and return
    storage.compress_store()
    return storage.get_file_path("")
```

### Step 3: Create Notebook

**example_pipeline.ipynb**:

```python
# Cell 1: Setup
import pathlib
import hoops_ai
from cad_tasks_example import gather_parts, encode_part, cad_schema

# Cell 2: Configuration
datasources_dir = pathlib.Path("../packages/cadfiles/dataset1")
output_dir = pathlib.Path("./output")

# Cell 3: Create and Run Flow
cad_flow = hoops_ai.create_flow(
    name="part_classification_flow",
    tasks=[gather_parts, encode_part],
    flows_outputdir=str(output_dir),
    max_workers=12,
    ml_task="Part Classification (10 categories)",
    debug=False,
    auto_dataset_export=True
)

print("Starting flow execution...")
flow_output, summary, flow_file = cad_flow.process(
    inputs={'cad_datasources': [str(datasources_dir)]}
)

# Cell 4: Inspect Results
print("\n" + "="*70)
print("FLOW EXECUTION COMPLETED")
print("="*70)
print(f"Files processed: {summary['file_count']}")
print(f"Total time: {summary['Duration [seconds]']['total']:.2f}s")
print(f"Dataset: {summary['flow_data']}")
print(f"Info: {summary['flow_info']}")
print(f"Attributes: {summary['flow_attributes']}")

# Cell 5: Explore Dataset
from hoops_ai.dataset import DatasetExplorer

explorer = DatasetExplorer(flow_output_file=flow_file)
explorer.print_table_of_contents()

# Cell 6: Query Dataset
# Get all parts with category == 3 (Gear)
gear_parts = explorer.get_file_list(
    group="labels",
    where=lambda ds: ds['part_category'] == 3
)
print(f"Found {len(gear_parts)} gear parts")

# Cell 7: Prepare for ML Training
from hoops_ai.dataset import DatasetLoader

loader = DatasetLoader(
    merged_store_path=summary['flow_data'],
    parquet_file_path=summary['flow_info']
)

train_size, val_size, test_size = loader.split(
    key="part_category",
    group="labels",
    train=0.7,
    validation=0.15,
    test=0.15,
    random_state=42
)

print(f"Dataset split: Train={train_size}, Val={val_size}, Test={test_size}")

train_dataset = loader.get_dataset("train")
print(f"Training dataset ready with {len(train_dataset)} samples")
```

### Expected Output

```
Starting flow execution...

######### Flow 'part_classification_flow' start #######

Flow Execution Summary
==================================================
Task 1: gather_part_files
    Inputs : cad_datasources
    Outputs: cad_dataset
Task 2: encode_part_features
    Inputs : cad_file, cad_loader, storage
    Outputs: encoded_path
Task 3: AutoDatasetExportTask
    Inputs : encoded_path
    Outputs: encoded_dataset, encoded_dataset_info, encoded_dataset_attribs

Executing ParallelTask 'gather_part_files' with 250 items.
gather_part_files: 100%|██████████| 250/250 [00:02<00:00, 98.3it/s]

Executing ParallelTask 'encode_part_features' with 250 items.
Using ProcessPoolExecutor with 12 worker processes
encode_part_features: 100%|██████████| 250/250 [03:45<00:00, 1.11it/s, errors=2]

Executing SequentialTask 'AutoDatasetExportTask'.
[Auto] PREP DATA SERVING: Merging 250 encoded files...

Time taken: 235.67 seconds
######### Flow 'part_classification_flow' end ######

======================================================================
FLOW EXECUTION COMPLETED
======================================================================
Files processed: 250
Total time: 235.67s
Dataset: output/flows/part_classification_flow/part_classification_flow.dataset
Info: output/flows/part_classification_flow/part_classification_flow.infoset
Attributes: output/flows/part_classification_flow/part_classification_flow.attribset

========================================
DATASET TABLE OF CONTENTS
========================================

Available Groups:
--------------------------------------------------

Group: faces
  Arrays:
    - face_areas: (125430,) float32
    - face_types: (125430,) int32
    - file_id_code_faces: (125430,) int32

Group: labels
  Arrays:
    - part_category: (250,) int32
    - complexity_score: (250,) float32
    - file_id_code_labels: (250,) int32

Metadata Files:
  - Info: part_classification_flow.infoset (file-level metadata)
  - Attributes: part_classification_flow.attribset (categorical metadata)

Total Files: 250
```

---

## Advanced Topics

### 1. Custom Storage Providers (available for future versions)

By default, the framework uses `LocalStorageProvider` for file-based storage. You can create custom providers:

```python
from hoops_ai.storage import StorageProvider

class DatabaseStorageProvider(StorageProvider):
    """Store encoded data in a database instead of files"""
    
    def __init__(self, connection_string):
        self.conn = connect_to_database(connection_string)
    
    def save_array(self, key, array):
        self.conn.execute(f"INSERT INTO arrays (key, data) VALUES (?, ?)", 
                          (key, array.tobytes()))
    
    # Implement other required methods...
```

**Usage in Task**:
```python
@flowtask.transform(...)
def encode_with_custom_storage(cad_file, cad_loader, storage):
    # storage is now DatabaseStorageProvider instance
    storage.save_array("faces/areas", face_areas)
```

### 2. Dynamic Worker Allocation

Adjust `max_workers` based on system resources:

```python
import os
import psutil

# Use 80% of available CPUs
available_cpus = os.cpu_count()
memory_gb = psutil.virtual_memory().total / (1024**3)

# Rule: 1 worker per 4GB RAM, capped at CPU count
max_workers = min(int(memory_gb / 4), int(available_cpus * 0.8))

cad_flow = hoops_ai.create_flow(
    name="adaptive_flow",
    tasks=[...],
    max_workers=max_workers
)
```

### 3. Custom Error Handling

Override `finalize()` to implement custom error handling:

```python
from hoops_ai.flowmanager import ParallelTask

class CustomEncodingTask(ParallelTask):
    def process_item(self, item):
        # Your encoding logic
        pass
    
    def finalize(self):
        """Custom error handling after all items processed"""
        super().finalize()  # Call base implementation
        
        # Send alerts for critical errors
        critical_errors = [e for e in self.errors if 'MemoryError' in e['error']]
        if critical_errors:
            send_alert(f"Critical errors in {len(critical_errors)} files")
        
        # Generate custom error report
        with open("custom_error_report.html", "w") as f:
            f.write(self.generate_html_error_report())
```

### 4. Multi-Stage Pipelines

Chain multiple flows for complex pipelines:

```python
# Flow 1: Data preparation
prep_flow = hoops_ai.create_flow(
    name="data_preparation",
    tasks=[gather_files, validate_files, fix_corrupted_files],
    flows_outputdir="./output",
    max_workers=8
)
prep_output, prep_summary, prep_file = prep_flow.process(inputs={'cad_datasources': [...]})

# Flow 2: Feature extraction
feature_flow = hoops_ai.create_flow(
    name="feature_extraction",
    tasks=[encode_geometry, encode_topology, encode_materials],
    flows_outputdir="./output",
    max_workers=16
)
# Use output from prep_flow as input
feature_output, feature_summary, feature_file = feature_flow.process(
    inputs={'cad_dataset': prep_output.task_instances[0].results}
)

# Flow 3: ML training
training_flow = hoops_ai.create_flow(
    name="model_training",
    tasks=[prepare_datasets, train_model, evaluate_model],
    flows_outputdir="./output",
    max_workers=1  # Training often uses single process with GPU
)
training_output, training_summary, training_file = training_flow.process(
    inputs={'encoded_dataset': feature_summary['flow_data']}
)
```

### 5. Conditional Task Execution

Use task outputs to decide whether to execute subsequent tasks:

```python
@flowtask.custom(...)
def check_file_count(encoded_files):
    """Only proceed if we have enough files"""
    if len(encoded_files) < 100:
        raise ValueError(f"Insufficient files: {len(encoded_files)} < 100")
    return encoded_files

cad_flow = hoops_ai.create_flow(
    name="conditional_flow",
    tasks=[
        gather_files,
        check_file_count,  # ← Will stop flow if < 100 files
        encode_data,
        train_model
    ],
    flows_outputdir="./output"
)
```

---

## Best Practices

### 1. Task Organization

✅ **DO**: Group related tasks in domain-specific files
```
tasks/
├── cad_ingestion_tasks.py      # File gathering, validation
├── cad_encoding_tasks.py       # Feature extraction
└── ml_preparation_tasks.py     # Dataset splitting, augmentation
```

❌ **DON'T**: Put all tasks in one monolithic file

### 2. Error Handling

✅ **DO**: Let the framework handle errors at the item level
```python
@flowtask.transform(...)
def encode_data(cad_file, cad_loader, storage):
    # Framework catches exceptions automatically
    cad_model = cad_loader.create_from_file(cad_file)  # May raise
    # ... rest of encoding
```

❌ **DON'T**: Catch all exceptions yourself (prevents error aggregation)
```python
@flowtask.transform(...)
def encode_data(cad_file, cad_loader, storage):
    try:
        cad_model = cad_loader.create_from_file(cad_file)
        # ...
    except Exception:
        return None  # ❌ Error not logged properly
```

### 3. Schema Design

✅ **DO**: Design schemas upfront for predictable data organization
```python
# Define schema before task definitions
builder = SchemaBuilder(domain="MyDomain", version="1.0")
faces_group = builder.create_group("faces", "face", "Face data")
faces_group.create_array("areas", ["face"], "float32", "Face areas")
schema = builder.build()
```

❌ **DON'T**: Mix schema-driven and ad-hoc data saving
```python
# Inconsistent: some data uses schema, some doesn't
storage.set_schema(schema)
storage.save_data("faces/areas", areas)  # ✓ Uses schema
storage.save_data("random_key", data)    # ❌ Not in schema (will fail or be ignored)
```

### 4. Parallel Execution

✅ **DO**: Use `debug=False` for production, `debug=True` for development
```python
# Development: Sequential execution for easy debugging
dev_flow = hoops_ai.create_flow(..., debug=True, max_workers=1)

# Production: Parallel execution for performance
prod_flow = hoops_ai.create_flow(..., debug=False, max_workers=24)
```

✅ **DO**: Profile task execution times to optimize `max_workers`
```python
# Check .flow file for task durations
with open(flow_file) as f:
    summary = json.load(f)
    print(summary["Duration [seconds]"])
# Output: {"total": 245.6, "GatherCADFiles": 5.2, "EncodingTask": 235.8, ...}
```

### 6. Testing and Validation

✅ **DO**: Test with a small subset first
```python
# Test with 10 files before processing 10,000
test_flow = hoops_ai.create_flow(
    name="test_run",
    tasks=[gather_files, encode_data],
    flows_outputdir="./test_output",
    max_workers=2,
    debug=True
)
test_output, _, _ = test_flow.process(
    inputs={'cad_datasources': ["/path/to/small/dataset"]}
)
```

✅ **DO**: Validate schema compliance after encoding
```python
# Check that all expected groups are present
explorer = DatasetExplorer(flow_output_file=flow_file)
expected_groups = ["faces", "edges", "labels"]
actual_groups = explorer.available_groups()

assert set(expected_groups).issubset(actual_groups), \
    f"Missing groups: {set(expected_groups) - actual_groups}"
```

### 7. Documentation

✅ **DO**: Document task functions with clear docstrings
```python
@flowtask.transform(...)
def encode_manufacturing_features(cad_file: str, cad_loader: HOOPSLoader, 
                                   storage: DataStorage) -> str:
    """
    Extract manufacturing-specific features from CAD models.
    
    Features extracted:
    - Machining complexity score (1-5)
    - Material type classification
    - Estimated machining time
    
    Args:
        cad_file: Path to CAD file (.step, .stp, .iges)
        cad_loader: HOOPSLoader instance (provided by framework)
        storage: DataStorage with schema set (provided by framework)
    
    Returns:
        Path to compressed .data file containing encoded features
        
    Raises:
        ValueError: If CAD file is invalid or unsupported format
        MemoryError: If model exceeds 2GB memory limit
    """
    # Implementation...
```

### 8. Performance Optimization

✅ **DO**: Use appropriate chunk sizes for dataset merging
```python
# For large datasets (millions of faces)
merger.merge_data(
    face_chunk=1_000_000,      # 1M faces per chunk
    edge_chunk=1_000_000,
    faceface_flat_chunk=5_000_000,
    batch_size=500             # Process 500 files at a time
)

# For small datasets (thousands of faces)
merger.merge_data(
    face_chunk=100_000,        # 100K faces per chunk
    edge_chunk=100_000,
    faceface_flat_chunk=500_000,
    batch_size=50
)
```

✅ **DO**: Balance `max_workers` with I/O constraints
```python
# Rule of thumb for CAD processing:
# - CPU-bound tasks (encoding): max_workers = CPU count
# - I/O-bound tasks (file gathering): max_workers = 2 * CPU count
# - Memory-intensive: max_workers = RAM_GB / 4

import os
cpu_count = os.cpu_count()

gather_flow = hoops_ai.create_flow(
    tasks=[gather_files],
    max_workers=cpu_count * 2  # I/O-bound
)

encode_flow = hoops_ai.create_flow(
    tasks=[encode_data],
    max_workers=cpu_count  # CPU-bound
)
```

---

## Summary

The **HOOPS AI Flow** module is the cornerstone of the framework, providing:

1. **Decorator-Based Task Definition**: Simple `@flowtask` decorators for clean, declarative pipelines
2. **Automatic Parallel Execution**: ProcessPoolExecutor with per-worker HOOPSLoader instances
3. **Robust Error Handling**: Item-level error catching with continued processing
4. **Automatic Dataset Merging**: Schema-driven data organization with `.flow` file generation
5. **Comprehensive Logging**: Detailed execution logs with task-specific contexts
6. **Windows Compatibility**: Proper handling of ProcessPoolExecutor requirements

**Key Workflow**:
```
Define Tasks (cad_tasks.py or any python file) → Create Flow → Execute → Analyze (.flow file) → ML Training
```

**The Flow module transforms CAD processing from this:**
```python
# Manual approach: 100+ lines of boilerplate
files = []
for source in sources:
    files.extend(glob.glob(...))

results = []
errors = []
with ProcessPoolExecutor(max_workers=8) as executor:
    futures = [executor.submit(encode, f) for f in files]
    for future in as_completed(futures):
        try:
            result = future.result()
            results.append(result)
        except Exception as e:
            errors.append(e)
# ... merge data manually ...
# ... route metadata manually ...
# ... generate summary manually ...
```

**To this:**
```python
# Flow approach: 10 lines, fully automated
cad_flow = hoops_ai.create_flow(
    name="my_pipeline",
    tasks=[gather_files, encode_data],
    flows_outputdir="./output",
    max_workers=8
)
flow_output, summary, flow_file = cad_flow.process(
    inputs={'cad_datasources': ["/path/to/cad"]}
)
```

**The HOOPS AI Flow module is production-ready, battle-tested on datasets with 100,000+ CAD files, and designed to scale from rapid prototyping to enterprise deployments.**

---

## See Also

- [DatasetExplorer Documentation](DatasetExplorer_DatasetLoader_Documentation.md): Query and analyze merged datasets
- [SchemaBuilder Documentation](SchemaBuilder_Documentation.md): Define structured data schemas
- [DataStorage Documentation](DataStorage_Documentation.md): Understand the storage layer
- [Module Access and Encoder](Module_Access_and_Encoder.md): Deep dive into CAD loading and encoding
