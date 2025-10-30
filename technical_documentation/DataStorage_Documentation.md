# DataStorage Documentation

## Overview

The **DataStorage** module provides a unified, abstract interface for persisting and retrieving data in HOOPS AI. It supports multiple storage backends (Zarr, JSON, in-memory) while maintaining a consistent API. The system integrates with SchemaBuilder to enable schema-driven validation, metadata routing, and organized data merging.

**Key Architecture**: DataStorage implementations follow a plugin pattern where:
1. The `DataStorage` abstract base class defines the interface
2. Concrete implementations (OptStorage, MemoryStorage, JsonStorageHandler) handle specific formats
3. Schema dictionaries from SchemaBuilder configure storage behavior
4. Metadata routing automatically organizes information into `.infoset` and `.attribset` files

The system follows a push-based storage pattern:
```
Data Producer (Encoder) → DataStorage.save_data() → Backend-Specific Persistence
Schema Dictionary → DataStorage.set_schema() → Validation & Routing Logic
```

---

## Table of Contents

1. [DataStorage Base Class](#datastorage-base-class)
   - [Core Abstract Methods](#core-abstract-methods)
   - [Schema Support Methods](#schema-support-methods)
   - [Validation Methods](#validation-methods)
2. [OptStorage (Zarr-Based)](#optstorage-zarr-based)
   - [Initialization](#optstorage-initialization)
   - [Data Operations](#optstorage-data-operations)
   - [Compression](#optstorage-compression)
   - [Dimension Naming](#dimension-naming-for-xarray)
3. [MemoryStorage (In-Memory)](#memorystorage-in-memory)
   - [Initialization](#memorystorage-initialization)
   - [Use Cases](#memorystorage-use-cases)
4. [JsonStorageHandler (JSON Files)](#jsonstoragehandler-json-files)
   - [Initialization](#jsonstoragehandler-initialization)
   - [Serialization](#json-serialization)
5. [Schema Integration](#schema-integration)
   - [Setting Schemas](#setting-schemas)
   - [Schema-Driven Validation](#schema-driven-validation)
   - [Group Membership](#group-membership)
6. [Metadata Management](#metadata-management)
   - [File-Level vs Categorical](#file-level-vs-categorical)
   - [Nested Metadata Keys](#nested-metadata-keys)
   - [Automatic Size Tracking](#automatic-size-tracking)
7. [Usage Examples](#usage-examples)
8. [Implementation Comparison](#implementation-comparison)

---

## DataStorage Base Class

The `DataStorage` abstract base class defines the interface that all storage implementations must follow.

```python
from hoops_ai.storage.datastorage import DataStorage
```

### Core Abstract Methods

All DataStorage implementations must provide these methods:

#### save_data(data_key: str, data: Any) → None

**Purpose:** Persists data under a unique key.

```python
storage.save_data("face_areas", face_areas_array)
storage.save_data("metadata/description", "CAD part analysis")
```

**Parameters:**
- `data_key` (str): Unique identifier for the data (can be hierarchical using '/')
- `data` (Any): Data to store (numpy arrays, lists, dicts, scalars, strings)

**Behavior:**
- Overwrites existing data if key already exists
- May validate data against schema if schema is set
- Automatically calculates and stores data size in metadata

---

#### load_data(data_key: str) → Any

**Purpose:** Retrieves data associated with a specific key.

```python
face_areas = storage.load_data("face_areas")
description = storage.load_data("metadata/description")
```

**Parameters:**
- `data_key` (str): The key of the data to load

**Returns:**
- `Any`: The loaded data in its original format

**Raises:**
- `KeyError`: If the data_key does not exist

---

#### save_metadata(key: str, value: Any) → None

**Purpose:** Stores metadata as key-value pairs, supporting nested structures.

```python
storage.save_metadata("size_cadfile", 1024000)
storage.save_metadata("file_sizes_KB/face_areas", 45.2)
storage.save_metadata("processing/duration", 12.5)
```

**Parameters:**
- `key` (str): Metadata key (supports nesting with '/' separator)
- `value` (Any): Metadata value (bool, int, float, string, list, or array)

**Behavior:**
- Creates nested dictionary structure based on '/' separators
- Merges with existing metadata (doesn't overwrite entire structure)
- When schema is set, routes to `.infoset` or `.attribset` files

---

#### load_metadata(key: str) → Any

**Purpose:** Loads metadata by key, supporting nested access.

```python
file_size = storage.load_metadata("size_cadfile")
face_size = storage.load_metadata("file_sizes_KB/face_areas")
```

**Parameters:**
- `key` (str): Metadata key (supports nested keys with '/' separator)

**Returns:**
- `Any`: The metadata value

**Raises:**
- `KeyError`: If the key does not exist

---

#### get_keys() → list

**Purpose:** Returns a list of all top-level data keys in storage.

```python
keys = storage.get_keys()
# Returns: ['face_indices', 'face_areas', 'edge_indices', 'graph', ...]
```

**Returns:**
- `list`: All top-level keys (arrays and groups)

---

#### get_file_path(data_key: str) → str

**Purpose:** Gets the file system path for a specific data key.

```python
path = storage.get_file_path("face_areas")
# OptStorage: "./encoded_data/my_part.zarr/face_areas"
# JsonStorage: "./json_data/face_areas.json"
# MemoryStorage: "In-memory storage: No file path for key 'face_areas'"
```

**Parameters:**
- `data_key` (str): The data key

**Returns:**
- `str`: File path or descriptive message for in-memory storage

---

#### close() → None

**Purpose:** Cleanup and resource deallocation.

```python
storage.close()
```

**Behavior:**
- OptStorage: Copies visualization files, exports metadata, deletes temporary directory
- MemoryStorage: Clears all data from memory
- JsonStorage: No-op (JSON operations are stateless)

---

#### format() → str

**Purpose:** Returns the storage format identifier.

```python
fmt = storage.format()
# OptStorage: "zarr"
# MemoryStorage: "memory"
# JsonStorage: "json"
```

**Returns:**
- `str`: Format identifier string

---

#### compress_store() → int

**Purpose:** Compresses the storage (if applicable).

```python
compressed_size = storage.compress_store()
# OptStorage: Creates .data zip file, returns size in bytes
# MemoryStorage/JsonStorage: Returns 0 (no compression)
```

**Returns:**
- `int`: Size of compressed file in bytes, or 0 if not applicable

---

### Schema Support Methods

These methods are provided by the base class with default implementations that can be overridden.

#### set_schema(schema: dict) → None

**Purpose:** Configures the storage with a schema definition from SchemaBuilder.

```python
from hoops_ai.storage.datasetstorage import SchemaBuilder

builder = SchemaBuilder(domain="CAD_analysis")
faces_group = builder.create_group("faces", "face", "Face data")
faces_group.create_array("face_areas", ["face"], "float32")
schema = builder.build()

storage.set_schema(schema)  # ← Schema dictionary applied here
```

**Parameters:**
- `schema` (dict): Schema dictionary from `SchemaBuilder.build()`

**Behavior:**
- Default implementation saves schema as metadata under key `"_storage_schema"`
- Subclasses can override for more efficient schema storage
- Enables validation and metadata routing

---

#### get_schema() → dict

**Purpose:** Retrieves the currently configured schema.

```python
schema = storage.get_schema()
# Returns: Schema dictionary or {} if no schema is set
```

**Returns:**
- `dict`: The schema definition, or empty dict if no schema

---

#### get_group_for_array(array_name: str) → str

**Purpose:** Determines which group an array belongs to based on schema.

```python
group = storage.get_group_for_array("face_areas")
# Returns: "faces" (based on schema definition)

group = storage.get_group_for_array("edge_lengths")
# Returns: "edges"
```

**Parameters:**
- `array_name` (str): Name of the array

**Returns:**
- `str`: Group name for the array, or None if not found in schema

**Use Case:** Dataset merging uses this to group arrays correctly

---

### Validation Methods

#### validate_data_against_schema(data_key: str, data: Any) → bool

**Purpose:** Validates data against the stored schema if present.

```python
import numpy as np

# Assuming schema defines face_areas as ["face"] dimension, float32
valid_data = np.array([1.5, 2.3, 4.1], dtype=np.float32)
is_valid = storage.validate_data_against_schema("face_areas", valid_data)
# Returns: True

invalid_data = np.array([[1.5, 2.3], [4.1, 3.2]])  # Wrong dimensions
is_valid = storage.validate_data_against_schema("face_areas", invalid_data)
# Returns: False
```

**Parameters:**
- `data_key` (str): The key under which data will be stored
- `data` (Any): The data to validate

**Returns:**
- `bool`: True if valid or no schema present, False if validation fails

**Validation Checks:**
- Dimension count matches schema specification
- Data type matches or is convertible to specified dtype
- Arrays not in schema are allowed (extensible schema)

---

## OptStorage (Zarr-Based)

`OptStorage` is the primary storage implementation using Zarr format for efficient, chunked, compressed array storage.

### OptStorage Initialization

```python
from hoops_ai.storage import OptStorage

storage = OptStorage(
    store_path="./flow_output/flows/my_flow/encoded/part_001.zarr",
    compress_extension=".data"
)
```

**Parameters:**
- `store_path` (str): Path to the Zarr directory store
- `compress_extension` (str): Extension for compressed archive (default: ".data")

**Initialization Behavior:**
- If `.zarr.data` file exists and directory doesn't: Opens in read-only mode
- Otherwise: Creates directory structure and initializes writable store
- Creates `metadata.json` file for metadata storage
- Uses `DirectoryStore` for writing, `ZipStore` for reading compressed archives

---

### OptStorage Data Operations

#### Saving Data

OptStorage recursively handles nested data structures:

```python
import numpy as np

# Scalars
storage.save_data("num_faces", 42)

# 1D Arrays
storage.save_data("face_areas", np.array([1.5, 2.3, 4.1], dtype=np.float32))

# Multi-dimensional Arrays
storage.save_data("face_normals", np.random.randn(100, 3).astype(np.float32))

# Nested Dictionaries
storage.save_data("graph", {
    "edges_source": np.array([0, 1, 2]),
    "edges_destination": np.array([1, 2, 3]),
    "num_nodes": 4
})

# Strings
storage.save_data("description", "High-complexity CAD part")
```

**Data Type Handling:**
- **NumPy arrays**: Stored with compression, chunking, and dimension names
- **Lists**: Converted to NumPy arrays
- **Dicts**: Become Zarr groups with nested structure
- **Scalars** (int, float, bool): Stored as 0-dimensional arrays
- **Strings**: Stored as object arrays with MsgPack codec

**Automatic Features:**
- **NaN Detection**: Raises error if NaNs found in floating-point arrays
- **Compression**: Zstd level 12 compression applied
- **Chunking**: Automatic chunk sizing (~1M elements per chunk)
- **Filters**: Delta filter for integer arrays
- **Size Tracking**: Data size automatically recorded in metadata

---

#### Loading Data

```python
# Load arrays
face_areas = storage.load_data("face_areas")
# Returns: numpy array

# Load nested structures
graph = storage.load_data("graph")
# Returns: {'edges_source': array([...]), 'edges_destination': array([...]), 'num_nodes': 4}

# Load scalars
num_faces = storage.load_data("num_faces")
# Returns: 42

# Load strings
description = storage.load_data("description")
# Returns: "High-complexity CAD part"
```

---

### OptStorage Compression

OptStorage supports compression into a single `.data` file:

```python
# After all data is saved
compressed_size = storage.compress_store()
# Returns: Size of compressed .data file in bytes

# Result: Creates part_001.zarr.data (ZipStore format)
# Original directory remains until close() is called
```

**Compression Process:**
1. Validates no NaNs exist in arrays (safety check)
2. Copies all data from DirectoryStore to ZipStore
3. Preserves all array attributes (including dimension names)
4. Includes metadata.json in the archive
5. Returns compressed file size

**Benefits:**
- Single-file distribution
- Reduced disk space (Zstd compression)
- Atomic operations (write-then-rename pattern)
- Read-only access to prevent accidental modification

---

### Dimension Naming for xarray

OptStorage sets the `_ARRAY_DIMENSIONS` attribute on all arrays to enable xarray compatibility:

```python
# When saving "face_areas" array
# OptStorage automatically sets:
# arr.attrs["_ARRAY_DIMENSIONS"] = ["face_areas_dim_0"]

# For nested data "faceface/a3_distance"
# Dimensions become: ["faceface_a3_distance_dim_0", "faceface_a3_distance_dim_1", ...]
```

**Why This Matters:**
- Enables direct loading with `xarray.open_zarr()`
- Preserves dimension semantics across save/load cycles
- Supports multi-dimensional indexing and slicing
- Facilitates interoperability with other Zarr tools

---

### OptStorage Cleanup Behavior

```python
storage.close()
```

**Close Operations:**
1. **Copy visualization files** (`visu*`) to `stream_cache/` directory
2. **Export metadata** to `files_summary/{filename}.json`
3. **Delete temporary directory** (if compression was performed)
4. **Thread-safe**: Handles concurrent close() calls gracefully

---

## MemoryStorage (In-Memory)

`MemoryStorage` stores all data in RAM using Python dictionaries, ideal for testing and small datasets.

### MemoryStorage Initialization

```python
from hoops_ai.storage.datastorage import MemoryStorage

storage = MemoryStorage()
```

**No Parameters:** Creates empty in-memory storage

---

### MemoryStorage Use Cases

**Ideal For:**
- **Unit Testing**: Fast, isolated, no disk I/O
- **Prototyping**: Quick iteration without file cleanup
- **Small Datasets**: When data fits comfortably in RAM
- **Temporary Caching**: Short-lived data that doesn't need persistence

**Not Suitable For:**
- Large datasets (memory limits)
- Long-running processes (lost on crash)
- Shared data across processes (not serialized)

---

### MemoryStorage Data Operations

```python
import numpy as np

storage = MemoryStorage()

# Save data (stored in internal dict)
storage.save_data("face_areas", np.array([1.5, 2.3, 4.1]))

# Load data (retrieved from dict)
face_areas = storage.load_data("face_areas")

# Metadata (separate internal dict)
storage.save_metadata("size_cadfile", 1024000)
size = storage.load_metadata("size_cadfile")

# Get keys
keys = storage.get_keys()  # ['face_areas']

# Close (clears all data)
storage.close()
```

**Features:**
- **Instant operations**: No disk I/O overhead
- **Size tracking**: Approximates memory usage with `sys.getsizeof()`
- **Nested metadata**: Supports hierarchical keys like OptStorage
- **No compression**: `compress_store()` returns 0

---

## JsonStorageHandler (JSON Files)

`JsonStorageHandler` stores each data key as a separate JSON file, suitable for human-readable storage.

### JsonStorageHandler Initialization

```python
from hoops_ai.storage.datastorage import JsonStorageHandler

storage = JsonStorageHandler(json_dir_path="./json_output")
```

**Parameters:**
- `json_dir_path` (str): Directory where JSON files will be stored

**Initialization:**
- Creates directory if it doesn't exist
- Creates `metadata.json` for metadata storage
- Each data key becomes a separate `.json` file

---

### JSON Serialization

JsonStorageHandler handles NumPy types automatically:

```python
import numpy as np

storage = JsonStorageHandler("./json_data")

# NumPy arrays → JSON lists
storage.save_data("face_areas", np.array([1.5, 2.3, 4.1], dtype=np.float32))
# Saved as: face_areas.json → [1.5, 2.3, 4.1]

# Complex numbers → JSON objects
storage.save_data("complex_data", np.array([1+2j, 3+4j]))
# Saved as: [{"real": 1.0, "imag": 2.0, "_numpy_complex": true}, ...]

# Dictionaries → JSON objects
storage.save_data("metadata", {"version": "1.0", "author": "HOOPS AI"})

# Load (automatic deserialization)
face_areas = storage.load_data("face_areas")
# Returns: numpy array([1.5, 2.3, 4.1], dtype=float32)
```

**Serialization Rules:**
- **NumPy arrays** → JSON lists (with auto-conversion on load)
- **NumPy scalars** → Python primitives
- **Complex numbers** → JSON objects with `{"real", "imag", "_numpy_complex"}`
- **Nested structures** → Recursive serialization

**File Naming:**
- Keys are sanitized: `"face/areas"` → `face_areas.json`
- Only alphanumeric, `-`, and `_` allowed in filenames

---

## Schema Integration

The DataStorage base class provides schema integration that works across all implementations.

### Setting Schemas

```python
from hoops_ai.storage import OptStorage
from hoops_ai.storage.datasetstorage import SchemaBuilder

# Build schema
builder = SchemaBuilder(domain="CAD_analysis", version="1.0")
faces_group = builder.create_group("faces", "face", "Face data")
faces_group.create_array("face_areas", ["face"], "float32")
faces_group.create_array("face_normals", ["face", "coordinate"], "float32")

edges_group = builder.create_group("edges", "edge", "Edge data")
edges_group.create_array("edge_lengths", ["edge"], "float32")

schema = builder.build()

# Apply to storage
storage = OptStorage(store_path="./data.zarr")
storage.set_schema(schema)
```

**What set_schema() Does:**
1. Stores schema dictionary as metadata (key: `"_storage_schema"`)
2. Enables `validate_data_against_schema()` checks
3. Enables `get_group_for_array()` lookups
4. Configures metadata routing (if schema includes routing rules)

---

### Schema-Driven Validation

When schema is set, DataStorage can validate data before saving:

```python
import numpy as np

# Schema defines face_areas as ["face"] dimension, float32
schema = {...}  # From SchemaBuilder
storage.set_schema(schema)

# Valid data
valid_data = np.array([1.5, 2.3, 4.1], dtype=np.float32)
is_valid = storage.validate_data_against_schema("face_areas", valid_data)
# Returns: True

# Invalid: wrong dimensions (2D instead of 1D)
invalid_data = np.array([[1.5, 2.3], [4.1, 3.2]])
is_valid = storage.validate_data_against_schema("face_areas", invalid_data)
# Returns: False

# Invalid: wrong dtype (int instead of float)
invalid_data = np.array([1, 2, 3], dtype=np.int32)
is_valid = storage.validate_data_against_schema("face_areas", invalid_data)
# Returns: False (dtype mismatch)
```

**Validation Logic:**
1. Extracts array specification from schema
2. Checks number of dimensions matches
3. Checks dtype matches or is convertible
4. Returns True if no schema or array not in schema (extensible)

---

### Group Membership

Schema enables storage to determine group membership for arrays:

```python
schema = {...}  # Schema with "faces" and "edges" groups
storage.set_schema(schema)

# Lookup group for array
group = storage.get_group_for_array("face_areas")
# Returns: "faces"

group = storage.get_group_for_array("edge_lengths")
# Returns: "edges"

group = storage.get_group_for_array("unknown_array")
# Returns: None (not in schema)
```

**Use Case: Dataset Merging**

During merging, the merger uses `get_group_for_array()` to:
1. Group arrays from multiple files by their logical group
2. Concatenate arrays along the correct dimension
3. Apply special processing (e.g., matrix flattening for "faceface" group)

---

## Metadata Management

### File-Level vs Categorical

DataStorage distinguishes between two types of metadata:

**File-Level Metadata** (`.infoset` files):
- Information about each individual data file
- Examples: file size, processing time, file path, timestamps
- One row per file in merged datasets
- Routing patterns: `"size_*"`, `"duration_*"`, `"processing_*"`, `"flow_name"`

**Categorical Metadata** (`.attribset` files):
- Classification and labeling information
- Examples: part category, material type, complexity rating
- Used for grouping and filtering datasets
- Routing patterns: `"*_label"`, `"category"`, `"type"`, `"material_*"`

**Routing Configuration:**

Schemas can define routing rules:
```python
builder.set_metadata_routing_rules(
    file_level_patterns=["size_*", "duration_*", "processing_*", "flow_name"],
    categorical_patterns=["*_label", "category", "type"],
    default_numeric="file_level",
    default_categorical="categorical"
)
```

When schema is set, `save_metadata()` automatically routes based on:
1. Explicit definitions in schema
2. Pattern matching
3. Default rules based on data type

---

### Nested Metadata Keys

All DataStorage implementations support nested metadata using '/' as separator:

```python
# Top-level metadata
storage.save_metadata("size_cadfile", 1024000)

# Nested metadata
storage.save_metadata("file_sizes_KB/face_areas", 45.2)
storage.save_metadata("file_sizes_KB/edge_lengths", 12.3)
storage.save_metadata("processing/duration", 12.5)
storage.save_metadata("processing/timestamp", "2025-10-30T10:30:00")

# Load nested metadata
face_size = storage.load_metadata("file_sizes_KB/face_areas")
# Returns: 45.2

duration = storage.load_metadata("processing/duration")
# Returns: 12.5

# Load entire nested section
file_sizes = storage.load_metadata("file_sizes_KB")
# Returns: {'face_areas': 45.2, 'edge_lengths': 12.3}
```

**Metadata Structure:**
```json
{
  "size_cadfile": 1024000,
  "file_sizes_KB": {
    "face_areas": 45.2,
    "edge_lengths": 12.3
  },
  "processing": {
    "duration": 12.5,
    "timestamp": "2025-10-30T10:30:00"
  }
}
```

---

### Automatic Size Tracking

All DataStorage implementations automatically track data sizes:

```python
storage.save_data("face_areas", large_array)
# Automatically stores size in metadata["file_sizes_KB"]["face_areas"]

# Retrieve size
size_kb = storage.load_metadata("file_sizes_KB/face_areas")
# Returns: Size in kilobytes
```

**Size Calculation:**
- **OptStorage**: Actual disk usage (sum of Zarr chunk files)
- **MemoryStorage**: Approximate memory usage (`sys.getsizeof()`)
- **JsonStorageHandler**: File size of `.json` file

---

## Usage Examples

### Complete CAD Encoding Workflow

```python
from hoops_ai.storage import OptStorage
from hoops_ai.storage.datasetstorage import SchemaBuilder
from hoops_ai.cadaccess import HOOPSLoader
from hoops_ai.cadencoder import BrepEncoder
import time

# 1. Build schema
builder = SchemaBuilder(domain="CAD_analysis", version="1.0")

faces_group = builder.create_group("faces", "face", "Face data")
faces_group.create_array("face_indices", ["face"], "int32")
faces_group.create_array("face_areas", ["face"], "float32")
faces_group.create_array("face_types", ["face"], "int32")

edges_group = builder.create_group("edges", "edge", "Edge data")
edges_group.create_array("edge_indices", ["edge"], "int32")
edges_group.create_array("edge_lengths", ["edge"], "float32")

builder.define_file_metadata("size_cadfile", "int64", "File size")
builder.define_file_metadata("processing_time", "float32", "Processing time")
builder.define_categorical_metadata("file_label", "int32", "Classification")

builder.set_metadata_routing_rules(
    file_level_patterns=["size_*", "processing_*"],
    categorical_patterns=["*_label"]
)

schema = builder.build()

# 2. Initialize storage with schema
storage = OptStorage(store_path="./encoded/part_001.zarr")
storage.set_schema(schema)

# 3. Encode CAD file
loader = HOOPSLoader()
model = loader.create_from_file("part_001.step")
brep = model.get_brep()

encoder = BrepEncoder(brep_access=brep, storage_handler=storage)

start_time = time.time()

# Push geometric features (validated by schema)
encoder.push_face_indices()
encoder.push_face_attributes()
encoder.push_edge_indices()
encoder.push_edge_attributes()

processing_time = time.time() - start_time

# 4. Save metadata (automatically routed)
import os
storage.save_metadata("size_cadfile", os.path.getsize("part_001.step"))  # → .infoset
storage.save_metadata("processing_time", processing_time)                 # → .infoset
storage.save_metadata("file_label", 2)                                    # → .attribset

# 5. Compress and close
compressed_size = storage.compress_store()
print(f"Compressed to {compressed_size / 1024:.2f} KB")
storage.close()
```

---

### Testing with MemoryStorage

```python
from hoops_ai.storage.datastorage import MemoryStorage
import numpy as np

def test_encoder():
    # Use MemoryStorage for fast testing
    storage = MemoryStorage()
    
    # Mock encoder operations
    storage.save_data("face_indices", np.array([0, 1, 2, 3]))
    storage.save_data("face_areas", np.array([1.5, 2.3, 4.1, 3.7]))
    storage.save_metadata("num_faces", 4)
    
    # Assertions
    assert len(storage.load_data("face_indices")) == 4
    assert storage.load_metadata("num_faces") == 4
    
    # Fast cleanup (no disk operations)
    storage.close()

test_encoder()
```

---

### JSON Export for Visualization

```python
from hoops_ai.storage.datastorage import JsonStorageHandler
import numpy as np

# Store as human-readable JSON for external tools
storage = JsonStorageHandler("./json_export")

storage.save_data("face_areas", np.array([1.5, 2.3, 4.1], dtype=np.float32))
storage.save_data("metadata", {
    "part_name": "Bracket_V2",
    "complexity": "Medium",
    "num_features": 42
})

storage.save_metadata("export_timestamp", "2025-10-30T10:30:00")

# Result: 
# ./json_export/face_areas.json        → [1.5, 2.3, 4.1]
# ./json_export/metadata.json           → {"part_name": "Bracket_V2", ...}
# ./json_export/metadata.json (meta)    → {"export_timestamp": "..."}
```

---

### Schema Validation in Production

```python
from hoops_ai.storage import OptStorage
from hoops_ai.storage.datasetstorage import SchemaBuilder
import numpy as np

# Build strict schema
builder = SchemaBuilder(domain="production_data")
group = builder.create_group("measurements", "sample")
group.create_array("temperature", ["sample"], "float32")
group.create_array("pressure", ["sample"], "float32")

schema = builder.build()

storage = OptStorage("./transformed_data.data")
storage.set_schema(schema)

# Valid data passes
valid_temps = np.array([20.5, 25.3, 22.1], dtype=np.float32)
if storage.validate_data_against_schema("temperature", valid_temps):
    storage.save_data("temperature", valid_temps)
    print("✓ Data validated and saved")

# Invalid data is caught
invalid_temps = np.array([[20.5, 25.3], [22.1, 23.4]])  # Wrong dimensions
if not storage.validate_data_against_schema("temperature", invalid_temps):
    print("✗ Validation failed: wrong dimensions")
    # Handle error appropriately
```

---

## Implementation Comparison

| Feature | OptStorage (Zarr) | MemoryStorage | JsonStorageHandler |
|---------|------------------|---------------|-------------------|
| **Persistence** | Disk (Zarr format) | RAM (volatile) | Disk (JSON files) |
| **Compression** | Yes (Zstd) | No | No |
| **Size Limit** | Disk capacity | RAM capacity | Disk capacity |
| **Speed** | Fast (chunked) | Fastest (in-memory) | Slow (JSON parsing) |
| **Human-Readable** | No | N/A | Yes |
| **Multi-File Output** | Single directory | N/A | One file per key |
| **xarray Support** | Yes (dimension names) | No | No |
| **Chunking** | Automatic | N/A | N/A |
| **Compression Ratio** | ~10:1 typical | N/A | Minimal |
| **Concurrent Access** | Read-only after compress | No | Read-only |
| **Best For** | Production, large data | Testing, small data | Export, inspection |
| **Schema Support** | Full | Full | Full |
| **Metadata Files** | metadata.json | Internal dict | metadata.json |
| **NaN Detection** | Yes (automatic) | No | No |

---

## Summary

The **DataStorage** system provides HOOPS AI with:

- **Unified Interface**: Consistent API across Zarr, JSON, and in-memory storage
- **Schema Integration**: Validates data and routes metadata using SchemaBuilder dictionaries
- **Flexible Backends**: Choose storage based on use case (production, testing, export)
- **Automatic Features**: Size tracking, compression, dimension naming, NaN detection
- **Metadata Organization**: Separates file-level (`.infoset`) and categorical (`.attribset`) metadata

**Integration with SchemaBuilder:**
```
SchemaBuilder.build() → Schema Dictionary
                            ↓
                DataStorage.set_schema(schema)
                            ↓
        ┌──────────────────┴──────────────────┐
        ↓                                       ↓
save_data() with validation          save_metadata() with routing
        ↓                                       ↓
Group-organized storage              .infoset / .attribset files
```

The schema dictionary serves as the **configuration contract** between data producers and storage, ensuring:
- Consistent data organization
- Validated data types and dimensions
- Predictable metadata routing
- Schema-guided dataset merging

Choose the appropriate DataStorage implementation based on your workflow:
- **OptStorage**: Production pipelines, large-scale data, compression needed
- **MemoryStorage**: Unit tests, prototyping, temporary data
- **JsonStorageHandler**: Data export, human inspection, external tool integration
