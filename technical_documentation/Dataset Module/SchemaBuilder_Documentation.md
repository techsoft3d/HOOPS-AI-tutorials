# SchemaBuilder Documentation

## Overview

The **SchemaBuilder** module provides a user-friendly, explicit API for defining data storage schemas in HOOPS AI. Schemas define how data should be organized into logical groups and arrays, enabling predictable data merging, validation, and metadata routing. The SchemaBuilder creates Python dictionaries that serve as configuration blueprints for DataStorage implementations.

**Key Concept**: The SchemaBuilder produces a **schema dictionary** that tells DataStorage implementations:
1. How to organize arrays into logical groups
2. What dimensions each array should have
3. How to validate incoming data
4. Where to route metadata (file-level vs. categorical)

The system follows a declarative pattern:
```
SchemaBuilder → Schema Dictionary → DataStorage.set_schema() → Validated Storage Operations
```

---

## Table of Contents

1. [SchemaBuilder Overview](#schemabuilder-overview)
   - [Initialization](#initialization)
   - [Core Concepts](#core-concepts)
2. [Group Class](#group-class)
   - [Creating Arrays](#creating-arrays)
   - [Managing Arrays](#managing-arrays)
3. [SchemaBuilder Methods](#schemabuilder-methods)
   - [Group Management](#group-management)
   - [Metadata Definition](#metadata-definition)
   - [Metadata Routing](#metadata-routing)
4. [Schema Templates](#schema-templates)
   - [Predefined Templates](#predefined-templates)
   - [Template Extension](#template-extension)
5. [Schema Dictionary Structure](#schema-dictionary-structure)
6. [Integration with DataStorage](#integration-with-datastorage)
7. [Usage Examples](#usage-examples)

---

## SchemaBuilder Overview

The `SchemaBuilder` class provides a standard, object-oriented API for creating data storage schemas without requiring method chaining.

### Initialization

```python
from hoops_ai.storage.datasetstorage import SchemaBuilder

builder = SchemaBuilder(
    domain="CAD_analysis",
    version="1.0",
    description="Schema for CAD geometric feature extraction"
)
```

**Parameters:**
- `domain` (str): Domain name for this schema (e.g., 'CAD_analysis', 'manufacturing_data')
- `version` (str): Schema version for compatibility tracking (default: '1.0')
- `description` (str, optional): Human-readable description of the schema's purpose

### Core Concepts

#### 1. **Groups**
Groups are logical containers that organize related arrays. Each group has:
- **Name**: Unique identifier for the group (e.g., 'faces', 'edges')
- **Primary Dimension**: The main indexing dimension (e.g., 'face', 'edge', 'batch')
- **Description**: What data this group contains
- **Special Processing**: Optional processing hint (e.g., 'matrix_flattening', 'nested_edges')

#### 2. **Arrays**
Arrays are the actual data containers within groups. Each array specifies:
- **Name**: Unique identifier within the group
- **Dimensions**: List of dimension names defining the array's shape
- **Dtype**: Data type (e.g., 'float32', 'int32', 'float64')
- **Description**: What this array represents
- **Validation Rules**: Optional constraints (min_value, max_value, etc.)

#### 3. **Metadata**
Metadata is divided into two categories:
- **File-level Metadata**: Stored in `.infoset` files (e.g., file size, processing time, file path)
- **Categorical Metadata**: Stored in `.attribset` files (e.g., labels, categories, complexity ratings)

---

## Group Class

### Creating a Group

```python
faces_group = builder.create_group(
    name="faces",
    primary_dimension="face",
    description="Face geometric data",
    special_processing=None  # Optional
)
```

**Returns:** `Group` object that can be used to define arrays

### Creating Arrays

#### Basic Array Definition

```python
faces_group.create_array(
    name="face_areas",
    dimensions=["face"],
    dtype="float32",
    description="Surface area of each face"
)
```

**Parameters:**
- `name` (str): Unique array name within the group
- `dimensions` (List[str]): Ordered list of dimension names
- `dtype` (str): Data type - 'float32', 'float64', 'int32', 'int64', 'bool', 'str'
- `description` (str, optional): Human-readable description
- `**validation_rules`: Optional validation constraints

#### Multi-Dimensional Arrays

```python
# 2D array: face normals (N_faces × 3 coordinates)
faces_group.create_array(
    name="face_normals",
    dimensions=["face", "coordinate"],
    dtype="float32",
    description="Normal vectors for each face (N x 3)"
)

# 4D array: UV grid samples (N_faces × U × V × components)
faces_group.create_array(
    name="face_uv_grids",
    dimensions=["face", "uv_x", "uv_y", "component"],
    dtype="float32",
    description="Sampled points on face surfaces"
)
```

#### Arrays with Validation Rules

```python
faces_group.create_array(
    name="face_areas",
    dimensions=["face"],
    dtype="float32",
    description="Surface area of each face",
    min_value=0.0,  # Validation: areas must be positive
    max_value=1e6   # Validation: reasonable upper bound
)
```

### Managing Arrays

#### Remove an Array

```python
success = faces_group.remove_array("face_areas")
# Returns: True if removed, False if not found
```

#### Get Array Specification

```python
array_spec = faces_group.get_array("face_areas")
# Returns: {'dims': ['face'], 'dtype': 'float32', 'description': '...'}
```

#### List All Arrays in Group

```python
array_names = faces_group.list_arrays()
# Returns: ['face_areas', 'face_types', 'face_normals', ...]
```

---

## SchemaBuilder Methods

### Group Management

#### Create a Group

```python
edges_group = builder.create_group(
    name="edges",
    primary_dimension="edge",
    description="Edge-related geometric properties",
    special_processing=None
)
```

#### Get an Existing Group

```python
faces_group = builder.get_group("faces")
# Returns: Group object or None if not found
```

#### Remove a Group

```python
success = builder.remove_group("edges")
# Returns: True if removed, False if not found
```

#### List All Groups

```python
group_names = builder.list_groups()
# Returns: ['faces', 'edges', 'graph', 'metadata']
```

---

### Metadata Definition

Metadata definitions tell DataStorage where to route metadata and how to validate it.

#### Define File-Level Metadata

File-level metadata is stored in `.infoset` Parquet files and represents information about each data file.

```python
builder.define_file_metadata(
    name="size_cadfile",
    dtype="int64",
    description="File size in bytes",
    required=False,
    min_value=0
)

builder.define_file_metadata(
    name="processing_time",
    dtype="float32",
    description="Processing time in seconds",
    required=False
)

builder.define_file_metadata(
    name="flow_name",
    dtype="str",
    description="Name of the flow that processed this file",
    required=True
)
```

**Parameters:**
- `name` (str): Metadata field name
- `dtype` (str): Data type ('str', 'int32', 'int64', 'float32', 'float64', 'bool')
- `description` (str, optional): Field description
- `required` (bool): Whether this field must be present (default: False)
- `**validation_rules`: Additional constraints (min_value, max_value, etc.)

#### Define Categorical Metadata

Categorical metadata is stored in `.attribset` Parquet files and represents categorical classifications.

```python
builder.define_categorical_metadata(
    name="machining_category",
    dtype="int32",
    description="Machining complexity classification",
    values=[1, 2, 3, 4, 5],
    labels=["Simple", "Easy", "Medium", "Hard", "Complex"],
    required=False
)

builder.define_categorical_metadata(
    name="material_type",
    dtype="str",
    description="Material classification",
    values=["steel", "aluminum", "plastic", "composite"],
    required=True
)
```

**Parameters:**
- `name` (str): Metadata field name
- `dtype` (str): Data type
- `description` (str, optional): Field description
- `values` (List, optional): List of allowed values
- `labels` (List[str], optional): Human-readable labels for values
- `required` (bool): Whether this field must be present (default: False)
- `**validation_rules`: Additional constraints

---

### Metadata Routing

The SchemaBuilder provides flexible metadata routing using pattern matching and defaults.

#### Set Routing Rules

```python
builder.set_metadata_routing_rules(
    file_level_patterns=[
        "description",
        "flow_name",
        "stream *",      # Wildcard: matches 'stream .scs', 'stream .prc', etc.
        "Item",
        "size_*",        # Wildcard: matches 'size_cadfile', 'size_compressed', etc.
        "duration_*",    # Wildcard: matches all duration fields
        "processing_*"
    ],
    categorical_patterns=[
        "category",
        "type",
        "*_label",       # Wildcard: matches 'file_label', 'part_label', etc.
        "material_*",
        "complexity"
    ],
    default_numeric="file_level",      # Where numeric metadata goes by default
    default_categorical="categorical", # Where categorical metadata goes by default
    default_string="categorical"       # Where string metadata goes by default
)
```

**Pattern Matching:**
- `*` wildcard matches any characters
- Patterns are case-insensitive
- Explicit definitions override pattern matching

#### Query Metadata Routing

```python
# Get routing destination for a specific field
destination = builder.get_metadata_routing("file_label")
# Returns: "categorical" or "file_level"

# List all metadata fields by category
fields = builder.list_metadata_fields()
# Returns: {'file_level': ['size_cadfile', 'processing_time', ...],
#           'categorical': ['file_label', 'material_type', ...]}
```

#### Validate Metadata

```python
# Validate a specific field's value
is_valid = builder.validate_metadata_field("machining_category", 3)
# Returns: True (3 is in allowed values [1,2,3,4,5])

is_valid = builder.validate_metadata_field("machining_category", 10)
# Returns: False (10 not in allowed values)

# Validate entire schema
errors = builder.validate_metadata_schema()
# Returns: List of error messages, empty if valid
```

---

## Schema Templates

The SchemaBuilder supports predefined templates for common use cases, reducing boilerplate code.

### Predefined Templates

#### Load from Template

```python
# Start with a complete CAD analysis template
builder = SchemaBuilder().from_template('cad_basic')

# Or use convenience functions
from hoops_ai.storage.datasetstorage import create_cad_schema
builder = create_cad_schema()
```

**Available Templates:**

1. **`cad_basic`**: Basic CAD analysis with faces, edges, and graph data
   - Groups: faces, edges, graph, metadata
   - Arrays: face_areas, face_indices, edge_lengths, etc.

2. **`cad_advanced`**: Advanced CAD with surface properties and relationships
   - Groups: faces, edges, faceface, graph, performance
   - Arrays: face_uv_grids, edge_dihedral_angles, extended_adjacency, etc.

3. **`manufacturing_basic`**: Manufacturing data with quality metrics
   - Groups: production, sensors, materials
   - Arrays: quality_score, temperature, pressure, composition, etc.

4. **`sensor_basic`**: Sensor data with timestamps and readings
   - Groups: timeseries, sensors, events
   - Arrays: timestamp, value, sensor_type, event_type, etc.

#### List Available Templates

```python
from hoops_ai.storage.datasetstorage.schema_templates import SchemaTemplates

templates = SchemaTemplates.list_templates()
# Returns: ['cad_basic', 'cad_advanced', 'manufacturing_basic', 'sensor_basic']

description = SchemaTemplates.get_template_description('cad_advanced')
# Returns: "Advanced CAD analysis including surface properties and relationships"
```

### Template Extension

Templates can be extended to add custom groups and arrays:

```python
# Start with CAD basic template and add custom data
builder = SchemaBuilder().extend_template('cad_basic')

# Add custom group for ML predictions
predictions_group = builder.create_group(
    "predictions",
    "face",
    "ML model predictions for faces"
)
predictions_group.create_array("predicted_class", ["face"], "int32")
predictions_group.create_array("confidence_score", ["face"], "float32")

# Add custom metadata
builder.define_categorical_metadata(
    "model_version",
    "str",
    "ML model version used for predictions"
)

schema = builder.build()
```

---

## Schema Dictionary Structure

The `build()` method produces a Python dictionary with the following structure:

```python
schema = builder.build()

# Schema structure:
{
    "domain": "CAD_analysis",
    "version": "1.0",
    "description": "Schema for CAD geometric feature extraction",
    "groups": {
        "faces": {
            "primary_dimension": "face",
            "description": "Face geometric data",
            "arrays": {
                "face_areas": {
                    "dims": ["face"],
                    "dtype": "float32",
                    "description": "Surface area of each face"
                },
                "face_normals": {
                    "dims": ["face", "coordinate"],
                    "dtype": "float32",
                    "description": "Normal vectors for each face (N x 3)"
                },
                # ... more arrays
            }
        },
        "edges": {
            "primary_dimension": "edge",
            # ... edge arrays
        },
        # ... more groups
    },
    "metadata": {
        "file_level": {
            "size_cadfile": {
                "dtype": "int64",
                "description": "File size in bytes",
                "required": False
            },
            # ... more file-level metadata
        },
        "categorical": {
            "file_label": {
                "dtype": "int32",
                "description": "Classification label",
                "values": [0, 1, 2, 3, 4],
                "required": False
            },
            # ... more categorical metadata
        },
        "routing_rules": {
            "file_level_patterns": ["description", "flow_name", "size_*"],
            "categorical_patterns": ["*_label", "category"],
            "default_numeric": "file_level",
            "default_categorical": "categorical",
            "default_string": "categorical"
        }
    }
}
```

### Exporting Schemas

```python
# Export to JSON string
json_string = builder.to_json(indent=2)

# Save to file
builder.save_to_file("my_schema.json")

# Load from file
loaded_builder = SchemaBuilder.load_from_file("my_schema.json")
```

---

## Integration with DataStorage

The schema dictionary produced by SchemaBuilder is consumed by DataStorage implementations via the `set_schema()` method.

### Schema Flow Diagram

```
┌─────────────────┐
│  SchemaBuilder  │
│   .build()      │
└────────┬────────┘
         │
         ▼
   Schema Dictionary
   (Python dict)
         │
         ▼
┌─────────────────────┐
│   DataStorage       │
│   .set_schema(dict) │
└────────┬────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  Storage Operations with Validation     │
│  • save_data() → validates against dims │
│  • save_metadata() → routes correctly   │
│  • get_group_for_array() → uses schema  │
└─────────────────────────────────────────┘
```

### Setting Schema on Storage

```python
from hoops_ai.storage import OptStorage
from hoops_ai.storage.datasetstorage import SchemaBuilder

# Build schema
builder = SchemaBuilder(domain="CAD_analysis", version="1.0")
faces_group = builder.create_group("faces", "face", "Face data")
faces_group.create_array("face_areas", ["face"], "float32")
schema = builder.build()

# Apply schema to storage
storage = OptStorage(store_path="./encoded_data/my_part.data")
storage.set_schema(schema)  # ← Schema dictionary passed here

# Now storage knows:
# 1. "face_areas" belongs to "faces" group
# 2. It should have dimensions ["face"]
# 3. It should be float32 type
```

### Schema-Driven Operations

Once a schema is set, DataStorage can:

**1. Validate Data Dimensions**
```python
import numpy as np

# This will be validated against schema
face_areas = np.array([1.5, 2.3, 4.1, 3.7], dtype=np.float32)
storage.save_data("face_areas", face_areas)
# ✓ Validates: correct dtype, 1D array as expected
```

**2. Route Metadata Correctly**
```python
# Metadata routing based on schema rules
storage.save_metadata("size_cadfile", 1024000)     # → .infoset (file-level)
storage.save_metadata("file_label", 3)              # → .attribset (categorical)
storage.save_metadata("flow_name", "my_flow")       # → .infoset (file-level pattern match)
```

**3. Determine Group Membership**
```python
group_name = storage.get_group_for_array("face_areas")
# Returns: "faces"

group_name = storage.get_group_for_array("edge_lengths")
# Returns: "edges"
```

**4. Enable Schema-Aware Merging**

During dataset merging, the schema guides:
- Which arrays belong to the same group
- What dimensions to concatenate along
- How to handle special processing (e.g., matrix flattening)

---

## Usage Examples

### Complete Workflow Example

```python
from hoops_ai.storage import OptStorage
from hoops_ai.storage.datasetstorage import SchemaBuilder
from hoops_ai.cadaccess import HOOPSLoader
from hoops_ai.cadencoder import BrepEncoder

# 1. Define schema for CAD encoding
builder = SchemaBuilder(domain="CAD_analysis", version="1.0")

# Define faces group
faces_group = builder.create_group("faces", "face", "Face geometric data")
faces_group.create_array("face_indices", ["face"], "int32", "Face IDs")
faces_group.create_array("face_areas", ["face"], "float32", "Face surface areas")
faces_group.create_array("face_types", ["face"], "int32", "Surface type classification")
faces_group.create_array("face_uv_grids", ["face", "uv_x", "uv_y", "component"], 
                        "float32", "UV-sampled points and normals")

# Define edges group
edges_group = builder.create_group("edges", "edge", "Edge geometric data")
edges_group.create_array("edge_indices", ["edge"], "int32", "Edge IDs")
edges_group.create_array("edge_lengths", ["edge"], "float32", "Edge lengths")
edges_group.create_array("edge_types", ["edge"], "int32", "Curve type classification")

# Define graph group
graph_group = builder.create_group("graph", "graphitem", "Topology graph")
graph_group.create_array("edges_source", ["edge"], "int32", "Source face indices")
graph_group.create_array("edges_destination", ["edge"], "int32", "Dest face indices")
graph_group.create_array("num_nodes", ["graphitem"], "int32", "Number of nodes")

# Define metadata
builder.define_file_metadata("size_cadfile", "int64", "CAD file size in bytes")
builder.define_file_metadata("processing_time", "float32", "Encoding time in seconds")
builder.define_categorical_metadata("file_label", "int32", "Part classification label")

# Set routing rules
builder.set_metadata_routing_rules(
    file_level_patterns=["size_*", "processing_*", "duration_*"],
    categorical_patterns=["*_label", "category", "type"]
)

# Build schema
schema = builder.build()

# 2. Apply schema to storage
storage = OptStorage(store_path="./encoded/part_001.zarr")
storage.set_schema(schema)

# 3. Encode CAD data with schema-validated storage
loader = HOOPSLoader()
model = loader.create_from_file("part_001.step")
brep = model.get_brep()

encoder = BrepEncoder(brep_access=brep, storage_handler=storage)

# These operations are now schema-validated
encoder.push_face_indices()        # → "faces" group
encoder.push_face_attributes()     # → "faces" group
encoder.push_facegrid(ugrid=5, vgrid=5)  # → "faces" group

encoder.push_edge_indices()        # → "edges" group
encoder.push_edge_attributes()     # → "edges" group

encoder.push_face_adjacency_graph()  # → "graph" group

# Metadata is automatically routed
import os
import time
start_time = time.time()
# ... encoding happens ...
storage.save_metadata("size_cadfile", os.path.getsize("part_001.step"))  # → .infoset
storage.save_metadata("processing_time", time.time() - start_time)       # → .infoset
storage.save_metadata("file_label", 2)                                    # → .attribset

storage.close()
```

### Using Templates for Quick Setup

```python
from hoops_ai.storage.datasetstorage import create_cad_schema
from hoops_ai.storage import OptStorage

# Quick setup with template
builder = create_cad_schema()  # Loads 'cad_basic' template

# Customize as needed
predictions = builder.create_group("predictions", "face", "ML predictions")
predictions.create_array("predicted_label", ["face"], "int32")
predictions.create_array("confidence", ["face"], "float32")

schema = builder.build()

# Apply to storage
storage = OptStorage(store_path="./output/part.data")
storage.set_schema(schema)
```

### Schema Validation Example

```python
import numpy as np

# Create schema with validation rules
builder = SchemaBuilder(domain="validated_data")
group = builder.create_group("measurements", "sample")
group.create_array("temperature", ["sample"], "float32", 
                  min_value=-273.15,  # Absolute zero
                  max_value=5000.0)   # Reasonable max

schema = builder.build()
storage = OptStorage("./data.zarr")
storage.set_schema(schema)

# Valid data
valid_temps = np.array([20.5, 25.3, 22.1], dtype=np.float32)
storage.save_data("temperature", valid_temps)  # ✓ Success

# Invalid data (contains value below min)
invalid_temps = np.array([20.5, -300.0, 22.1], dtype=np.float32)
try:
    storage.save_data("temperature", invalid_temps)  # ✗ Validation fails
except ValueError as e:
    print(f"Validation error: {e}")
```

---

## Performance Considerations

### Schema Impact on Performance

**Minimal Runtime Overhead:**
- Schema validation is optional (controlled by DataStorage implementation)
- Schema lookup is dictionary-based (O(1) operations)
- Schema is set once per storage instance

**Benefits for Large-Scale Data:**
- **Predictable Merging**: Schema-guided dataset merging is deterministic
- **Type Safety**: Prevents type mismatches that cause downstream errors
- **Memory Efficiency**: Dimension information enables efficient chunk sizing
- **Parallelization**: Schema enables safe parallel writes to different groups

### Best Practices

1. **Define Schema Early**: Set schema before any data operations
2. **Use Templates**: Start with templates for common patterns
3. **Validate Once**: Schema validation during development, disable in production
4. **Document Dimensions**: Clear dimension names improve code readability
5. **Version Schemas**: Increment version when making breaking changes

---

## Summary

The **SchemaBuilder** is HOOPS AI's declarative interface for defining data organization:

- **Creates schema dictionaries** that configure DataStorage behavior
- **Defines logical groups** to organize related arrays
- **Specifies array dimensions** for validation and merging
- **Routes metadata** to appropriate storage locations (.infoset vs .attribset)
- **Provides templates** for common use cases (CAD, manufacturing, sensors)
- **Enables validation** to catch data issues early

The schema dictionary serves as the **contract** between data producers (encoders) and data consumers (storage, merging, ML pipelines), ensuring consistent, validated, and well-organized data throughout the HOOPS AI system.
