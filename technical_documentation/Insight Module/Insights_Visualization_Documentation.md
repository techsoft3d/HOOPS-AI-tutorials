# HOOPS AI Insights: Visualization Documentation

## Overview

The **Insights** module provides powerful visualization tools for exploring CAD datasets and viewing individual CAD models directly from Python. It bridges the gap between data analysis and visual understanding, enabling you to:

- **Preview datasets** with image grids showing multiple CAD files at once
- **Interactive 3D viewing** of CAD models in Jupyter notebooks
- **Filter and visualize** query results from DatasetExplorer
- **Color-coded predictions** for ML model outputs on 3D geometry

The module consists of three main public components:
1. **DatasetViewer**: Batch visualization of dataset query results
2. **CADViewer**: Interactive 3D viewing of individual CAD files
3. **Utils (ColorPalette)**: Tools for generating color schemes and grouping predictions

---

## Table of Contents

1. [DatasetViewer](#datasetviewer)
   - [Initialization and Setup](#initialization-and-setup)
   - [Image Grid Visualization](#image-grid-visualization)
   - [Interactive 3D Visualization](#interactive-3d-visualization)
   - [Integration with DatasetExplorer](#integration-with-datasetexplorer)
2. [CADViewer](#cadviewer)
   - [Basic Usage](#basic-usage)
   - [Face Coloring and Interaction](#face-coloring-and-interaction)
   - [Quick View Function](#quick-view-function)
3. [Visualization Utils](#visualization-utils)
   - [ColorPalette](#colorpalette)
   - [Grouping Predictions](#grouping-predictions)
4. [Complete Workflow Examples](#complete-workflow-examples)
5. [Best Practices](#best-practices)

---

## DatasetViewer

### Purpose

`DatasetViewer` enables visualization of multiple CAD files from dataset queries. It's designed to work seamlessly with `DatasetExplorer`, allowing you to:
- Filter files based on criteria (labels, complexity, features)
- Visualize the filtered results as image grids or interactive 3D views
- Compare multiple parts side-by-side

### Initialization and Setup

#### Method 1: From DatasetExplorer (Recommended)

```python
from hoops_ai.dataset import DatasetExplorer
from hoops_ai.insights import DatasetViewer

# Initialize explorer
explorer = DatasetExplorer(flow_output_file="fabwave.flow")

# Create viewer from explorer (automatically extracts visualization paths)
viewer = DatasetViewer.from_explorer(explorer)

# Check availability
viewer.print_statistics()
```

**Output:**
```
==================================================
Dataset Visualization Statistics
==================================================
Total files:              234
Files with PNG preview:   234 (100.0%)
Files with 3D cache:      234 (100.0%)
Overall coverage:         100.0%
==================================================
```

#### Method 2: Manual Initialization

```python
from hoops_ai.insights import DatasetViewer

# Extract data from explorer manually
cache_df = explorer.get_stream_cache_paths()

file_ids = cache_df['id'].astype(int).tolist()
png_paths = cache_df['stream_cache_png'].tolist()
scs_paths = cache_df['stream_cache_3d'].tolist()
file_names = cache_df['name'].tolist()

# Create viewer with explicit data
viewer = DatasetViewer(file_ids, png_paths, scs_paths, file_names)
```

---

### Image Grid Visualization

#### Basic Grid Display

```python
# Get all file IDs
all_ids = viewer.get_available_file_ids()

# Display first 25 files as image grid
fig = viewer.show_preview_as_image(all_ids, k=25)
```

**Result:** Creates a 5×5 grid showing PNG previews of 25 CAD files.

#### Customized Grid Layout

```python
# Custom 4-column grid with file names
fig = viewer.show_preview_as_image(
    all_ids,
    k=20,
    grid_cols=4,
    figsize=(16, 10),
    label_format='name',
    title='High Complexity Parts',
    show_labels=True
)
```

**Parameters:**
- `file_ids` (List[int]): File IDs to visualize
- `k` (int): Maximum number of files to display (default: 25)
- `grid_cols` (int, optional): Columns in grid. If None, auto-calculated
- `figsize` (tuple, optional): Figure size (width, height). If None, auto-calculated
- `show_labels` (bool): Show file labels on images (default: True)
- `label_format` (str): 'id', 'name', or 'both' (default: 'id')
- `title` (str, optional): Overall figure title
- `missing_color` (tuple): RGB color for missing previews (default: gray)
- `save_path` (str, optional): Save figure to this path

#### Label Format Options

```python
# Show only file IDs
viewer.show_preview_as_image(file_ids, label_format='id')
# Labels: "ID: 42", "ID: 87", ...

# Show only file names
viewer.show_preview_as_image(file_ids, label_format='name')
# Labels: "bracket.step", "housing.step", ...

# Show both ID and name
viewer.show_preview_as_image(file_ids, label_format='both')
# Labels: "ID:42\nbracket.step"
```

#### Save Visualization

```python
# Create and save grid
fig = viewer.show_preview_as_image(
    file_ids,
    k=100,
    save_path='results/dataset_preview.png'
)
```

---

### Interactive 3D Visualization

#### Basic 3D Viewing

```python
# Open 3 interactive 3D viewers (inline in notebook)
viewers_3d = viewer.show_preview_as_3d(file_ids, k=3)

# Each viewer is a CADViewer instance
print(f"Created {len(viewers_3d)} 3D viewers")
```

**Result:** Opens 3 inline 3D viewers in the notebook, allowing rotation, zoom, and interaction with each model.

#### Customized 3D Display

```python
# Larger inline viewers
viewers_3d = viewer.show_preview_as_3d(
    file_ids,
    k=5,
    display_mode='inline',
    width=600,
    height=500
)

# Sidecar layout (opens in side panel) [AVAILABLE IN FUTURE RELEASES]
viewers_3d = viewer.show_preview_as_3d(
    file_ids,
    k=5,
    display_mode='sidecar'
)
```

**Parameters:**
- `file_ids` (List[int]): File IDs to visualize
- `k` (int): Maximum number of 3D viewers (default: 5, be conservative!)
- `display_mode` (str): 'inline', 'sidecar', or 'none' (default: 'inline')
- `layout` (str): 'sequential' or 'grid' (default: 'sequential')
- `host` (str): Server host (default: '127.0.0.1')
- `start_port` (int): Starting port for servers (default: 8000)
- `silent` (bool): Suppress server output (default: True)
- `width` (int): Inline viewer width in pixels (default: 400)
- `height` (int): Inline viewer height in pixels (default: 400)

**Returns:** List of `CADViewer` instances (one per displayed file)

#### Interacting with 3D Viewers

```python
# Get selected faces from user interaction
selected_faces = viewers_3d[0].get_selected_faces()
print(f"Selected faces: {selected_faces}")

# Color selected faces red
viewers_3d[0].set_face_color(selected_faces, [255, 0, 0])

# Clear all face colors
viewers_3d[0].clear_face_colors()

# Clean up when done
for v in viewers_3d:
    v.terminate()
```

#### Side-by-Side Comparison (AVAILABLE IN FUTURE RELEASES)

```python
# Compare two specific files
viewer_a, viewer_b = viewer.create_comparison_view(
    file_id_a=42,
    file_id_b=87,
    display_mode='sidecar'
)

# Highlight same features in both
viewer_a.set_face_color([1, 2, 3], [255, 0, 0])
viewer_b.set_face_color([1, 2, 3], [255, 0, 0])
```

---

### Integration with DatasetExplorer

The real power of `DatasetViewer` comes from combining it with `DatasetExplorer` filtering capabilities.

#### Workflow: Filter → Visualize

```python
from hoops_ai.dataset import DatasetExplorer
from hoops_ai.insights import DatasetViewer

# Step 1: Initialize explorer and viewer
explorer = DatasetExplorer(flow_output_file="fabwave.flow")
viewer = DatasetViewer.from_explorer(explorer)

# Step 2: Define filter condition
high_complexity = lambda ds: ds['num_nodes'] > 30

# Step 3: Get file IDs matching condition
complex_file_ids = explorer.get_file_list(
    group="graph",
    where=high_complexity
)

print(f"Found {len(complex_file_ids)} complex files")

# Step 4: Visualize filtered results
fig = viewer.show_preview_as_image(
    complex_file_ids,
    k=25,
    title='High Complexity Parts (>30 faces)',
    grid_cols=5
)
```

#### Example: Filter by Label and Visualize

```python
# Filter files with specific label
pipe_fittings = lambda ds: ds['file_label'] == 15

pipe_file_ids = explorer.get_file_list(
    group="file",
    where=pipe_fittings
)

print(f"Found {len(pipe_file_ids)} pipe fittings")

# Show image grid
viewer.show_preview_as_image(
    pipe_file_ids,
    k=16,
    title='Pipe Fittings (Label 15)',
    label_format='name'
)

# Show 3D views of first 4
viewers_3d = viewer.show_preview_as_3d(pipe_file_ids, k=4)
```

#### Example: Multi-Criteria Filtering

```python
# Complex query: high face count AND specific label
def complex_brackets(ds):
    return (ds['num_nodes'] > 25) & (ds['file_label'] == 3)

bracket_ids = explorer.get_file_list(group="graph", where=complex_brackets)

# Visualize results
viewer.show_preview_as_image(
    bracket_ids,
    k=20,
    title=f'Complex Brackets ({len(bracket_ids)} files)',
    grid_cols=4,
    save_path='results/complex_brackets.png'
)
```
---

### Utility Methods

#### Get File Information

```python
# Get info for specific file
file_info = viewer.get_file_info(42)
print(f"Name: {file_info['name']}")
print(f"PNG available: {file_info['png_path'] is not None}")
print(f"3D available: {file_info['stream_cache_path'] is not None}")
```

#### Get Available Files

```python
# Get all file IDs with visualization data
available_ids = viewer.get_available_file_ids()
print(f"Total files with visualization: {len(available_ids)}")
```

#### Statistics

```python
# Get detailed statistics
stats = viewer.get_statistics()
print(f"Total files: {stats['total_files']}")
print(f"PNG coverage: {stats['png_percentage']:.1f}%")
print(f"3D coverage: {stats['3d_percentage']:.1f}%")

# Pretty print
viewer.print_statistics()
```

---

## CADViewer

### Purpose

`CADViewer` provides interactive 3D visualization of individual CAD files with face coloring, selection, and manipulation capabilities. It's ideal for:
- Detailed inspection of single CAD models
- Visualizing ML predictions on geometry
- Interactive feature highlighting
- Educational demonstrations

### Basic Usage

#### Simple Visualization

```python
from hoops_ai.insights import CADViewer

# Create viewer (auto-finds free port)
viewer = CADViewer()

# Load CAD file
viewer.load_cad_file("bracket.step")

# Display in notebook
viewer.show()
```

#### Quick View (One-Liner)

```python
from hoops_ai.insights import quick_view

# Load and display in one call
viewer = quick_view("bracket.step")
```

#### Display Modes

```python
# Inline display (embedded in notebook)
viewer = CADViewer(display_mode='inline')
viewer.load_cad_file("bracket.step")
viewer.show(width=600, height=500)

# Sidecar display (side panel)
viewer = CADViewer(display_mode='sidecar')
viewer.load_cad_file("bracket.step")
viewer.show()

# No display (server only)
viewer = CADViewer(display_mode='none')
viewer.load_cad_file("bracket.step")
print(f"Viewer URL: {viewer.get_viewer_url()}")
```

#### Port Management

```python
# Auto-find free port (default, recommended)
viewer = CADViewer()  # Finds port 8000-8099

# Use specific port (strict mode)
viewer = CADViewer(port=9000)  # Must be available or fails
```

#### Context Manager (Automatic Cleanup)

```python
# Automatic resource cleanup
with CADViewer() as viewer:
    viewer.load_cad_file("model.step")
    viewer.show()
    # ... interact with viewer ...
# Automatically terminates on exit
```

---

### Face Coloring and Interaction

#### Select and Color Faces

```python
# User: Click faces in 3D viewer (Ctrl+Click for multiple)
# Get selected face indices
selected = viewer.get_selected_faces()
print(f"Selected {len(selected)} faces: {selected}")

# Color selected faces
viewer.set_face_color(selected, [255, 0, 0])  # Red
viewer.set_face_color(selected, [0, 255, 0])  # Green
viewer.set_face_color(selected, [0, 0, 255])  # Blue

# Default highlight color
viewer.set_face_color(selected)  # Light blue
```

#### Color Specific Faces

```python
# Color faces by index
hole_faces = [1, 2, 5, 7]
viewer.set_face_color(hole_faces, [255, 100, 0])
```

#### Clear Colors

```python
# Remove all face coloring
viewer.clear_face_colors()
```

#### Color Groups with Visual Feedback

```python
# Define feature groups
groups = [
    ([1, 2, 6], (255, 0, 0), 'through hole'),
    ([3, 5], (0, 0, 255), 'blind hole'),
    ([8, 9, 10, 11], (0, 255, 0), 'pocket')
]

# Color with progress feedback
viewer.color_faces_by_groups(
    groups,
    delay=0.5,          # Pause between groups
    clear_first=True,   # Clear existing colors
    verbose=True        # Show colored terminal output
)
```

**Terminal Output:**
```
🟥 through hole (3 faces)
🟦 blind hole (2 faces)
🟩 pocket (4 faces)
```

---

### Loading Different File Types

#### CAD Files (Auto-Convert to SCS)

```python
# Automatically converts STEP/IGES to SCS format
viewer.load_cad_file("model.step", auto_convert=True)
viewer.load_cad_file("model.iges", auto_convert=True)
```

#### SCS Files (Direct Loading)

```python
# Load pre-converted SCS file directly (faster)
viewer.load_scs_file("model.scs")
```

#### Background Options

```python
# White background (default, good for presentations)
viewer.load_cad_file("model.step", white_background=True)

# Black background (optional)
viewer.load_cad_file("model.step", white_background=False)
```

---

### Advanced Features

#### Get Viewer Status

```python
status = viewer.get_status()
print(f"Active: {status['active']}")
print(f"Model loaded: {status['model_loaded']}")
print(f"Viewer URL: {status['viewer_url']}")
print(f"Port: {status['port']}")
```

#### Validate Colors

```python
from hoops_ai.insights import CADViewer

# Check if color is valid RGB
CADViewer.validate_color([255, 0, 0])   # True
CADViewer.validate_color([255, 0, 256]) # False (out of range)
CADViewer.validate_color([255, 0])      # False (wrong length)
```

#### Manual Cleanup

```python
# Terminate viewer and release resources
viewer.terminate()

# Check if still active
print(f"Active: {viewer.is_active}")  # False
```

---

### Quick View Function

Convenience function for one-line visualization:

```python
from hoops_ai.insights import quick_view

# Basic usage (auto-finds port)
viewer = quick_view("model.step")

# Inline with custom size
viewer = quick_view("model.step", display_mode='inline')

# Sidecar display
viewer = quick_view("model.step", display_mode='sidecar')

# Specific port (strict mode)
viewer = quick_view("model.step", port=9000)
```

---

## Visualization Utils

### ColorPalette

`ColorPalette` manages label-to-color mappings for classification tasks.

#### Create from Labels

```python
from hoops_ai.insights.utils import ColorPalette

# Define label descriptions
labels = {
    0: "background",
    1: "through hole",
    2: "blind hole",
    3: "pocket",
    4: "slot"
}

# Create palette with automatic colors
palette = ColorPalette.from_labels(
    labels,
    cmap_name='hsv',  # Matplotlib colormap
    reserved_colors={
        0: (200, 200, 200),  # Gray for background
        1: (255, 0, 0)        # Red for through holes
    }
)
```

#### Access Colors and Descriptions

```python
# Get color for label
color = palette.get_color(1)  # (255, 0, 0)

# Get description
desc = palette.get_description(1)  # "through hole"
# Or use alias
label = palette.get_label(1)  # "through hole"

# Get all mappings
all_colors = palette.get_all_colors()
# {0: (200, 200, 200), 1: (255, 0, 0), ...}

all_descs = palette.get_all_descriptions()
# {0: "background", 1: "through hole", ...}
```

#### Palette Operations

```python
# Check membership
1 in palette  # True

# Get size
len(palette)  # 5

# Iterate
for label_id in palette:
    color = palette.get_color(label_id)
    desc = palette.get_description(label_id)
    print(f"Label {label_id}: {desc} = {color}")

# Iterate with items
for label_id, (color, desc) in palette.items():
    print(f"{label_id}: {desc} -> {color}")
```

---

### Grouping Predictions

Use `group_predictions_by_label()` to prepare predictions for visualization:

```python
from hoops_ai.insights.utils import group_predictions_by_label
import numpy as np

# Predictions array (one per face)
predictions = np.array([0, 1, 1, 2, 0, 2, 1, 3, 3])
# Face 0: background, Face 1-2,6: hole, Face 3,5: blind hole, etc.

# Group by label with colors
groups = group_predictions_by_label(
    predictions,
    palette,
    exclude_labels={0}  # Skip background
)

# Result format: [(face_indices, color, description), ...]
# [
#     ([1, 2, 6], (255, 0, 0), 'through hole'),
#     ([3, 5], (0, 0, 255), 'blind hole'),
#     ([7, 8], (0, 255, 0), 'pocket')
# ]

# Use directly with CADViewer
viewer.color_faces_by_groups(groups, verbose=True)
```

---

## Complete Workflow Examples

### Example 1: Dataset Exploration with Visualization

```python
from hoops_ai.dataset import DatasetExplorer
from hoops_ai.insights import DatasetViewer

# Initialize
explorer = DatasetExplorer(flow_output_file="fabwave.flow")
viewer = DatasetViewer.from_explorer(explorer)

# Print statistics
viewer.print_statistics()

# Get all files
all_ids = viewer.get_available_file_ids()

# Visualize random sample
import random
sample_ids = random.sample(all_ids, 25)
viewer.show_preview_as_image(sample_ids, title='Random Sample')

# Filter by complexity
complex_parts = lambda ds: ds['num_nodes'] > 40
complex_ids = explorer.get_file_list(group="graph", where=complex_parts)

# Visualize complex parts
viewer.show_preview_as_image(
    complex_ids,
    k=16,
    title=f'High Complexity Parts ({len(complex_ids)} files)',
    save_path='results/complex_parts.png'
)

# Interactive 3D view of first 3
viewers_3d = viewer.show_preview_as_3d(complex_ids, k=3, width=500, height=400)

# Cleanup
for v in viewers_3d:
    v.terminate()
explorer.close()
```

---

### Example 2: Label-Based Filtering and Visualization

```python
from hoops_ai.dataset import DatasetExplorer
from hoops_ai.insights import DatasetViewer

# Setup
explorer = DatasetExplorer(flow_output_file="fabwave.flow")
viewer = DatasetViewer.from_explorer(explorer)

# Get label descriptions
label_df = explorer.get_descriptions("file_label")
print(label_df)

# Filter by specific label
pipe_fittings = lambda ds: ds['file_label'] == 15
pipe_ids = explorer.get_file_list(group="file", where=pipe_fittings)

print(f"\nFound {len(pipe_ids)} pipe fittings")

# Create visualization
fig = viewer.show_preview_as_image(
    pipe_ids,
    k=25,
    grid_cols=5,
    title='Pipe Fittings (Label 15)',
    label_format='name',
    figsize=(15, 8)
)

# Save high-resolution version
fig.savefig('results/pipe_fittings_overview.png', dpi=300, bbox_inches='tight')

# Cleanup
explorer.close()
```

---

### Example 3: ML Predictions Visualization

```python
from hoops_ai.insights import CADViewer
from hoops_ai.insights.utils import ColorPalette, group_predictions_by_label
import numpy as np

# Load model predictions (example)
predictions = np.load('predictions.npy')  # Shape: (n_faces,)

# Define label palette
labels = {
    0: "no feature",
    17: "through hole",
    18: "blind hole",
    23: "pocket",
    24: "slot"
}

palette = ColorPalette.from_labels(
    labels,
    cmap_name='Set3',
    reserved_colors={
        0: (220, 220, 220),  # Light gray for no feature
        17: (255, 0, 0),      # Red for through holes
        18: (255, 165, 0)     # Orange for blind holes
    }
)

# Group predictions by label
groups = group_predictions_by_label(
    predictions,
    palette,
    exclude_labels={0}
)

# Visualize on 3D model
viewer = CADViewer()
viewer.load_cad_file("test_part.step")
viewer.show(display_mode='sidecar')

# Color faces by prediction
viewer.color_faces_by_groups(groups, delay=0.3, verbose=True)

# Get statistics
print("\nPrediction Distribution:")
for indices, color, desc in groups:
    print(f"  {desc}: {len(indices)} faces")

# Cleanup
viewer.terminate()
```

---

### Example 4: Side-by-Side Comparison

```python
from hoops_ai.insights import DatasetViewer

# Setup viewer
viewer = DatasetViewer.from_explorer(explorer)

# Compare original vs optimized design
viewer_original, viewer_optimized = viewer.create_comparison_view(
    file_id_a=100,  # Original design
    file_id_b=150,  # Optimized design
    display_mode='sidecar'
)

# Highlight same features in both
critical_faces = [5, 7, 12, 18]

viewer_original.set_face_color(critical_faces, [255, 0, 0])
viewer_optimized.set_face_color(critical_faces, [255, 0, 0])

# User can interact with both viewers simultaneously
# Compare geometry, analyze changes, etc.

# Cleanup
viewer_original.terminate()
viewer_optimized.terminate()
```

---

### Example 5: Batch Processing with Visualization

```python
from hoops_ai.dataset import DatasetExplorer
from hoops_ai.insights import DatasetViewer
import matplotlib.pyplot as plt

# Initialize
explorer = DatasetExplorer(flow_output_file="dataset.flow")
viewer = DatasetViewer.from_explorer(explorer)

# Get distribution of face counts
dist = explorer.create_distribution(
    key="num_nodes",
    bins=10,
    group="graph"
)

# Visualize distribution
bin_centers = 0.5 * (dist['bin_edges'][1:] + dist['bin_edges'][:-1])
plt.figure(figsize=(10, 5))
plt.bar(bin_centers, dist['hist'], width=(dist['bin_edges'][1] - dist['bin_edges'][0]))
plt.xlabel('Number of Faces')
plt.ylabel('Count')
plt.title('Face Count Distribution')
plt.savefig('results/face_count_distribution.png', dpi=300)
plt.show()

# Visualize samples from each bin
for i, bin_files in enumerate(dist['file_id_codes_in_bins']):
    if len(bin_files) > 0:
        # Take up to 9 samples from this bin
        sample_ids = bin_files[:9]
        
        # Create visualization
        fig = viewer.show_preview_as_image(
            sample_ids,
            k=9,
            grid_cols=3,
            title=f'Bin {i+1}: {int(dist["bin_edges"][i])}-{int(dist["bin_edges"][i+1])} faces',
            figsize=(9, 9)
        )
        
        # Save
        fig.savefig(f'results/bin_{i+1}_samples.png', dpi=150)
        plt.close(fig)

print("Batch visualization complete!")
explorer.close()
```

---

## Best Practices

### Performance Tips

1. **Limit 3D Viewers**: Opening many 3D viewers consumes resources
   ```python
   # Good: Limit to 3-5 viewers
   viewers = viewer.show_preview_as_3d(file_ids, k=3)
   
   # Avoid: Too many simultaneous 3D viewers
   viewers = viewer.show_preview_as_3d(file_ids, k=50)  # May crash!
   ```

2. **Use Image Grids for Overview**: Fast and memory-efficient
   ```python
   # Efficiently preview 100 files
   viewer.show_preview_as_image(file_ids, k=100)
   ```

3. **Clean Up Resources**: Always terminate 3D viewers when done
   ```python
   # Manual cleanup
   for v in viewers_3d:
       v.terminate()
   
   # Or use context manager
   with CADViewer() as viewer:
       # ... use viewer ...
       pass  # Auto-cleanup
   ```

4. **Filter Before Visualizing**: Reduce data before visualization
   ```python
   # Filter first
   filtered_ids = viewer.filter_by_availability(
       all_ids,
       require_png=True
   )
   
   # Then visualize
   viewer.show_preview_as_image(filtered_ids, k=25)
   ```

---

### Integration Patterns

#### Pattern 1: Explore → Filter → Visualize

```python
# 1. Explore dataset
explorer = DatasetExplorer(flow_output_file="dataset.flow")
explorer.print_table_of_contents()

# 2. Filter files
interesting_files = explorer.get_file_list(
    group="graph",
    where=lambda ds: ds['num_nodes'] > 30
)

# 3. Visualize
viewer = DatasetViewer.from_explorer(explorer)
viewer.show_preview_as_image(interesting_files, k=25)
```

#### Pattern 2: Analyze → Sample → Inspect

```python
# 1. Analyze distribution
dist = explorer.create_distribution(key="num_nodes", bins=5)

# 2. Sample from specific bin
high_complexity_bin = dist['file_id_codes_in_bins'][-1]  # Last bin
sample = high_complexity_bin[:10]

# 3. Inspect in 3D
viewers = viewer.show_preview_as_3d(sample, k=3)
```

#### Pattern 3: Predict → Visualize → Validate

```python
# 1. Run predictions (from ML model)
predictions = model.predict(test_data)

# 2. Group by prediction
groups = group_predictions_by_label(predictions, palette)

# 3. Visualize on geometry
cad_viewer = CADViewer()
cad_viewer.load_cad_file("test_part.step")
cad_viewer.show()
cad_viewer.color_faces_by_groups(groups)

# 4. Validate visually and correct if needed
```

---

### Color Scheme Guidelines

1. **Use Reserved Colors for Important Labels**
   ```python
   palette = ColorPalette.from_labels(
       labels,
       reserved_colors={
           0: (200, 200, 200),  # Gray for background/no-label
           1: (255, 0, 0)        # Red for critical features
       }
   )
   ```

2. **Choose Appropriate Colormaps**
   - **Discrete labels**: 'tab20', 'Set3', 'Paired'
   - **Sequential data**: 'viridis', 'plasma', 'cividis'
   - **Diverging data**: 'RdBu', 'coolwarm'
   - **Many labels**: 'hsv' (but can be hard to distinguish)

3. **Exclude Background from Visualization**
   ```python
   groups = group_predictions_by_label(
       predictions,
       palette,
       exclude_labels={0}  # Don't color background
   )
   ```

---

### Troubleshooting

#### Issue: Port Already in Use

```python
# Problem: Specified port is busy
viewer = CADViewer(port=8000)  # Error if port 8000 is busy

# Solution 1: Use auto port selection (recommended)
viewer = CADViewer()  # Auto-finds free port

# Solution 2: Find and kill process using port
# Windows PowerShell:
# netstat -ano | findstr :8000
# taskkill /F /PID <pid>
```

#### Issue: 3D Viewer Not Displaying

```python
# Check if hoops-viewer is installed
from hoops_ai.insights.hoops_viewer_interface import is_viewer_available

if not is_viewer_available():
    print("Install hoops-viewer: pip install hoops-viewer")
```

#### Issue: Missing PNG/SCS Files

```python
# Check availability
stats = viewer.get_statistics()
print(f"PNG coverage: {stats['png_percentage']:.1f}%")
print(f"3D coverage: {stats['3d_percentage']:.1f}%")

# Filter to available files only
available = viewer.filter_by_availability(
    all_ids,
    require_png=True,
    require_3d=True
)
```

#### Issue: Image Grid Not Displaying

```python
# Ensure matplotlib backend is configured
import matplotlib
matplotlib.use('inline')  # For Jupyter notebooks

import matplotlib.pyplot as plt
plt.ion()  # Interactive mode

# Then create visualization
fig = viewer.show_preview_as_image(file_ids)
plt.show()  # Explicitly show if needed
```

---

## Summary

The **Insights** module provides a complete visualization solution for CAD datasets:

### DatasetViewer
- ✅ Batch visualization of query results
- ✅ Image grids for quick overview
- ✅ Interactive 3D for detailed inspection
- ✅ Seamless DatasetExplorer integration
- ✅ Side-by-side comparison

### CADViewer
- ✅ Interactive 3D viewing in notebooks
- ✅ Face coloring and selection
- ✅ ML prediction visualization
- ✅ Multiple display modes
- ✅ Automatic resource management

### Visualization Utils
- ✅ ColorPalette for label-color management
- ✅ Automatic color generation
- ✅ Prediction grouping utilities
- ✅ Matplotlib colormap integration

**Typical Workflow:**
```
DatasetExplorer → Filter Files → DatasetViewer → Image Grid / 3D Views
                                      ↓
                               CADViewer → Face Coloring → Visual Analysis
```

The Insights module transforms data analysis into visual understanding, making it easy to explore large CAD datasets, validate ML predictions, and communicate findings effectively.

---

## See Also

- **[DatasetExplorer Documentation](./DatasetExplorer_DatasetLoader_Documentation.md)** - Query and filter datasets
- **[Flow Documentation](./Flow_Documentation.md)** - Build processing pipelines
- **[FlowModel Architecture](./FlowModel_Architecture.md)** - ML model integration

---
