# Module Access & Encoder Documentation

## Overview

This document provides comprehensive documentation for the **CAD Access** and **Encoder** modules in HOOPS AI. These modules form the foundation for extracting geometric and topological information from CAD files and encoding them into machine learning-ready formats.

The system follows a flow-based architecture:
```
CAD File → HOOPSLoader → HOOPSModel → BrepAccess → BrepEncoder → Storage
```

---

## Table of Contents

1. [CAD Access Module](#cad-access-module)
   - [HOOPSLoader](#hoopsloader)
   - [HOOPSModel](#hoopsmodel)
   - [HOOPSBrep](#hoopsbrep)
2. [Encoder Module](#encoder-module)
   - [BrepEncoder Overview](#brepencoder-overview)
   - [Geometric Methods](#geometric-methods)
   - [Topological Methods](#topological-methods)
   - [Histogram Methods](#histogram-methods)

---

## CAD Access Module

The CAD Access module provides interfaces to load and navigate CAD files using HOOPS Exchange.

### HOOPSLoader

The `HOOPSLoader` class implements a singleton pattern for efficient CAD file loading.

#### Initialization

```python
from hoops_ai.cadaccess import HOOPSLoader

loader = HOOPSLoader()
```

**Key Responsibilities:**
- Initialize HOOPS Exchange with license management
- Maintain a single Exchange instance across the application
- Load CAD files and create model representations

#### Methods

##### `create_from_file(filename: str) -> HOOPSModel`

Loads a CAD file and returns a `HOOPSModel` instance.

**Parameters:**
- `filename` (str): Absolute path to the CAD file

**Returns:**
- `HOOPSModel`: A model object containing the loaded CAD data

**Example:**
```python
loader = HOOPSLoader()
model = loader.create_from_file("path/to/file.step")
```

##### `get_general_options() -> Dict[str, Any]`

Returns current loading options.

**Returns:**
- Dictionary with keys:
  - `read_feature`: Whether to read feature information
  - `read_solid`: Whether to read solid geometry

##### `set_general_options(general_options: Dict[str, Any]) -> None`

Configures CAD file loading behavior.

**Parameters:**
- `general_options`: Dictionary with loading configuration

---

### HOOPSModel

Represents a loaded CAD model with access to its body, BREP, and mesh representations.

#### Methods Required for BrepEncoder

##### `get_brep(body_index: int = 0) -> BrepAccess`

Extracts the Boundary Representation (BREP) from the CAD model.

**Parameters:**
- `body_index` (int): Index of the body to extract (default: 0)

**Returns:**
- `HOOPSBrep`: A BREP access object for geometry/topology extraction

**Usage:**
```python
model = loader.create_from_file("part.step")
brep = model.get_brep()
```

---

### HOOPSBrep

Provides access to BREP geometry and topology through the `BrepAccess` interface.

#### Key Methods Used by BrepEncoder

- `get_face_indices()`: Returns list of face identifiers
- `get_edge_indices()`: Returns list of edge identifiers
- `get_bounding_box()`: Returns (xmin, ymin, zmin, xmax, ymax, zmax)
- `build_face_adjacency_graph()`: Creates NetworkX graph of face connectivity
- `uvgrid(face_index, num_u, num_v, method)`: Samples points/normals on faces
- `ugrid(edge_index, num_u)`: Samples points along edges
- `get_face_attributes(face_idx, spec)`: Extracts face properties
- `get_edge_attributes(edge_idx, spec)`: Extracts edge properties

---

## Encoder Module

### BrepEncoder Overview

The `BrepEncoder` class computes and persists geometric and topological features from BREP data. It follows a **push-based architecture** where each method:

1. Checks if data already exists in storage
2. Ensures the appropriate schema definition exists for the data
3. Computes the feature if needed
4. Saves to storage with schema management
5. Returns `None` (if storage is used) or the computed data (if no storage)

The encoder automatically manages schemas for data organization, creating groups and arrays as needed during the encoding process.

#### Initialization

```python
from hoops_ai.cadencoder import BrepEncoder
from hoops_ai.storage import DataStorage

# With storage
storage = DataStorage(...)
encoder = BrepEncoder(brep_access=brep, storage_handler=storage)

# Without storage (returns raw data)
encoder = BrepEncoder(brep_access=brep)
```

**Parameters:**
- `brep_access` (`BrepAccess`): BREP interface from a loaded CAD model
- `storage_handler` (`DataStorage`, optional): Storage backend for persistence

---

## Geometric Methods

### push_face_indices()

**Purpose:** Extract and store unique identifiers for all faces in the model.

**Mathematical Formulation:**

$$
\mathcal{F} = \{f_0, f_1, \ldots, f_{N_f-1}\}
$$

where $\mathcal{F}$ is the set of face indices and $N_f$ is the total number of faces.

**Storage:**
- **Group:** `faces`
- **Array:** `face_indices`
- **Shape:** `[face]`
- **Dtype:** `int32`

**Returns:**
- With storage: `str` - key name `"face_indices"`
- Without storage: `np.ndarray` of shape `(N_f,)`

---

### push_edge_indices()

**Purpose:** Extract and store unique identifiers for all edges in the model.

**Mathematical Formulation:**

$$
\mathcal{E} = \{e_0, e_1, \ldots, e_{N_e-1}\}
$$

where $\mathcal{E}$ is the set of edge indices and $N_e$ is the total number of edges.

**Storage:**
- **Group:** `edges`
- **Array:** `edge_indices`
- **Shape:** `[edge]`
- **Dtype:** `int32`

**Returns:**
- With storage: `str` - key name `"edge_indices"`
- Without storage: `np.ndarray` of shape `(N_e,)`

---

### push_face_attributes()

**Purpose:** Compute geometric and topological properties of each face.

**Mathematical Formulation:**

For each face $f_i \in \mathcal{F}$:

1. **Surface Type** $\tau(f_i)$: Categorical classification (plane, cylinder, sphere, etc.)

$$
\tau: \mathcal{F} \rightarrow \mathbb{Z}^+
$$

2. **Face Area** $A(f_i)$: Surface integral over the face

$$
A(f_i) = \iint_{S_i} dS
$$
   
3. **Loop Count** $L(f_i)$: Number of boundary loops (including holes)

$$
L(f_i) = |\{\text{loops in } f_i\}|
$$

**Storage:**
- **Arrays:** `face_types`, `face_areas`, `face_loops`
- **Shapes:** All `[face]`
- **Dtypes:** `int32`, `float32`, `int32`

**Returns:**
- With storage: Returns `None` (data is stored with keys: `"face_types"`, `"face_areas"`, `"face_loops"`, and metadata `"descriptions/face_types"`)
- Without storage: `Tuple[List[np.ndarray], Dict]` - (list of numpy arrays [face_types, face_areas, face_loops], face_types_descr dictionary mapping face type IDs to descriptions)

---

### push_face_discretization(pointsamples=25)

**Purpose:** Sample points and normals on face surfaces using uniform point sampling (rather than structured UV grids).

**Mathematical Formulation:**

For each face $f_i$, sample $P$ points uniformly across the surface:

$$
\mathbf{P}_{i} = \left[\mathbf{S}(\mathbf{u}_j), \mathbf{N}(\mathbf{u}_j), V(\mathbf{u}_j)\right]_{j=1}^{P}
$$

where:
- $\mathbf{S}: \Omega \rightarrow \mathbb{R}^3$ is the surface parameterization
- $\mathbf{N}: \Omega \rightarrow \mathbb{S}^2$ is the normal field
- $V: \Omega \rightarrow \{0,1\}$ is visibility status (inside/outside)
- $\mathbf{u}_j$ are uniformly sampled parameter points across the face
- $P$ is the number of sample points (default: 25)

The sampling uses three methods concatenated along the component axis:
1. **Point samples**: $(x, y, z)$ coordinates
2. **Normal samples**: $(n_x, n_y, n_z)$ unit normals
3. **Inside/outside flags**: visibility indicators

**Storage:**
- **Array:** `face_discretization`
- **Shape:** `[face, sample, component]` where component includes (x,y,z) + (nx,ny,nz) + (visibility)
- **Dtype:** `float32`

**Parameters:**
- `pointsamples` (int): Number of points to sample per face (default: 25)

**Returns:**
- With storage: `str` - key name `"face_discretization"`
- Without storage: `np.ndarray` of shape `(N_f, pointsamples, 7)`

---

### push_edge_attributes()

**Purpose:** Compute geometric and topological properties of each edge.

**Mathematical Formulation:**

For each edge $e_i \in \mathcal{E}$:

1. **Curve Type** $\kappa(e_i)$: Categorical classification (line, circle, spline, etc.)

$$
\kappa: \mathcal{E} \rightarrow \mathbb{Z}^+
$$

2. **Edge Length** $\ell(e_i)$: Arc length of the curve

$$
\ell(e_i) = \int_{0}^{1} \left\| \frac{d\mathbf{C}(t)}{dt} \right\| dt
$$

    where $\mathbf{C}(t)$ is the curve parameterization.

3. **Dihedral Angle** $\theta(e_i)$: Angle between adjacent face normals

$$
\theta(e_i) = \arccos(\mathbf{n}_1 \cdot \mathbf{n}_2)
$$

   where $\mathbf{n}_1, \mathbf{n}_2$ are unit normals of adjacent faces.

4. **Convexity** $\chi(e_i) \in \{-1, 0, 1\}$:

$$
\chi(e_i) = \begin{cases}
1 & \text{if convex} \\
0 & \text{if smooth} \\
-1 & \text{if concave}
\end{cases}
$$

**Storage:**
- **Arrays:** `edge_types`, `edge_lengths`, `edge_dihedral_angles`, `edge_convexities`
- **Shapes:** All `[edge]`
- **Dtypes:** `int32`, `float32`, `float32`, `int32`

**Returns:**
- With storage: Returns `None` (data is stored with keys: `"edge_types"`, `"edge_lengths"`, `"edge_dihedral_angles"`, `"edge_convexities"`, and metadata `"descriptions/edge_types"`)
- Without storage: `Tuple[List[np.ndarray], Dict]` - (list of numpy arrays [edge_types, edge_lengths, edge_dihedrals, edge_convexities], edge_type_descrip dictionary mapping edge type IDs to descriptions)

---

### push_curvegrid(ugrid=5)

**Purpose:** Sample points and tangents along edge curves.

**Mathematical Formulation:**

For each edge $e_i$, sample along the curve parameter:

$$
\mathbf{C}_i = \left[\mathbf{C}(t_j), \mathbf{T}(t_j)\right]_{j=0}^{U-1}
$$

where:
- $\mathbf{C}: [0,1] \rightarrow \mathbb{R}^3$ is the curve
- $\mathbf{T}(t) = \frac{d\mathbf{C}(t)}{dt}$ is the tangent vector
- $t_j = \frac{j}{U-1}$ for $j = 0, \ldots, U-1$

**Storage:**
- **Array:** `edge_u_grids`
- **Shape:** `[edge, u, component]` where component includes (x,y,z) + (tx,ty,tz)
- **Dtype:** `float32`

**Parameters:**
- `ugrid` (int): Number of samples along edge (default: 5)

**Returns:**
- With storage: `str` - key name `"edge_u_grids"`
- Without storage: `np.ndarray` of shape `(N_e, ugrid, 6)`

---

## Topological Methods

### push_face_adjacency_graph()

**Purpose:** Build a graph representation of face connectivity where faces are nodes and edges represent shared boundaries.

**Mathematical Formulation:**

Define an undirected graph $G = (V, E)$ where:

$$
V = \mathcal{F} = \{f_0, f_1, \ldots, f_{N_f-1}\}
$$

$$
E = \{(f_i, f_j) : f_i \text{ and } f_j \text{ share an edge}\}
$$

The graph is represented by:
- Node count: $|V| = N_f$
- Edge list: $\{(s_k, d_k)\}_{k=0}^{|E|-1}$ where $s_k, d_k \in V$

**Storage:**
- **Arrays:** 
  - `num_nodes`: scalar count of nodes in the graph
  - `edges_source`: source node indices for each edge
  - `edges_destination`: destination node indices for each edge
  - `graph`: nested structure containing edges dict and num_nodes (for backward compatibility)
- **Dtypes:** `int32`

**Returns:**
- With storage: Returns `None` (data is stored with keys: `"num_nodes"`, `"edges_source"`, `"edges_destination"`, and `"graph"`)
- Without storage: `nx.Graph` - NetworkX graph object with edge attributes

---

### push_extended_adjacency()

**Purpose:** Compute all-pairs shortest path distances in the face adjacency graph.

**Mathematical Formulation:**

Compute the graph distance matrix $\mathbf{D}_G \in \mathbb{R}^{N_f \times N_f}$:

$$
D_G[i,j] = \begin{cases}
0 & \text{if } i = j \\
\min\{|p| : p \text{ is path from } f_i \text{ to } f_j\} & \text{if path exists} \\
\infty & \text{otherwise}
\end{cases}
$$

where $|p|$ is the number of edges in path $p$.

This is computed using the Floyd-Warshall or BFS algorithm via NetworkX's `all_pairs_shortest_path_length`.

**Storage:**
- **Array:** `extended_adjacency`
- **Shape:** `[node_i, node_j]`
- **Dtype:** `float32`

**Returns:**
- With storage: Returns `None` (data is stored with key `"extended_adjacency"`)
- Without storage: `np.ndarray` of shape `(N_f, N_f)`

---

### push_face_neighbors_count()

**Purpose:** Count the number of adjacent faces for each face (node degree in the graph).

**Mathematical Formulation:**

For each face $f_i$, compute the degree:

$$
\deg(f_i) = |\{f_j \in \mathcal{F} : (f_i, f_j) \in E\}|
$$

**Storage:**
- **Array:** `face_neighborscount`
- **Shape:** `[face]`
- **Dtype:** `int32`

**Returns:**
- With storage: Returns `None` (data is stored with key `"face_neighborscount"`)
- Without storage: `np.ndarray` of shape `(N_f,)`

---

### push_face_pair_edges_path(max_allow_edge_length=16)

**Purpose:** Store the sequence of shared edges along the shortest path between every pair of faces.

**Mathematical Formulation:**

For each face pair $(f_i, f_j)$, find the shortest path:

$$
p_{ij} = [f_i = v_0, v_1, \ldots, v_k = f_j]
$$

Then extract the edge sequence:

$$
\mathbf{e}_{ij} = [e(v_0, v_1), e(v_1, v_2), \ldots, e(v_{k-1}, v_k)]
$$

where $e(u,v)$ is the edge index connecting faces $u$ and $v$.

If $|\mathbf{e}_{ij}| > M$ (max_allow_edge_length), truncate to first $M$ edges.
Pad with $-1$ if path is shorter.

**Storage:**
- **Array:** `face_pair_edges_path`
- **Shape:** `[face_i, face_j, path_idx]`
- **Dtype:** `int32`

**Parameters:**
- `max_allow_edge_length` (int): Maximum path length to store (default: 16)

**Returns:**
- With storage: Returns `None` (data is stored with key `"face_pair_edges_path"`)
- Without storage: `np.ndarray` of shape `(N_f, N_f, M)`

---

## Histogram Methods

### push_average_face_pair_distance_histograms(grid=5, num_bins=64)

**Purpose:** Compute normalized histograms of pairwise point-to-point distances between all face pairs (D2 shape descriptor).

**Implementation Notes:**
- Uses optimized sampling: maximum 25 points per face (or fewer if face has less than 25 points)
- Employs 2-thread parallel processing for improved performance
- Processes faces in two chunks to balance memory and computation

**Mathematical Formulation:**

1. **Sample Points:** For each face $f_i$, sample $P$ points uniformly:

$$
\mathcal{P}_i = \{\mathbf{p}_1^i, \mathbf{p}_2^i, \ldots, \mathbf{p}_P^i\} \subset \mathbb{R}^3
$$

2. **Compute Distances:** For faces $f_i$ and $f_j$, compute all pairwise distances:

$$
d_{ij}^{mn} = \|\mathbf{p}_m^i - \mathbf{p}_n^j\|_2, \quad m,n = 1,\ldots,P
$$

3. **Normalize by Diagonal:** Let $D$ be the bounding box diagonal:

$$
D = \|\mathbf{b}_{\max} - \mathbf{b}_{\min}\|_2
$$
   
Normalized distances:

$$
\tilde{d}_{ij}^{mn} = \frac{d_{ij}^{mn}}{D}
$$

4. **Build Histogram:** Bin the normalized distances into $B$ bins over $[0,1]$:

$$
H_{ij}[b] = \frac{1}{P^2} \sum_{m=1}^P \sum_{n=1}^P \mathbb{1}\left[\frac{b}{B} \leq \tilde{d}_{ij}^{mn} < \frac{b+1}{B}\right]
$$

Result: $\mathbf{H} \in \mathbb{R}^{N_f \times N_f \times B}$ where $H_{ij}$ is the distance histogram between faces $i$ and $j$.

**Storage:**
- **Group:** `histograms`
- **Array:** `d2_distance`
- **Shape:** `[face_i, face_j, bin]`
- **Dtype:** `float32`

**Parameters:**
- `grid` (int): Grid density for sampling (default: 5)
- `num_bins` (int): Number of histogram bins (default: 64)

**Returns:**
- With storage: Returns `None` (data is stored with key `"d2_distance"`)
- Without storage: `np.ndarray` of shape `(N_f, N_f, num_bins)`

---

### push_average_face_pair_angle_histograms(grid=5, num_bins=64)

**Purpose:** Compute normalized histograms of pairwise normal-to-normal angles between all face pairs (A3 shape descriptor).

**Implementation Notes:**
- Uses optimized sampling: maximum 25 normals per face (or fewer if face has less than 25 normals)
- Employs 2-thread parallel processing for improved performance
- Processes faces in two chunks to balance memory and computation

**Mathematical Formulation:**

1. **Sample Normals:** For each face $f_i$, sample $P$ normal vectors:

$$
\mathcal{N}_i = \{\mathbf{n}_1^i, \mathbf{n}_2^i, \ldots, \mathbf{n}_P^i\} \subset \mathbb{S}^2
$$

2. **Compute Angles:** For faces $f_i$ and $f_j$, compute all pairwise angles:

$$
\theta_{ij}^{mn} = \arccos(\mathbf{n}_m^i \cdot \mathbf{n}_n^j), \quad m,n = 1,\ldots,P
$$
   
Clamping: $\mathbf{n}_m^i \cdot \mathbf{n}_n^j \in [-1, 1]$ to avoid numerical issues.

3. **Normalize to [0,1]:** 

$$
\tilde{\theta}_{ij}^{mn} = \frac{\theta_{ij}^{mn}}{\pi}
$$

4. **Build Histogram:** Bin the normalized angles into $B$ bins:

$$
H_{ij}^{\theta}[b] = \frac{1}{P^2} \sum_{m=1}^P \sum_{n=1}^P \mathbb{1}\left[\frac{b}{B} \leq \tilde{\theta}_{ij}^{mn} < \frac{b+1}{B}\right]
$$

Result: $\mathbf{H}^{\theta} \in \mathbb{R}^{N_f \times N_f \times B}$ where $H_{ij}^{\theta}$ is the angle histogram between faces $i$ and $j$.

**Storage:**
- **Group:** `histograms`
- **Array:** `a3_distance`
- **Shape:** `[face_i, face_j, bin]`
- **Dtype:** `float32`

**Parameters:**
- `grid` (int): Grid density for sampling normals (default: 5)
- `num_bins` (int): Number of histogram bins (default: 64)

**Returns:**
- With storage: Returns `None` (data is stored with key `"a3_distance"`)
- Without storage: `np.ndarray` of shape `(N_f, N_f, num_bins)`

---

## Usage Example

### Complete Workflow

```python
from hoops_ai.cadaccess import HOOPSLoader
from hoops_ai.cadencoder import BrepEncoder
from hoops_ai.storage import OptStorage

# 1. Load CAD file
loader = HOOPSLoader()
model = loader.create_from_file("part.step")

# 2. Extract BREP
brep = model.get_brep()

# 3. Initialize storage and encoder
storage = OptStorage(output_path="./encoded_data")
encoder = BrepEncoder(brep_access=brep, storage_handler=storage)

# 4. Extract geometric features
encoder.push_face_indices()
encoder.push_edge_indices()
encoder.push_face_attributes()
encoder.push_edge_attributes()

# 5. Extract parameterized grids
encoder.push_face_discretization(pointsamples=100)
encoder.push_curvegrid(ugrid=20)

# 6. Extract topology
encoder.push_face_adjacency_graph()
encoder.push_extended_adjacency()
encoder.push_face_neighbors_count()

# 7. Extract shape descriptors
encoder.push_average_face_pair_distance_histograms(grid=7, num_bins=64)
encoder.push_average_face_pair_angle_histograms(grid=7, num_bins=64)

print("Encoding complete!")
```

---

## Performance Considerations

### Memory Management
- The encoder uses a **push-and-discard** pattern: data is computed, saved, and not kept in memory
- Large arrays (histograms) use chunked processing with ThreadPoolExecutor
- UV grids and curve grids are stacked only temporarily

### Parallelization
- Face pair histograms use 2-thread parallel processing
- Sampling operations are vectorized with NumPy
- Graph algorithms leverage NetworkX's optimized implementations

### Storage Efficiency
- Float32 is used throughout for memory/disk efficiency
- Zarr format provides compression and chunked access
- Schema management ensures consistent data organization

---

## Mathematical Notation Summary

| Symbol | Meaning |
|--------|---------|
| $\mathcal{F}$ | Set of faces |
| $\mathcal{E}$ | Set of edges |
| $N_f$ | Number of faces |
| $N_e$ | Number of edges |
| $f_i$ | Face with index $i$ |
| $e_i$ | Edge with index $i$ |
| $\mathbf{N}(u,v)$ | Normal field |
| $\mathbf{C}(t)$ | Curve parameterization |
| $\mathbf{T}(t)$ | Tangent vector |
| $G = (V,E)$ | Face adjacency graph |
| $D_G$ | Graph distance matrix |
| $\theta$ | Angle (dihedral or between normals) |
| $H_{ij}$ | Histogram between faces $i$ and $j$ |
| $\mathbb{S}^2$ | Unit sphere (surface of unit ball) |


---

## License

Copyright (c) 2025 by Tech Soft 3D, Inc. All rights reserved.
