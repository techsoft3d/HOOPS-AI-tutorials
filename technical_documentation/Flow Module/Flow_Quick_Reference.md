# HOOPS AI Flow - Quick Reference Card

## 🚀 Quick Start (3 Steps)

### 1. Create Task File (cad_tasks.py)

```python
import os
import hoops_ai
from hoops_ai.flowmanager import flowtask

# Set license at module level
hoops_ai.set_license(os.getenv("HOOPS_AI_LICENSE"), validate=False)

# Define schema at module level
from hoops_ai.storage.datasetstorage.schema_builder import SchemaBuilder
builder = SchemaBuilder(domain="MyDomain", version="1.0")
group = builder.create_group("data", "item", "Data group")
group.create_array("values", ["item"], "float32", "Values")
my_schema = builder.build()

@flowtask.extract(name="gather", inputs=["sources"], outputs=["files"])
def gather_files(source):
    return glob.glob(f"{source}/*.step")

@flowtask.transform(name="encode", inputs=["cad_file", "cad_loader", "storage"], 
                    outputs=["encoded"])
def encode_data(cad_file, cad_loader, storage):
    cad_model = cad_loader.create_from_file(cad_file)
    storage.set_schema(my_schema)
    # ... extract features ...
    storage.compress_store()
    return storage.get_file_path("")
```

### 2. Create Flow in Notebook

```python
from cad_tasks import gather_files, encode_data
import hoops_ai

flow = hoops_ai.create_flow(
    name="my_pipeline",
    tasks=[gather_files, encode_data],
    flows_outputdir="./output",
    max_workers=8,
    auto_dataset_export=True
)
```

### 3. Execute and Analyze

```python
output, summary, flow_file = flow.process(inputs={'sources': ["/path/to/cad"]})

print(f"Processed: {summary['file_count']} files")
print(f"Dataset: {summary['flow_data']}")

from hoops_ai.dataset import DatasetExplorer
explorer = DatasetExplorer(flow_output_file=flow_file)
explorer.print_table_of_contents()
```

---

## 📋 Decorator Signatures

### @flowtask.extract

```python
@flowtask.extract(
    name="task_name",              # Optional: defaults to function name
    inputs=["input_key"],          # List of input keys from flow inputs
    outputs=["output_key"],        # List of output keys produced
    parallel_execution=True        # Enable parallel execution
)
def my_extract(source: str) -> List[str]:
    return [...]
```

**Common Pattern**: File gathering, database queries

### @flowtask.transform

```python
@flowtask.transform(
    name="task_name",
    inputs=["cad_file", "cad_loader", "storage"],  # Framework provides loader & storage
    outputs=["result"],
    parallel_execution=True
)
def my_transform(cad_file: str, cad_loader: HOOPSLoader, storage: DataStorage) -> str:
    return "result_path"
```

**Common Pattern**: CAD encoding, feature extraction

### @flowtask.custom

```python
@flowtask.custom(
    name="task_name",
    inputs=["data"],
    outputs=["processed"],
    parallel_execution=False  # Often runs sequentially
)
def my_custom(data):
    return process(data)
```

**Common Pattern**: Aggregation, reporting, validation

---

## 🔧 Flow Configuration

```python
hoops_ai.create_flow(
    name="my_flow",                      # Required: Flow identifier
    tasks=[task1, task2, ...],           # Required: List of decorated functions
    flows_outputdir="./output",          # Required: Output directory
    max_workers=None,                    # Optional: None = auto-detect CPU count
    ml_task="Task description",          # Optional: ML task description
    debug=False,                         # False = parallel, True = sequential
    auto_dataset_export=True             # Auto-merge encoded data
)
```

---

## 📂 Output Structure

```
flows_outputdir/flows/{flow_name}/
├── {flow_name}.flow          # ⭐ Main output: JSON with all metadata
├── {flow_name}.dataset       # Merged Zarr dataset
├── {flow_name}.infoset       # File-level metadata (Parquet)
├── {flow_name}.attribset     # Categorical metadata (Parquet)
├── error_summary.json        # Errors encountered
├── flow_log.log              # Execution log
├── encoded/                  # Individual .data files
└── stream_cache/             # Visualization assets
```

---

## 🎯 Common Patterns

### Pattern: CAD Encoding

```python
@flowtask.transform(name="encode", inputs=["cad_file", "cad_loader", "storage"], 
                    outputs=["path"])
def encode(cad_file, cad_loader, storage):
    cad_model = cad_loader.create_from_file(cad_file)
    storage.set_schema(my_schema)
    
    hoops_tools = HOOPSTools()
    hoops_tools.adapt_brep(cad_model, None)
    
    brep_encoder = BrepEncoder(cad_model.get_brep(), storage)
    brep_encoder.push_face_attributes()
    
    storage.save_data("group/array", np.array([...]))
    storage.save_metadata("key", "value")
    storage.compress_store()
    
    return storage.get_file_path("")
```

### Pattern: Multi-Directory Gathering

```python
@flowtask.extract(name="gather", inputs=["sources"], outputs=["files"])
def gather(sources: List[str]) -> List[str]:
    all_files = []
    for source in sources:
        all_files.extend(glob.glob(f"{source}/**/*.step", recursive=True))
    return all_files
```

### Pattern: Conditional Processing

```python
@flowtask.transform(...)
def encode_selective(cad_file, cad_loader, storage):
    cad_model = cad_loader.create_from_file(cad_file)
    
    if cad_model.get_face_count() < 10:
        return None  # Skip simple models
    
    # Continue with encoding...
```

---

## ⚙️ Windows ProcessPoolExecutor Rules

### ✅ DO (Required)

1. **Define tasks in .py files** (not notebooks)
2. **Set license at module level**:
   ```python
   hoops_ai.set_license(os.getenv("HOOPS_AI_LICENSE"), validate=False)
   ```
3. **Define schema at module level**:
   ```python
   builder = SchemaBuilder(...)
   my_schema = builder.build()  # Available to all workers
   ```
4. **Import tasks into notebook**:
   ```python
   from cad_tasks import gather_files, encode_data
   ```

### ❌ DON'T (Will Fail)

1. ❌ Define tasks in notebook cells
2. ❌ Set license only in notebook
3. ❌ Define schema only in notebook
4. ❌ Use `max_workers > 1` with notebook-defined tasks on Windows

---

## 🔍 Debugging

### Enable Sequential Mode

```python
flow = hoops_ai.create_flow(..., debug=True)  # Sequential execution
```

### Check Logs

```python
# Read flow log
with open(f"{output_dir}/flows/{flow_name}/flow_log.log") as f:
    print(f.read())

# Check errors
import json
with open(f"{output_dir}/flows/{flow_name}/error_summary.json") as f:
    errors = json.load(f)
    print(f"Errors: {len(errors)}")
```

### Test with Small Dataset

```python
# Test with 10 files first
test_sources = ["/path/to/small/dataset"]
test_flow = hoops_ai.create_flow(
    name="test_run",
    tasks=[gather_files, encode_data],
    flows_outputdir="./test_output",
    max_workers=2,
    debug=True
)
```

---

## 📊 Performance Tuning

### Rule of Thumb: max_workers

| Task Type | Recommended max_workers |
|-----------|-------------------------|
| I/O-bound (file gathering) | `cpu_count * 2` |
| CPU-bound (CAD encoding) | `cpu_count` |
| Memory-intensive | `RAM_GB / 4` |

### Dataset Merging Chunk Sizes

```python
# Small datasets (<100K faces total)
merger.merge_data(face_chunk=100_000, batch_size=50)

# Large datasets (>1M faces total)
merger.merge_data(face_chunk=1_000_000, batch_size=500)
```

---

## 🔗 Flow Outputs → Next Steps

### Use DatasetExplorer

```python
from hoops_ai.dataset import DatasetExplorer

explorer = DatasetExplorer(flow_output_file=flow_file)
explorer.print_table_of_contents()

# Query data
file_list = explorer.get_file_list(
    group="labels",
    where=lambda ds: ds['category'] == 5
)
```

### Prepare for ML Training

```python
from hoops_ai.dataset import DatasetLoader

loader = DatasetLoader(
    merged_store_path=summary['flow_data'],
    parquet_file_path=summary['flow_info']
)

train_size, val_size, test_size = loader.split(
    key="category",
    group="labels",
    train=0.7,
    validation=0.15,
    test=0.15
)

train_dataset = loader.get_dataset("train")
```

---

## 🆘 Common Issues

### Issue: PicklingError on Windows

**Cause**: Tasks defined in notebook  
**Fix**: Move tasks to `.py` file

### Issue: License not found in workers

**Cause**: License set only in notebook  
**Fix**: Set license at module level in `cad_tasks.py`

### Issue: Schema not found in merged dataset

**Cause**: Schema defined only in notebook  
**Fix**: Define schema at module level in `cad_tasks.py`

### Issue: Out of memory during merging

**Cause**: Chunk sizes too large  
**Fix**: Reduce `face_chunk` parameter in `merge_data()`

### Issue: Slow file gathering

**Cause**: Sequential glob operations  
**Fix**: Increase `max_workers` for extract tasks

---

## 📖 Full Documentation

See [Flow_Documentation.md](Flow_Documentation.md) for complete details including:
- Architecture deep dive
- Advanced patterns
- Custom storage providers
- Multi-stage pipelines
- Best practices

---

## 💡 Tips

1. **Start small**: Test with 10-100 files before processing thousands
2. **Monitor resources**: Check memory usage during encoding
3. **Profile execution**: Use `.flow` file to identify bottlenecks
4. **Use schemas**: Always define schemas for predictable data organization
5. **Handle errors gracefully**: Let the framework collect errors, inspect later
6. **Clean up**: Delete test outputs before production runs
7. **Document tasks**: Add docstrings to all decorated functions
8. **Version control**: Track schemas and task definitions in git

---

**HOOPS AI Flow: From CAD files to ML-ready datasets in 3 simple steps!**
