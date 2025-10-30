# FlowModel Documentation - Navigation Guide

## ⚠️ EXPERIMENTAL STATUS
This architecture is currently **EXPERIMENTAL** and may change in future releases.

---

## 📚 Documentation Structure

This documentation is organized into three focused documents:

### 1. **[FlowModel Architecture](./FlowModel_Architecture.md)** - Core Concepts
Understanding the abstract interface, FlowTrainer/FlowInference, and design philosophy.

### 2. **[GraphClassification (UV-Net)](./GraphClassification_UVNet.md)** - Graph-Level Tasks  
Whole-part classification (e.g., "bolt", "bearing", "bracket").

### 3. **[GraphNodeClassification (BrepMFR)](./GraphNodeClassification_BrepMFR.md)** - Node-Level Tasks  
Per-face classification (e.g., "hole", "pocket", "slot").

---

## 🚀 Quick Start

**New to FlowModel?**  
→ Start with [FlowModel Architecture](./FlowModel_Architecture.md)

**Ready to implement?**  
→ **Part classification:** [GraphClassification](./GraphClassification_UVNet.md)  
→ **Face segmentation:** [GraphNodeClassification](./GraphNodeClassification_BrepMFR.md)

**Need Flow integration examples?**  
→ See decorator patterns in both concrete implementation docs

---

## 📖 What's in Each Document

### FlowModel Architecture
- Abstract interface methods (`encode_cad_data`, `convert_encoded_data_to_graph`, etc.)
- FlowTrainer for batch training with dataset splitting
- FlowInference for single-file deployment
- Best practices and troubleshooting
- Third-party model licensing (MIT)

### GraphClassification (UV-Net)
- UV-Net architecture (CVPR 2021, Autodesk AI Lab)
- Initialization with `num_classes` parameter
- CAD encoding: 10×10 UV-grids for faces
- **Flow integration with `flowtask` decorators**
- Complete FABWAVE example (45 part classes)
- Training and inference workflows

### GraphNodeClassification (BrepMFR)
- BrepMFR architecture (CAGD 2024, Zhang et al.)
- Initialization with Transformer hyperparameters
- Rich encoding: extended adjacency, histograms
- **Flow integration with `flowtask` decorators**
- Complete CADSynth-AAG example (162k models)
- Advanced hyperparameter tuning

---

## 🎯 Problem Statement

**The Challenge:** CAD → ML workflows require identical encoding logic in:
1. **Training:** Batch processing thousands of CAD files
2. **Inference:** Real-time processing of new CAD files

**The Solution:** FlowModel encapsulates all encoding logic in one reusable interface:
```python
# Training: Used by FlowTrainer
flow_model.encode_cad_data(cad_file, cad_loader, storage)

# Inference: Used by FlowInference (same method!)
flow_model.encode_cad_data(cad_file, cad_loader, storage)
```

---

## 📦 Architecture Overview

```
FlowModel (Abstract)
├── GraphClassification (UV-Net)
│   ├── Graph-level predictions
│   └── Used by: FABWAVE dataset
└── GraphNodeClassification (BrepMFR)
    ├── Node-level predictions
    └── Used by: CADSynth-AAG dataset

Both consumed by:
├── FlowTrainer (batch training)
└── FlowInference (single-file deployment)
```

---

## 🔗 Integration with HOOPS AI Flow

Both concrete implementations show how to wrap FlowModel methods inside Flow tasks using the `flowtask` decorator pattern:

```python
from hoops_ai.flowmanager import flowtask
from hoops_ai.ml.EXPERIMENTAL import GraphClassification

# Instantiate FlowModel once
flow_model = GraphClassification(num_classes=45)

# Wrap encoding method in Flow task
@flowtask.transform(
    name="cad_encoder",
    inputs=["cad_file", "cad_loader", "storage"],
    outputs=["face_count", "edge_count"]
)
def my_encoder(cad_file, cad_loader, storage):
    # Call FlowModel method
    return flow_model.encode_cad_data(cad_file, cad_loader, storage)
```

See complete examples in:
- [GraphClassification FABWAVE Example](./GraphClassification_UVNet.md#integration-with-flow-tasks)
- [GraphNodeClassification CADSynth-AAG Example](./GraphNodeClassification_BrepMFR.md#integration-with-flow-tasks)

---

## 📚 Additional Resources

- **Flow Framework:** See `Flow_Documentation.md` for complete Flow system guide
- **Dataset Tools:** See `DatasetExplorer_DatasetLoader_Documentation.md`
- **Storage System:** See `DataStorage_Documentation.md`

---

## 📝 Version History

**Current Version:** 1.0 (EXPERIMENTAL)  
**Last Updated:** January 2025  
**Status:** Under active development - APIs may change
