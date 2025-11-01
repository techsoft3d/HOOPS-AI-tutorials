# Acknowledgements

## Third-Party Machine Learning Architectures

HOOPS AI integrates open-source machine learning architectures to provide state-of-the-art CAD analysis capabilities. These models are located in the `src/hoops_ai/ml/_thirdparty/` directory and are used under their respective open-source licenses.

---

## Integrated Architectures

### 1. UV-Net - Graph Classification Architecture

**Original Authors:** Jayaraman, P. K., Sanghi, A., Lambourne, J. G., Willis, K. D. D., Davies, T., Shayani, H., & Morris, N.  
**Organization:** Autodesk AI Lab  
**Year:** 2021  
**License:** MIT License  
**Source:** https://github.com/AutodeskAILab/UV-Net  
**Location in HOOPS AI:** `src/hoops_ai/ml/_thirdparty/uvnet/`

**Publication:**
> Jayaraman, P. K., Sanghi, A., Lambourne, J. G., Willis, K. D. D., Davies, T., Shayani, H., & Morris, N. (2021).  
> UV-Net: Learning from Boundary Representations.  
> In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 11703-11712).  
> https://doi.org/10.1109/CVPR46437.2021.01153

**BibTeX Citation:**
```bibtex
@inproceedings{jayaraman2021uvnet,
  title={UV-Net: Learning from Boundary Representations},
  author={Jayaraman, Pradeep Kumar and Sanghi, Aditya and Lambourne, Joseph G and Willis, Karl DD and Davies, Thomas and Shayani, Hooman and Morris, Nigel},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11703--11712},
  year={2021}
}
```

**Original Applications:**
- Auto-complete of modeling operations in CAD software
- Smart selection tools
- Shape similarity search
- Design recommendation systems

---

### 2. BrepMFR - Graph Node Classification Architecture

**Original Authors:** Zhang, S., Guan, Z., Jiang, H., Wang, X., & Tan, P.  
**Year:** 2024  
**License:** MIT License  
**Source:** https://github.com/zhangshuming0668/BrepMFR  
**Location in HOOPS AI:** `src/hoops_ai/ml/_thirdparty/brepmfr/`

**Publication:**
> Zhang, S., Guan, Z., Jiang, H., Wang, X., & Tan, P. (2024).  
> BrepMFR: Enhancing machining feature recognition in B-rep models through deep learning and domain adaptation.  
> *Computer Aided Geometric Design*, 111, 102318.  
> https://www.sciencedirect.com/science/article/abs/pii/S0167839624000529

**BibTeX Citation:**
```bibtex
@article{zhang2024brepmfr,
  title={BrepMFR: Enhancing machining feature recognition in B-rep models through deep learning and domain adaptation},
  author={Zhang, Shuming and Guan, Zhiguang and Jiang, Han and Wang, Xiaojun and Tan, Ping},
  journal={Computer Aided Geometric Design},
  volume={111},
  pages={102318},
  year={2024},
  publisher={Elsevier}
}
```

**Original Applications:**
- Machining feature recognition in CAD/CAM workflows
- Recognizing highly intersecting features with complex geometries
- Automated process planning for CNC machining
- Design for manufacturability analysis

---

## Tech Soft 3D's Role and Modifications

### Attribution and Compliance

**Tech Soft 3D's Position on Third-Party Code:**

While HOOPS AI provides convenient wrappers (`GraphClassification`, `GraphNodeClassification`) to integrate these architectures into the Flow Model framework, **the original authors retain full credit for their pioneering work**. 

### Our Modifications

Our contributions are limited to:
1. **Interface Adaptation:** Implementing the `FlowModel` abstract interface for seamless integration with HOOPS AI workflows
2. **Storage Integration:** Connecting to HOOPS AI's data storage system (Zarr, DGL, etc.)
3. **Training Infrastructure:** Enabling use with `FlowTrainer` and `FlowInference` components
4. **Error Handling and Logging:** Enhanced debugging capabilities and error reporting
5. **Documentation:** Technical documentation adapted for HOOPS AI users

**We do NOT claim authorship of the underlying ML architectures.** Users of HOOPS AI should cite the original papers when publishing results using these models.

---

## MIT License Compliance

Both integrated models are distributed under the **MIT License**, which permits:
- ✅ Commercial use
- ✅ Modification
- ✅ Distribution
- ✅ Private use

**Requirements:**
- Include the original copyright notice
- Include the MIT license text
- Acknowledge modifications made by Tech Soft 3D

### How HOOPS AI Complies

HOOPS AI complies with these requirements by:
1. Preserving original copyright notices in source files
2. Including LICENSE files in `_thirdparty/` subdirectories
3. Clearly documenting modifications in technical documents
4. Providing citation information in this documentation
5. Maintaining this dedicated Acknowledgements document

---

## Using These Models in Your Research

### Citation Requirements

If you publish research results using HOOPS AI's `GraphClassification` or `GraphNodeClassification` models, please cite:

1. **The original architecture paper** (see BibTeX citations above)
2. **HOOPS AI** (if relevant to your workflow):
   ```
   Tech Soft 3D. (2025). HOOPS AI - Machine Learning Framework for CAD Data Analysis.
   https://github.com/techsoft3d/hoops-ai
   ```

### Example Acknowledgement Text

> "This work uses the UV-Net architecture [1] and BrepMFR architecture [2] integrated into the HOOPS AI framework [3] for CAD data processing and machine learning."
>
> [1] Jayaraman et al., "UV-Net: Learning from Boundary Representations", CVPR 2021  
> [2] Zhang et al., "BrepMFR: Enhancing machining feature recognition...", CAGD 2024  
> [3] Tech Soft 3D, "HOOPS AI", 2025

---

## License Texts

### UV-Net MIT License

```
MIT License

Copyright (c) 2021 Autodesk AI Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### BrepMFR MIT License

```
MIT License

Copyright (c) 2024 Zhang Shuming and contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Contact

For questions about:
- **Original UV-Net architecture:** Contact Autodesk AI Lab or refer to the GitHub repository
- **Original BrepMFR architecture:** Contact the authors or refer to the GitHub repository
- **HOOPS AI integration:** Contact Tech Soft 3D support

---

**Document Version:** 1.0  
**Last Updated:** November 1, 2025  
**Maintainer:** Tech Soft 3D ML Team
