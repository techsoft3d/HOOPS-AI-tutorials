![HOOPSAI](images/HOOPS-AI.png)

# HOOPS AI 1.0 - March 31th, 2026

see release notes here: https://docs.techsoft3d.com/hoops/ai/resources/release_notes/index.html

# Documentation

visit our official documentation here: https://docs.techsoft3d.com/hoops/ai/

# HOOPS-AI-tutorials
Tech Soft 3D tutorials of python CAD framework for machine learning development - HOOPS AI

Recording of the webinar hold on November 6th. Presenting HOOPS AI, check it out!:
[![HOOPS AI module presentation video](https://img.youtube.com/vi/1refiX8maBY/hqdefault.jpg)](https://www.youtube.com/watch?v=1refiX8maBY&t=15m40s "Explanation of HOOPS AI Modules + live demos")


This repository provides a collection of tutorial materials designed to help users learn and apply the HOOPS AI library.

Developed in Python, it features Jupyter notebooks that guide users through practical use cases and workflows.

---

## Install HOOPS AI

1. Get Access first! You will need to set your HOOPS AI LICENSE to use HOOPS AI

Go to: https://www.techsoft3d.com/developers/products/hoops-ai

see section Evaluate and Install from our doc: https://docs.techsoft3d.com/hoops/ai/getting_started/evaluate.html


## Running the Notebooks

Download the input data for the notebooks
### Fetch test files for the tutorials - [optional] please contact your Tech Soft 3D representative
Go to: https://transfer.techsoft3d.com/link/KQTgpzP9tRFBHCuzhm4nu2

extract the zip and place the packages folde rnext to the notebooks folder. 

### Activate the development environment
Open a cmd in the location of this repo. If HOOPS AI is installed, then activate the environment of you choice (cpu or gpu):

```bash
conda activate hoops_ai_gpu
```

or

```bash
conda activate hoops_ai_cpu
```

then run inside the environment the jupyter lab command:
```bash
jupyter lab
```

A browser will open with your local folder, in the environment double click the folder notebooks and follow each notebook instructions.

Each notebook contains step-by-step instructions and annotations.

Proceed through each notebook in order, running the cells interactively.

```bash
hoops_ai_tutorials/
│
├── notebooks/         # Jupyter notebooks for each tutorial
├── packages/          # Downloaded binary assets (unzipped before accesing them)
└── README.md
|___ ...
```

If you have never used jupyter notebook for python code, you can learn the basics here:
https://justinbois.github.io/bootcamp/2020_fsri/lessons/l01_welcome.html