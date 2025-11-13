![HOOPSAI](images/HOOPS_AI.jpg)


# HOOPS-AI-tutorials
Tech Soft 3D tutorials of python CAD framework for machine learning development. Requires HOOPS AI

[![HOOPS AI module presentation video](https://img.youtube.com/vi/1refiX8maBY/hqdefault.jpg)](https://www.youtube.com/watch?v=1refiX8maBY&t=1179s "Focus on this segment of the video")


This repository provides a collection of tutorial materials designed to help users learn and apply the HOOPS AI library.

Developed in Python, it features Jupyter notebooks that guide users through practical use cases and workflows.

---

## Dependency

🪪 1. Get Access first!

Requirements: Python > 3.9 and MiniConda (24.x)
Ask your TECH SOFT 3D contact to give you acces to HOOPS AI 1.0 preview.
Go to: https://www.techsoft3d.com/developers/products/hoops-ai

```bash
pip install hoops-ai-installer .........(full code to be received by email)
hoops-ai-installer install --mode cpu
hoops-ai-installer install --mode gpu

```

Note: Set license in environmental variable before starting up Jupyter.

## Running the Notebooks

Download the input data for the notebooks
### Fetch test files for the tutorials - Require password to be received by email
Go to: https://transfer.techsoft3d.com/link/uu80PL9LRSyasrOuyBLNKL
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

If you have never used jupyter notebook for python code, you can learn the basics here:
https://justinbois.github.io/bootcamp/2020_fsri/lessons/l01_welcome.html

Each notebook contains step-by-step instructions and annotations.

visit our documentation here: https://docs.techsoft3d.com/hoops/ai/

Proceed through each notebook in order, running the cells interactively.

```bash
hoops_ai_tutorials/
│
├── notebooks/         # Jupyter notebooks for each tutorial
├── packages/          # Downloaded binary assets (unzipped before accesing them)
└── README.md
|___ ...
```

