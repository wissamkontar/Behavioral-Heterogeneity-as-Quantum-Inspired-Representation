# Behavioral Heterogeneity as Quantum-Inspired Representation

This repository accompanies the paper:

**Elayan, M., & Kontar, W. (2026)**. "*Behavioral Heterogeneity as Quantum-Inspired Representation*." arXiv:2603.22729

---

## Overview

This project introduces a quantum-inspired framework for modeling driver behavior as a dynamic latent state rather than static categories.

- Drivers are represented using density matrices
- Behavioral features are embedded via Random Fourier Features (RFF)
- The framework captures:
  - Temporal evolution
  - Context sensitivity
  - Interaction between behavioral features

The goal is to move beyond rigid labels (e.g., aggressive vs. timid) and instead model **continuous behavioral transitions**.

---

## Framework

![Quantum Pipeline](quantum_pipeline.png)

The pipeline consists of:

1. **Behavioral Vector Construction**
   - Speed differences
   - Acceleration
   - Headway

2. **Nonlinear Embedding (RFF Mapping)**
   - Projects behavioral features into a higher-dimensional space

3. **Feature Map Structure**
   - Forms a structured latent representation

4. **Density Matrix Formation**
   - Produces a symmetric, positive semi-definite matrix  
   - Diagonal → feature importance  
   - Off-diagonal → feature interactions  

5. **Context Weighting**
   - Distance to pedestrian  
   - Stop distance  
   - Density  
   - Average speed  

---

## Data

This repository uses **Third Generation Simulation Data (TGSIM)**:

- Foggy Bottom dataset  
  https://catalog.data.gov/dataset/third-generation-simulation-data-tgsim-foggy-bottom-trajectories  

- I-395 dataset  
  https://catalog.data.gov/dataset/third-generation-simulation-data-tgsim-i-395-trajectories  

### Important Notes
- Raw datasets are **not included** due to file size limits  
- Processed datasets are also not included  
- Users must download raw data and generate datasets locally  

---

## Reproducibility Pipeline

### Step 1 — Download Raw Data
Download datasets from the links above.

### Step 2 — Create Processed Datasets
Run:
- dataset_creation_FB.ipynb
- dataset_creation_I395.ipynb

These notebooks:
- Clean trajectory data  
- Extract behavioral features  
- Prepare inputs for profiling  

### Step 3 — Run Profiling Framework

```bash
python quantum_driver_profiling.py
```

---

## Citation

```
@article{elayan2026quantum,
  title={Behavioral Heterogeneity as Quantum-Inspired Representation},
  author={Elayan, Mohammad and Kontar, Wissam},
  journal={arXiv preprint arXiv:2603.22729},
  year={2026}
}
```

---

## Contact
- Mohammad Elayan — melayan2@nebraska.edu
- Wissam Kontar — wkontar2@nebraska.edu

---

## Notes
- The framework is model-agnostic and can be integrated into:
  - Reinforcement learning
  - Traffic simulation (e.g., SUMO)
  - Behavioral analysis pipelines
- Full reproducibility is supported given access to raw TGSIM data.
