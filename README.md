# 3D Fiber Resolution, Dual Neural Field & Topology Optimization Framework

A complete deep learning and geometric graph optimization framework for high-resolution 3D fiber microstructure segmentation, centerline extraction, transverse $H$-junction severing, and crossing resolution in dense biomaterials and micro-CT tomography volumes (collagen networks, fibrous composites, Altendorf-Jeulin stochastic models).

---

## 📑 Table of Contents

- [Core Principles & Decoupled Neural Fields](#-core-principles--decoupled-neural-fields)
- [Framework Architecture](#-framework-architecture)
- [Topology Optimization Formulation](#-topology-optimization-formulation)
- [Why H-Junctions Occur and How They Are Resolved](#-why-h-junctions-occur-and-how-they-are-resolved)
- [Interactive 3D Curation Web App & Engine](#-interactive-3d-curation-web-app--engine)
- [Directory Structure](#-directory-structure)
- [Quick Start & Usage Guide](#-quick-start--usage-guide)
  - [1. Dataset Precomputation](#1-dataset-precomputation)
  - [2. Interactive 3D Curation Web App](#2-interactive-3d-curation-web-app)
  - [3. Train Specialist Models](#3-train-specialist-models)
  - [4. Run Sliding-Window Inference Only](#4-run-sliding-window-inference-only)
  - [5. Run Topology Optimization Standalone](#5-run-topology-optimization-standalone)
  - [6. 1-Click End-to-End Master Resolution](#6-1-click-end-to-end-master-resolution)

---

## 💡 Core Principles & Decoupled Neural Fields

Rather than predicting fragile 1-voxel binary masks that coalesce touching fibers, this system decouples fiber segmentation into two specialist neural fields:

1. **3D Continuous Radial Intensity Potential Field $I(\vec{x}) \in [-1.0, 1.0]$**:
   - Peaks at $+1.0$ on the exact 1-voxel mathematical centerline $\mathcal{C}$.
   - Decays smoothly with Gaussian profile $G(\vec{x}) = \exp(-d(\vec{x}, \mathcal{C})^2 / 2\sigma^2)$.
   - Dips to negative values ($-1.0$) at multi-fiber crossing interfaces.
2. **3D Continuous Unit Tangent Field $\vec{O}(\vec{x}) = (V_z, V_y, V_x)$ ($\|\vec{O}\| = 1.0$)**:
   - Predicts the unit orientation tangent vector at every foreground voxel.

---

## 🧠 Framework Architecture

```
                    Input 3D Volume Patch (1 x 64 x 64 x 64)
                                       │
                ┌──────────────────────┴──────────────────────┐
                ▼                                             ▼
   [Intensity Specialist U-Net]                [Orientation Specialist U-Net]
        (IntensityUNet3D)                           (OrientationUNet3D)
                │                                             │
      Tanh() Potential Head                      L2-Normalized Tangent Head
                │                                             │
                ▼                                             ▼
   Intensity Field I(x) in [-1, 1]              Orientation Field O(x) in R^3
                │                                             │
                └──────────────────────┬──────────────────────┘
                                       │
                                       ▼
                     [Topology Optimization Engine]
                                       │
    ┌──────────────────────────────────┴──────────────────────────────────┐
    ▼                                                                     ▼
1. 3D Medial Axis Thinning & Spur Pruning             2. Transverse H-Severing (perp > 0.50)
    ▼                                                                     ▼
3. Fragment Graph & Durable Endpoints                 4. Post-H-Sever Bridge Candidate Search
    ▼                                                                     ▼
5. Min-Cost Global Matching (Degree=1)                6. Multi-Label Voronoi Diffusion
                                       │
                                       ▼
                       Final 3D Labeled Fiber Instances
```

---

## 📐 Topology Optimization Formulation

A good fiber satisfies:
- **Maximized continuous length**: $-\lambda_\ell \cdot N$
- **Low curvature**: $+\lambda_\kappa \cdot \bar{\kappa}$
- **No sharp bends**: flat penalty for any junction kink $> 55^\circ$
- **No branching**: every fragment endpoint connects to at most ONE partner (degree $\le 1$)

### Objective Function:
$$\min_{\mathcal{B}} \sum_{b \in \mathcal{B}} \left( w_g \cdot d_{\text{gap}}(b) + w_\theta \arccos|\vec{O}_A \cdot \vec{O}_B| + \mathcal{P}_{\text{durable}}(b) \right)$$

Subject to:
1. **Degree Constraint**: $\operatorname{deg}(v) \le 1$ for all fragment endpoints $v$.
2. **No-Cycle Constraint**: Graph contains no closed loops.

---

## 🔬 Why H-Junctions Occur and How They Are Resolved

### The Topological Cause (Thinning of 3D Crossing Volumes):
In continuous 3D space, ground-truth centerlines are simple, non-branching 1D curves. However, fibers have finite physical thickness. When two fibers cross or touch ($d < r_1 + r_2$), their voxelized cylinders merge into a 3D convex intersection solid (diamond). During discrete 3D thinning (e.g. Lee-Kashyap skeletonization on cubic grids), a 4-way ($X$) intersection point is topologically unstable. Thinning naturally collapses the diamond into **two 3-way ($Y$) branching nodes joined by a short transverse $H$-rung**.

### Prevention via Smart Separable Stamping:
In the data augmentation pipeline ([`core/dataset.py`](file:///C:/Users/Kataevskiy/Desktop/fiber_resolution_pipeline/core/dataset.py)), we enforce a **Centerline Separability Clearance Criterion**:
- A candidate fiber stamp is accepted **only if its centerline maintains $\ge 6.0\text{ voxels}$ distance** from all existing fiber centerlines.
- **Why 6 voxels?** With a $\sigma = 1.0$ Gaussian centerline profile ($I = +1.0$) and a $\sigma_{\text{cross}} = 1.5$ negative intersection dip ($I = -0.5$), a 6-voxel separation guarantees that the two positive centerline peaks remain distinct and separated by a negative energy valley, preventing synthetic $H$-junction formation.

### Resolution in Topology Optimization:
1. **Orientation-Decoupled Severing** ([`topology_optimizer/step1_sever_h_junctions.py`](file:///C:/Users/Kataevskiy/Desktop/fiber_resolution_pipeline/topology_optimizer/step1_sever_h_junctions.py)):
   If a short branch ($L \le 14\text{ vx}$) has a geometric direction perpendicular ($> 60^\circ$) to its adjacent fiber trunks:
   $$\text{Perpendicularity} = 1 - |\vec{D}_{\text{rung}} \cdot \vec{O}_{\text{trunk}}| > 0.50 \implies \text{SEVER}$$
2. **Durable Endpoint Averaging** ([`topology_optimizer/step3_build_fragment_graph.py`](file:///C:/Users/Kataevskiy/Desktop/fiber_resolution_pipeline/topology_optimizer/step3_build_fragment_graph.py)):
   Instead of using distorted skeleton voxels at the cut interface, we average $\vec{O}(\vec{x})$ over the **nearest 5 voxels back into the fragment body**.
3. **Collinear Continuation & Matching** ([`topology_optimizer/step2_bridge_gaps.py`](file:///C:/Users/Kataevskiy/Desktop/fiber_resolution_pipeline/topology_optimizer/step2_bridge_gaps.py), [`topology_optimizer/step4_optimize_topology.py`](file:///C:/Users/Kataevskiy/Desktop/fiber_resolution_pipeline/topology_optimizer/step4_optimize_topology.py)):
   Candidate pairs across the 1–2 voxel intersection gap with high mutual collinearity ($|\vec{O}_A \cdot \vec{O}_B| \ge 0.65$) are stitched straight through under degree $\le 1$ constraints.

---

## 🛠️ Interactive 3D Curation Web App & Engine

A WebGL-powered 3D annotation and geodesic solving suite located in `curation_tool/`:
- **3D WebGL Bounding Box & Orbit Controls** (Three.js).
- **Automated Geodesic Lane-Guided Routing** across 6 faces of the subvolume.
- **2D Slice Paintbrush & Label Editor** (XY, XZ, YZ, MIP).
- **Multi-Target Supervised Patch Export** (`.vol`, `.centerline`, `.instance`, `.intensity`, `.ori`).

---

## 📁 Directory Structure

```
fiber_resolution_pipeline/
├── README.md                           # Master documentation
├── requirements.txt                    # Python package dependencies
├── prepare_datasets.py                 # Precomputes memory-mapped NPY datasets & signed targets
├── run_end_to_end.py                   # 1-Click Master Runner (Inference + Optimization)
│
├── core/
│   ├── models.py                       # IntensityUNet3D & OrientationUNet3D architectures
│   ├── losses.py                       # Foreground-Weighted MSE, Dice & Cosine losses
│   └── dataset.py                      # 3D Morphological CT Augmentations & Separable Stamping (>= 6 vx)
│
├── topology_optimizer/
│   ├── cost_functions.py               # Modular fiber quality & bridging cost definitions
│   ├── step1_sever_h_junctions.py      # Orientation-decoupled H-severing engine
│   ├── step2_bridge_gaps.py            # Direction-durable multi-probe gap bridging
│   ├── step3_build_fragment_graph.py   # Vectorized fragment graph & durable endpoints
│   ├── step4_optimize_topology.py      # Min-cost priority matching with degree & cycle constraints
│   ├── step5_diffuse_labels.py         # Multi-label Voronoi diffusion to full fiber thickness
│   ├── evaluate_against_gt.py          # Ground-truth GAD evaluation metrics
│   └── visualize_results.py            # Diagnostic slices, histograms, and metric summaries
│
├── curation_tool/
│   ├── app.py                          # WebGL Three.js 3D annotation web application
│   ├── engine.py                       # Geodesic lane-guided centerline routing engine
│   └── test_engine.py                  # Curation engine validation tests
│
├── training/
│   ├── train_intensity.py              # Trainer for Intensity Specialist
│   ├── train_orientation.py            # Trainer for Orientation Specialist
│   └── train_both.py                   # Master sequential/joint trainer
│
└── inference/
    ├── run_inference.py                # 3D Gaussian sliding-window inference + NPY export
    └── run_topology_optimization.py    # Standalone CLI for running topology optimizer
```

---

## 🚀 Quick Start & Usage Guide

### 1. Dataset Precomputation
Precomputes analytical ground-truth orientation and signed probability fields from GAD geometry models into memory-mapped NPY format:

```bash
python prepare_datasets.py
```

---

### 2. Interactive 3D Curation Web App
Start the WebGL Three.js annotation server to inspect volumes, define seed waypoints, and curate real microscopy training blocks:

```bash
python curation_tool/app.py
```
Open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your web browser.

---

### 3. Train Specialist Models

```bash
# Train on ONLY real curated patches (no synthetic data)
python training/train_both.py \
    --real-only \
    --train-on-all-data \
    --pretrained \
    --epochs 50 \
    --patch-size 64 \
    --batch-size 4

# Train both specialist models sequentially with smart separable fiber stamping
python training/train_both.py \
    --epochs 20 \
    --patch-size 64 \
    --batch-size 2 \
    --grad-accum-steps 2 \
    --real-stamp-prob 0.25

# Train Intensity Specialist ONLY (e.g. real only)
python training/train_intensity.py \
    --real-only \
    --train-on-all-data \
    --epochs 25 \
    --patch-size 64 \
    --batch-size 4

# Train Orientation Specialist ONLY
python training/train_orientation.py \
    --real-only \
    --train-on-all-data \
    --epochs 25 \
    --patch-size 64 \
    --batch-size 4
```

---

### 4. Run Sliding-Window Inference Only
Generates continuous potential and tangent fields and saves them as `.npy` and `.tif`:

```bash
# Run both Intensity & Orientation specialists (sequential with RAM cleanup)
python inference/run_inference.py \
    --input process_data/COLLAGENCROP_003_0000.tif \
    --mode both \
    --batch-size 4 \
    --output-prefix outputs/dual_collagen

# Run Intensity Specialist ONLY
python inference/run_inference.py \
    --input process_data/COLLAGENCROP_003_0000.tif \
    --mode intensity \
    --output-prefix outputs/intensity_only

# Run Orientation Specialist ONLY
python inference/run_inference.py \
    --input process_data/COLLAGENCROP_003_0000.tif \
    --mode orientation \
    --output-prefix outputs/orientation_only
```

---

### 5. Run Topology Optimization Standalone
Optimizes topology directly from precomputed intensity and orientation fields using **direct whole-volume degree-1 linear assignment matching** (memory-mapped, zero false merges):

```bash
# Direct Global Optimization (degree <= 1, zero false merges)
python inference/run_topology_optimization.py \
    --intensity outputs/dual_collagen_intensity.npy \
    --orientation outputs/dual_collagen_orientation.npy \
    --volume outputs/dual_collagen_volume.npy \
    --out outputs/topology_resolved_full

# Optional: Chunked mode with overlap consensus
python inference/run_topology_optimization.py \
    --intensity outputs/dual_collagen_intensity.npy \
    --orientation outputs/dual_collagen_orientation.npy \
    --volume outputs/dual_collagen_volume.npy \
    --mode chunked \
    --chunk-size 512 \
    --overlap 256 \
    --out outputs/topology_resolved_full
```

---

### 6. 1-Click End-to-End Master Resolution
Runs GPU neural inference on raw microscopy and executes full topology optimization in a single command:

```bash
python run_end_to_end.py \
    --input process_data/COLLAGENCROP_003_0000.tif \
    --out outputs/collagen_resolved_final
```
