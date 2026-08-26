# 3D Fiber Resolution, Dual Neural Field & Topology Optimization Framework

A complete deep learning and geometric graph optimization framework for high-resolution 3D fiber microstructure segmentation, centerline extraction, transverse $H$-junction severing, and crossing resolution in dense biomaterials and micro-CT tomography volumes (collagen networks, fibrous composites, Altendorf-Jeulin stochastic models).

---

## 📑 Table of Contents

- [Core Principles & Decoupled Neural Fields](#core-principles--decoupled-neural-fields)
- [On-The-Fly Dynamic Spline Morphing & Augmentation Engine](#on-the-fly-dynamic-spline-morphing--augmentation-engine)
- [Framework Architecture](#framework-architecture)
- [Topology Optimization Formulation](#topology-optimization-formulation)
- [Why H-Junctions Occur and How They Are Resolved](#why-h-junctions-occur-and-how-they-are-resolved)
- [Border Margin Trimming & Boundary Hairpin Elimination (`--cut-border`)](#border-margin-trimming--boundary-hairpin-elimination---cut-border)
- [Interactive 3D Curation Web App & Engine](#interactive-3d-curation-web-app--engine)
- [Export Formats (Memory-Mapped NPY & 16-bit TIFF)](#export-formats-memory-mapped-npy--16-bit-tiff)
- [Directory Structure](#directory-structure)
- [Quick Start & Usage Guide](#quick-start--usage-guide)
  - [1. Dataset Precomputation](#1-dataset-precomputation)
  - [2. Interactive 3D Curation Web App](#2-interactive-3d-curation-web-app)
  - [3. Train Specialist Models (On-The-Fly Morphing)](#3-train-specialist-models-on-the-fly-morphing)
  - [4. Run Sliding-Window Inference Only](#4-run-sliding-window-inference-only)
  - [5. Run Topology Optimization Standalone](#5-run-topology-optimization-standalone)
  - [6. 1-Click End-to-End Master Resolution](#6-1-click-end-to-end-master-resolution)

---

## 💡 Core Principles & Decoupled Neural Fields

Rather than predicting fragile 1-voxel binary masks that coalesce touching fibers, this system decouples fiber segmentation into two specialist neural fields:

1. **3D Continuous Radial Intensity Potential Field $I(\vec{x}) \in [-1.0, 1.0]$**:
   - Peaks at $+1.0$ on the exact 1-voxel mathematical centerline $\mathcal{C}$.
   - Decays smoothly with Gaussian profile $G(\vec{x}) = \exp(-d(\vec{x}, \mathcal{C})^2 / 2\sigma^2)$.
   - Dips to negative values ($-1.0$) at multi-fiber crossing interfaces to force topological disconnectivity.
2. **3D Continuous Unit Tangent Field $\vec{O}(\vec{x}) = (V_z, V_y, V_x)$ ($\|\vec{O}\| = 1.0$)**:
   - Predicts the unit orientation tangent vector at every foreground voxel.

---

## 🧬 On-The-Fly Dynamic Spline Morphing & Augmentation Engine

To eliminate the synthetic-to-real domain gap without requiring thousands of manually labeled voxels, the pipeline dynamically deforms real biological donor fibers onto continuous 3D mathematical splines on-the-fly:

```
[1. Sample 96³ Spline Geometry from GAD Bank (5,120 Splines)]
                │
                ▼
[2. Apply 3D Continuous Augmentations FIRST]
   ├── Spatial 3D Jitter / Translation (±2.5 vx)
   ├── Sinusoidal Micro-Crimp / Wobble along Arc Length
   ├── 8 Mirror Symmetries (Z, Y, X flips at p=0.5)
   └── 48 Orthogonal Rotations
                │
                ▼
[3. Morph Real Biological Sleeves (237 Real Fibers)]
   └── Backward-warps continuous cross-sectional biological sleeves onto augmented splines
                │
                ▼
[4. Synthesize Exact Analytical Ground Truth]
   ├── Continuous Gaussian Centerline Potential I(x) in [-1.0, 1.0]
   └── Unit Tangent Orientation Field O(x) in R³
                │
                ▼
[5. Dynamic Multi-Block Pool (10 Fresh Blocks / Epoch + 28 Real Curated Blocks)]
   ├── Continuous Uniform 3D Sampling across the entire 96³ domain ([0 .. 32]³)
   └── Ultra-High Throughput (> 125 crops/sec, 0 ms dataloader stall)
```

### Key Technical Innovations:
- **Sub-Voxel Centroid Auto-Centering (`_center_curve_to_mask`)**: Eliminates discrete Dijkstra corner-cutting drift across all 14 curated real patches (reduced from $1.26\text{ vx} \to 0.66\text{ vx}$ noise floor).
- **Continuous Segment Projection (`project_points_onto_curve_continuous`)**: Fully vectorized NumPy projection onto continuous 3D segments rather than discrete integer nodes.
- **Geodesic Candidate Selection (`get_candidate_voxels_fast`)**: Spherical geodesic dilation along 1D paths delivers a **200x speedup** for 3D deformation.

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
                                       ▼
                    1. 3D Thinning (Full Boundary Context) 
                                       │
                                       ▼               
                    2. Context-Preserving Margin Cut (--cut-border)
                                       │
                                       ▼   
                    3. Transverse H-Severing (perp > 0.50)
                                       │
                                       ▼        
                    4. Fragment Graph & Durable Endpoints
                                       │
                                       ▼  
                    5. Direction-Durable Gap Search      
                                       │
                                       ▼                   
                    6. Min-Cost Global Matching (Degree=1)
                                       │
                                       ▼  
                    7. Multi-Label Voronoi Diffusion
                                       │
                                       ▼                        
                    8. Dual Export (.npy + uint16 .tif)
                                       │
                                       ▼
                    Final 3D Labeled Fiber Instances & Centerlines
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
In the data augmentation pipeline ([`core/dataset.py`](core/dataset.py)), we enforce a **Centerline Separability Clearance Criterion**:
- A candidate fiber stamp is accepted **only if its centerline maintains $\ge 6.0\text{ voxels}$ distance** from all existing fiber centerlines.
- **Why 6 voxels?** With a $\sigma = 1.0$ Gaussian centerline profile ($I = +1.0$) and a $\sigma_{\text{cross}} = 1.5$ negative intersection dip ($I = -0.5$), a 6-voxel separation guarantees that the two positive centerline peaks remain distinct and separated by a negative energy valley, preventing synthetic $H$-junction formation.

### Resolution in Topology Optimization:
1. **Orientation-Decoupled Severing** ([`topology_optimizer/step1_sever_h_junctions.py`](topology_optimizer/step1_sever_h_junctions.py)):
   If a short branch ($L \le 14\text{ vx}$) has a geometric direction perpendicular ($> 60^\circ$) to its adjacent fiber trunks:
   $$\text{Perpendicularity} = 1 - |\vec{D}_{\text{rung}} \cdot \vec{O}_{\text{trunk}}| > 0.50 \implies \text{SEVER}$$
2. **Durable Endpoint Averaging** ([`topology_optimizer/step3_build_fragment_graph.py`](topology_optimizer/step3_build_fragment_graph.py)):
   Instead of using distorted skeleton voxels at the cut interface, we average $\vec{O}(\vec{x})$ over the **nearest 5 voxels back into the fragment body**.
3. **Collinear Continuation & Matching** ([`topology_optimizer/step2_bridge_gaps.py`](topology_optimizer/step2_bridge_gaps.py), [`topology_optimizer/step4_optimize_topology.py`](topology_optimizer/step4_optimize_topology.py)):
   Candidate pairs across the 1–2 voxel intersection gap with high mutual collinearity ($|\vec{O}_A \cdot \vec{O}_B| \ge 0.65$) are stitched straight through under degree $\le 1$ constraints.

---

## ✂️ Border Margin Trimming & Boundary Hairpin Elimination (`--cut-border`)

### Why Boundary Hairpins Occur:
In 3D sliding-window neural network inference, the outermost margins of the volume ($M \approx 16\text{–}32\text{ voxels}$) can suffer from edge padding artifacts and boundary blur. When two parallel biological fibers travel side-by-side toward a volume face ($Y=0, X=0, Z=0$), boundary merge artifacts can connect their endpoints into an artificial **180° U-turn loop (hairpin)**.

If topology optimization is run on the uncropped domain:
1. The global optimizer traces through this outer boundary loop and unifies both parallel fibers into a **single continuous chain**.
2. Multi-label Voronoi diffusion then floods both fiber tracks across the entire volume with the same instance label ID.
3. Cropping the volume *after* diffusion removes the outer loop apex, but leaves the two parallel tracks fused under the same label.

### The Context-Preserving Pre-Cut Solution:
When `--cut-border <M>` (e.g. `--cut-border 32`) is passed:
1. **Stage 1 (Centerline Thinning)** runs on the **full volume with full 3D boundary context**, ensuring mathematical centerlines are accurately centered up to the cut plane.
2. **Pre-Optimization Slicing**: The binary skeleton and memory-mapped input volumes are trimmed by $M$ voxels (`skel = skel[M:D-M, M:H-M, M:W-M]`) **before** Stage 2 (H-severing) and Stage 3 (Fragment Graph Construction).
3. The outer boundary loops are **completely eliminated before any graph nodes or bridge candidates are created**.
4. The two incoming strands become **two distinct, independent endpoints** that are resolved as separate straight fibers (e.g. splitting a false 354-vx hairpin into two clean 176-vx tracks).

```
Full Volume (564³) [Full 3D Context]
      │
      ▼
Stage 1: Medial Axis Thinning & Spur Pruning
      │
      ▼
✂️ Pre-Optimization Border Cut: skel[32:532, 32:532, 32:532] -> (500³)
   └── Slices outer loops -> Turns merged hairpins into clean independent endpoints
      │
      ▼
Stages 2–7: H-Severing, Fragment Graph, Topology Matching, Voronoi Diffusion on clean 500³
```

---

## 🛠️ Interactive 3D Curation Web App & Engine

A high-performance WebGL-powered 3D annotation and geodesic solving suite located in `curation_tool/`:
- **🖥️ Direct PC File Picker & Segmented Instance TIFF Loader**: Open any raw volume (`.tif`, `.npy`) or segmented instance output (`outputs/topology_resolved_morpho/instance_volume.tif`) directly using your native Windows File Explorer dialog or browser file selector.
- **⚡ Uncapped 3D Skeletonization & Centerline Extraction**: Automatically runs 3D skeletonization across all segmented fiber labels in the 96³ patch (resolving 40+ continuous fibers with $\ge 8\text{ voxels}$), extracts boundary/internal endpoints and intermediate waypoints, and pre-populates interactive fiber seeds.
- **🛠️ Interactive Wiring Correction & Geodesic Lane Solver**: Inspect predicted fiber paths in 3D WebGL (Three.js), easily fix false mergers, disconnect bad bridges, move waypoints, add missing seeds, and re-solve continuous geodesic paths (`Space`).
- **Always-Active Visual Crop ROI**: Real-time volume slicing with 1-voxel slider & mouse-wheel precision (`step=1`) without distracting wireframe borders.
- **💾 1-Click Ground Truth Patch Export**: Press `Enter` to export curated 96³ samples (`.vol`, `.instance`, `.centerline`, `.intensity`, `.ori`, `_meta.json`) directly into `real_train_data/curated_patches/` for retraining specialist models.

---

## 💾 Export Formats (Memory-Mapped NPY & 16-bit TIFF)

The pipeline automatically exports all instance segmentation volumes and centerline skeletons in two complementary formats:

| File | Format | Description |
| :--- | :--- | :--- |
| `instance_volume.tif` | `uint16` Compressed TIFF | 3D instance volume ready for **ImageJ / Fiji**, **napari**, **Dragonfly**, and **3D Slicer**. Supports streaming BigTIFF (>1 GB) with 0 GB RAM overhead. |
| `instance_skeleton.tif` | `uint16` Compressed TIFF | 3D labeled centerline skeleton where each voxel value equals its fiber instance ID. |
| `instance_volume.npy` | `int32` Memory-Mapped NPY | Fast zero-copy memory-mapped array for downstream Python scientific processing. |
| `instance_skeleton.npy` | `int32` Memory-Mapped NPY | Memory-mapped centerline array. |
| `slices/` | `.png` Slice Previews | Multi-slice diagnostic comparisons (Raw Volume, Potential Field, Final Instances). |
| `fiber_length_histogram.png` | `.png` Distribution Plot | Histogram of resolved continuous fiber lengths. |

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
│   ├── dataset.py                      # OnTheFlyMorphedDataset, GADSplineBank & Full-Volume Sampling
│   └── fiber_morpher.py                # Vectorized backward-warping & sub-voxel continuous projection
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
│   ├── engine.py                       # Geodesic lane-guided centerline routing engine with auto-centering
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

### 2. Interactive 3D Fiber Curator Web App
Start the WebGL Three.js annotation server to inspect volumes, load segmented instances, fix wiring, and curate real training samples:

```bash
python curation_tool/app.py
```
Open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your web browser:
1. Click **`📁 Upload Volume`** in the top navbar to select any `.tif`, `.tiff`, or `.npy` file from your PC (or simply drag-and-drop the file directly onto the browser window, or click **`📂 Presets`**).
2. Select any segmented TIFF (e.g. `outputs/topology_resolved_morpho/instance_volume.tif`) or raw microscopy volume.
3. The engine automatically runs 3D skeletonization across all segmented fibers in the 96³ cube, extracts endpoints & waypoints, and renders all 40+ 3D fiber tracks.
4. **Orient with 3D Coordinate Arrows**: Use the 3D scene coordinate arrows anchored directly next to the cube origin (Red: +X width, Green: +Y height, Cyan: +Z depth) that orbit with natural 3D depth and perspective.
5. **Fix wiring**: Select a fiber ID, adjust or add/delete waypoints, split false mergers, or reconnect broken fibers.
6. Press **`Space`** to re-resolve geodesic continuous paths and inspect updated 3D centerlines.
7. Press **`Enter`** to save the curated 96³ patch directly into `real_train_data/curated_patches/`.

---

### 3. Train Specialist Models (On-The-Fly Morphing)

```bash
# Recommended: Train both specialists with 10-block/epoch dynamic biological morphing
python training/train_both.py \
    --morph-on-the-fly \
    --epochs 50 \
    --patch-size 64 \
    --samples-per-epoch 200 \
    --batch-size 4

# Train Intensity Specialist ONLY on-the-fly:
python training/train_intensity.py \
    --morph-on-the-fly \
    --epochs 50 \
    --patch-size 64 \
    --samples-per-epoch 250 \
    --batch-size 4

# Train Orientation Specialist ONLY on-the-fly:
python training/train_orientation.py \
    --morph-on-the-fly \
    --epochs 50 \
    --patch-size 64 \
    --samples-per-epoch 250 \
    --batch-size 4

# Train on ONLY real curated patches (no synthetic data)
python training/train_both.py \
    --real-only \
    --train-on-all-data \
    --pretrained \
    --epochs 50 \
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
# Recommended: Direct Global Optimization with 32px boundary loop severing
python inference/run_topology_optimization.py \
    --intensity outputs/dual_collagen_intensity.npy \
    --orientation outputs/dual_collagen_orientation.npy \
    --volume outputs/dual_collagen_volume.npy \
    --cut-border 32 \
    --out outputs/topology_resolved_full

# Optional: Chunked mode with overlap consensus for multi-gigavoxel volumes
python inference/run_topology_optimization.py \
    --intensity outputs/dual_collagen_intensity.npy \
    --orientation outputs/dual_collagen_orientation.npy \
    --volume outputs/dual_collagen_volume.npy \
    --cut-border 32 \
    --mode chunked \
    --chunk-size 512 \
    --overlap 256 \
    --out outputs/topology_resolved_full
```

#### Output Artifacts:
- `instance_volume.tif`: 16-bit compressed TIFF stack of segmented 3D fiber instances.
- `instance_skeleton.tif`: 16-bit compressed TIFF stack of labeled 1-voxel mathematical centerlines.
- `instance_volume.npy` & `instance_skeleton.npy`: Memory-mapped int32 NumPy volumes for rapid downstream analysis.
- `slices/`: Diagnostic slice montages comparing raw input, potential field, and instance color maps.
- `metrics.txt` & `fiber_length_histogram.png`: Global quantitative metrics and fiber length distribution.

---

### 6. 1-Click End-to-End Master Resolution
Runs GPU neural inference on raw microscopy and executes full topology optimization in a single command:

```bash
python run_end_to_end.py \
    --input process_data/COLLAGENCROP_003_0000.tif \
    --cut-border 32 \
    --out outputs/collagen_resolved_final
```
