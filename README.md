# Multimodal Graph Neural Network for ccRCC Survival Prediction

> Master's thesis project — *Pipeline optimization of multimodal systems (model updates and data fusion strategies)*
> Predicting 12-month survival in Clear Cell Renal Cell Carcinoma (ccRCC) by combining Whole Slide Images, clinical data, and genomic markers.
>
> Ariel University — Faculty of Engineering, Department of Industrial Engineering and Management
> Author: Yona Elbar — Supervisor: Prof. Chen Hajaj — November 2025

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Motivation](#2-motivation)
3. [Dataset](#3-dataset)
4. [Methodology & Pipeline](#4-methodology--pipeline)
5. [Models](#5-models)
6. [Results](#6-results)
7. [Exploratory Directions Not Reported in the Thesis](#7-exploratory-directions-not-reported-in-the-thesis)
8. [Repository Structure](#8-repository-structure)
9. [Installation & Usage](#9-installation--usage)
10. [Requirements](#10-requirements)
11. [Reproducibility](#11-reproducibility)
12. [Citation / Contact](#12-citation--contact)

---

## 1. Project Overview

This project addresses the binary classification task of **12-month survival prediction** in **Clear Cell Renal Cell Carcinoma (ccRCC)**, the most common subtype of kidney cancer (~75% of all renal tumors).

The work is organized around two research hypotheses:

1. **Metric learning on WSI embeddings** (triplet loss and supervised contrastive learning) improves class separability in latent space, and therefore improves prediction over the unoptimized ResNet-50 baseline.
2. **Structured multimodal fusion** based on Graph Neural Networks outperforms unimodal models, demonstrating a measurable gain from integrating heterogeneous sources.

Three complementary modalities are used:

- **Histopathological images** — Whole Slide Images (WSI), encoded as 2048-dim ResNet-50 embeddings, one vector per slide
- **Clinical data** — 11 variables covering demographics, tumor stage and grade, and cancer history
- **Genomic markers** — somatic mutations in *VHL*, *PBRM1*, *TTN*

Fusion is performed inside a **heterogeneous Graph Neural Network** where **slide nodes** are connected to **patient nodes** carrying the clinical and genomic vector. Three GNN architectures are benchmarked: **GraphSAGE**, **RGCN**, and **GAT**.

**Best result:** GraphSAGE reaches **BACC = 0.8029** and **ROC-AUC = 0.8483** on the held-out test set, compared to **0.6799** for the best unimodal (WSI-only) model and **0.53** for the WSI-only baseline reported by Mota et al. (2024).

---

## 2. Motivation

### Why ccRCC?

Clear cell renal cell carcinoma represents approximately 75% of all renal tumors and shows strong inter-individual variability. Accurate prediction of survival within the first 12 months after detection is critical: it allows clinicians to stratify patients by mortality risk, intensify protocols for high-risk cases, and avoid unnecessary procedures for low-risk patients.

### Why Multimodal?

Tumor biology is heterogeneous, and no single data source captures a complete view of the disease:

| Modality | What it captures |
|---|---|
| Histopathology (WSI) | Micro-architectural patterns associated with tumor aggressiveness |
| Clinical features | Patient demographics, tumor stage and grade, cancer history |
| Genomic markers | Driver mutations linked to recurrence and progression risk |

Combining them is not trivial: uncontrolled fusion leads to information overload, redundancy, and reduced generalizability. This is the problem the fusion strategies here are designed to address.

### Why Graph Neural Networks?

Classical early fusion (concatenation) struggles with misaligned feature distributions and large dimensionality gaps between modalities; late fusion fails to exploit cross-modal interactions. A graph formulation instead makes the relationships explicit:

- Each patient has a variable number of slides — a structure that a fixed-size concatenation cannot represent naturally
- Message passing lets image-derived evidence flow into the patient representation while clinical and genomic features are preserved on the patient node
- No explicit concatenation layer is required; fusion emerges from the graph structure itself

---

## 3. Dataset

### Source

Data is drawn from **TCGA-KIRC** (The Cancer Genome Atlas — Kidney Renal Clear Cell Carcinoma) and **CPTAC-CCRCC** (Clinical Proteomic Tumor Analysis Consortium), accessed through the GDC Data Portal and the Cancer Imaging Archive. Patients were matched across sources using unique patient identifiers, with consistency checks to avoid labeling errors.

| Dataset | Access |
|---|---|
| **TCGA-KIRC** | [GDC Data Portal](https://portal.gdc.cancer.gov/projects/TCGA-KIRC) |
| **CPTAC-CCRCC** | [Cancer Imaging Archive](https://www.cancerimagingarchive.net/collection/cptac-ccrcc/) |

> **Note:** Access to controlled-access TCGA/CPTAC data requires registration on the GDC portal and acceptance of the data access agreement.

### Cohort

617 patients were retained: **428 from TCGA** and **189 from CPTAC-ccRCC**. Only WSI was used as the imaging modality, as it was the only imaging source without missing values.

| Split | Patients | Survived ≥ 12 mo. (class 1) | Deceased < 12 mo. (class 0) |
|---|---|---|---|
| Train | 472 | 416 | 56 |
| Validation | 25 | 22 | 3 |
| Test | 120 | 104 | 16 |
| **Total** | **617** | **542** | **75** |

Class imbalance is approximately **7:1** in favour of survivors. It is addressed through weighted cross-entropy, focal loss, oversampling, weighted sampling, and by using **balanced accuracy (BACC)** as the primary metric rather than raw accuracy.

Split quality was verified statistically: ANOVA for continuous variables (age: F = 0.86, p = 0.426) and chi-square for categorical ones (AJCC metastasis: χ² = 10.79, p = 0.095; race: χ² = 0.36, p = 0.834). Tumor grade showed a significant difference (χ² = 20.27, p = 0.009), attributable to the rarity of high grades in the baseline distribution.

### Modality availability

| Split | Patients | WSI | Clinical | Genomic |
|---|---|---|---|---|
| Train | 472 | 472 | 472 | 336 |
| Validation | 25 | 25 | 25 | 25 |
| Test | 120 | 120 | 120 | 101 |
| **Total** | **617** | **617** | **617** | **462** |

Genomic data is available for **74%** of patients. Missing genomic values are **not imputed and patients are not excluded**: missingness is encoded as `-1` and treated as a potentially informative signal in the patient profile.

### Modalities

**Whole Slide Images (WSI)**
- Format: `.svs` (Aperio), pyramidal multi-resolution
- **2,573 slides** across 617 patients, at least two slides per patient
- Each slide is reduced to a single **2048-dim** vector (see Step 1–2)

**Clinical & Genomic Data** (`medical_genomic_data/clinical+genomic_split.csv`)

| Feature group | Variables |
|---|---|
| Demographics | `gender` (binary), `age_diag` (discretized by decade), `race` (5 standard categories) |
| Tumor staging | `grade`, `ajcc_path_tumor_pt`, `ajcc_path_nodes_pn`, `ajcc_path_metastasis_pm`, `ajcc_path_tumor_stage` |
| Medical history | `cancer_history` (binary) |
| Genomic mutations | `VHL_mutation`, `PBRM1_mutation`, `TTN_mutation` (mutated / non-mutated / missing = -1) |
| Split assignment | `Split` (train / validation / test) |
| **Target** | `vital_status_12` (0 = deceased < 12 mo., 1 = survived ≥ 12 mo.) |

Clinical variables from the two cohorts were harmonized (unified names and formats). Unordered categorical variables are one-hot encoded, so that no artificial ordinal relationship is introduced between categories. Genomic markers use a compact binary representation, so that the model estimates the effect of each mutation independently.

---

## 4. Methodology & Pipeline

```
WSI (.svs)                        Clinical + Genomic (CSV)
    │                                       │
    ▼                                       │
[Step 1] Tissue segmentation +              │
         patch extraction +                 │
         ResNet-50 feature extraction       │
    │  2048-dim embedding per patch         │
    ▼                                       │
[Step 2] Max pooling over patches           │
    │  → one 2048-dim vector per SLIDE      │
    │                                       │
    ├──────────────┬────────────────┐       │
    ▼              ▼                ▼       ▼
[Step 3a]      [Step 3b]        [Step 4] Multimodal GNN fusion
SupCon         Triplet loss     Heterogeneous graph:
128-dim        256-dim            slide nodes (2048-dim ResNet)
    │              │               + patient nodes (clinical + genomic)
    ▼              ▼               → GraphSAGE / RGCN / GAT
Patient-level  Patient-level              │
MLP (WSI-only  MLP (WSI-only              ▼
 baseline)      baseline)         Binary survival prediction
```

**Important:** Steps 3a and 3b are **two independent alternatives**, both applied to the same 2048-dim slide embeddings produced by Step 2. Triplet loss is *not* applied on top of SupCon embeddings. Both are evaluated as WSI-only baselines and serve to test Hypothesis 1.

**Step 4 consumes the raw 2048-dim ResNet slide embeddings**, not the metric-learning outputs. This is deliberate: it keeps the fusion gain attributable to the graph structure rather than to the representation refinement, and allows a clean comparison between Hypothesis 1 and Hypothesis 2.

---

### Step 1 — Feature Extraction

**Notebook:** `Feature extraction & preparation/Features_Extraction.ipynb`

**Tissue detection**

| Parameter | Value |
|---|---|
| Segmentation pyramid level | Level 2 (low-resolution overview) |
| Color space | HSV |
| Hue range (tissue) | [120, 180] |
| Saturation / Value threshold | ≥ 50 |
| Morphological operations | `MORPH_CLOSE` then `MORPH_OPEN` |
| Structuring element | Elliptical kernel, 5×5 px |
| Tissue coverage method | Summed area table (integral image) |

HSV thresholding is used rather than a grayscale intensity threshold because it identifies the characteristic purple-pink coloration of H&E staining more reliably.

**Patch extraction**

| Parameter | Value |
|---|---|
| Extraction pyramid level | Level 0 (full resolution) |
| Patch size | 256 × 256 px |
| Tiling strategy | Non-overlapping sliding window |
| Minimum tissue ratio | 0.30 (30% of patch area) |

**Preprocessing & encoding**

| Parameter | Value |
|---|---|
| Resize before model input | 224 × 224 px |
| Normalization mean (RGB) | [0.485, 0.456, 0.406] (ImageNet) |
| Normalization std (RGB) | [0.229, 0.224, 0.225] (ImageNet) |
| Feature extractor | ResNet-50, pre-trained on ImageNet |
| Output layer | Global Average Pooling (fc layer removed) → 2048-dim |
| Batch size (GPU inference) | 32 patches |
| Random seed | 42 |

> **On the choice of ResNet-50.** A histopathology-specific encoder (UNI, CONCH, …) would produce richer representations. ResNet-50 was retained deliberately to replicate the extraction pipeline of Mota et al. (2024), so that any performance gain can be attributed to the methodological contributions of this work rather than to a change of backbone.

```
Input:  patient.svs  (multi-gigabyte pyramidal image)
Output: patient_features.npz  → shape (N_patches, 2048)
```

---

### Step 2 — Aggregation to Slide Level

**Notebook:** `Feature extraction & preparation/Mean_Max_Pooling.ipynb`

A single slide can generate several thousand patch embeddings. **Max pooling** is applied across all patches of a slide, yielding one 2048-dim vector summarizing the global morphological profile of that slide.

```
Input:  patient_features.npz          → (N_patches, 2048)
Output: WSI_features_MaxPooling/*.npy → (2048,)   ← used in all reported experiments
        WSI_features_MeanPooling/*.npy → (2048,)  ← computed, not used in the thesis
```

Mean-pooled features are kept in the repository for completeness but **all experiments reported in the thesis use max pooling**, in order to stay aligned with the reference pipeline.

At a later stage — when a single vector per *patient* is required by the MLP classifiers of Steps 3a and 3b — a second max pooling is applied across all slides of a patient.

---

### Step 3a — Supervised Contrastive Learning

**Notebook:** `Feature extraction & preparation/Constractive_learning_Cleaned.ipynb`

Supervised contrastive learning reorganizes the embedding space by pulling together samples sharing a clinical label and pushing apart samples with opposite labels.

**Projection network:** `2048 → 1024 → 512 → 128`, Batch Normalization and ReLU after the first two layers, dropout 0.3.

**Augmentations** (two independent views per input embedding):

| Augmentation | Value |
|---|---|
| Additive Gaussian noise | std = 0.031 |
| Random feature dropout | masking probability = 0.304 |
| Random scale jitter | multiplicative factor ∈ [0.819, 1.181] |

**Training configuration:**

| Hyperparameter | Value |
|---|---|
| Loss | Supervised Contrastive Loss + auxiliary classification loss |
| Temperature τ | 0.07 |
| Learning rate | 5.13 × 10⁻⁴ |
| Batch size | 128 |
| Optimizer | Adam |
| Gradient clipping | norm 5.0 |
| Epochs | 50 (search) / 51 (final, on train + validation) |
| Class imbalance | `WeightedRandomSampler`, weights inversely proportional to class frequency |
| Hyperparameter search | Random search, 100 configurations, early stopping (patience = 6) |

**Leakage control:** slides from the same patient are **never** treated as positive pairs, even when they share the same clinical label.

**Patient-level classifier:** max pooling over the patient's slide embeddings, then an MLP `128 → 512 → 256 → 128 → 64 → 1` with Batch Normalization and ReLU.

| Hyperparameter | Value |
|---|---|
| Loss | Focal Loss (α = 0.039, γ = 2.73) |
| Regularization | dropout, weight decay, Gaussian noise injected on minority-class embeddings only |
| Search | Optuna, 80 trials, MedianPruner, objective = BACC |
| Decision threshold | 0.55 (maximizing BACC on the validation set) |

---

### Step 3b — Triplet Loss

**Notebook:** `Feature extraction & preparation/Triplet_loss_model_Max.ipynb`

Triplet Margin Loss explicitly controls distances between an anchor, a positive (same label) and a negative (opposite label):

```
L(A, P, N) = max( d(f(A), f(P)) − d(f(A), f(N)) + α , 0 )
```

**Triplet construction:**
- Minority-class patients are oversampled at the patient level before triplet construction, until the number of their slide representations matches the majority class
- Anchor and positive are two different slides **from the same patient**
- Negative is a slide from a patient with the opposite clinical label

**Embedding network:** `2048 → 1024 → 512 → 256`, ReLU and dropout (p = 0.3) after the first two layers, L2-normalized output (unit hypersphere, so Euclidean distance is directly related to cosine similarity).

| Hyperparameter | Value |
|---|---|
| Margin α | 0.2 |
| Embedding dimension | 256 |
| Learning rate | 3 × 10⁻⁴ |
| Batch size | 64 |
| Dropout | 0.3 |
| Weight decay | 10⁻⁴ |
| Optimizer | Adam |
| Search | Manual grid search, 5 configurations, early stopping (patience = 4, ≤ 15 epochs/trial) |
| Final training | 20 epochs on train + validation |

**Patient-level classifier:** max pooling over the patient's slide embeddings, then an MLP.

| Hyperparameter | Value |
|---|---|
| Search | Optuna, 80 trials, TPE sampler (multivariate), MedianPruner |
| Search space | layers (2–4), hidden dims, dropout rates, lr, batch size, augmentation |
| Objective | `0.7 × BACC + 0.3 × min(precision_0, precision_1)` |
| Final architecture | `256 → 128 → 256 → 2`, dropout 0.37 / 0.54 / 0.39 |
| Loss | Cross-entropy with inverse-frequency class weighting |
| Scheduler | ReduceLROnPlateau (factor 0.5, patience 5) |
| Minority oversampling | factor 10 |

---

### Step 4 — Multimodal GNN Fusion

**Notebooks:** `Experiments/Fusion_*_Final.ipynb`

A **heterogeneous graph** is built over the whole cohort:

- **Patient nodes** (617): initialized with the encoded clinical + genomic vector
- **Slide nodes** (2,573): initialized with the 2048-dim ResNet-50 slide embedding
- **Edges:** each patient node is connected to its own slide nodes, forming a star-shaped subgraph per patient

Additional edge types are introduced by specific architectures (see [Section 5](#5-models)):

| Edge type | GraphSAGE | RGCN | GAT |
|---|---|---|---|
| WSI → Patient | ✓ | ✓ | ✓ |
| Patient → WSI | ✗ | ✓ | ✗ |
| Patient → Patient (k-NN) | ✗ | ✓ | ✗ |

Fusion requires no explicit concatenation layer: slide nodes send information to their patient node through graph convolutions, while clinical and genomic features attached to the patient node are preserved through propagation. The final patient embedding therefore encodes both image-derived and tabular information.

**Classification head:** linear layer + ReLU + dropout, then a final linear layer (`hidden_dim → 2`) producing binary survival logits. Classification is performed on patient nodes only, and the network is trained end-to-end.

---

## 5. Models

### GraphSAGE — best model

**Notebook:** `Experiments/Fusion_SAGEConv_Final.ipynb`

GraphSAGE (Hamilton et al., 2017) aggregates neighbor features with simple statistical operations. Here, each patient embedding is computed by averaging the feature vectors of its connected slide nodes. A residual connection after each convolution adds the pre-convolution patient representation back to the output, preserving patient-specific information while progressively integrating image-derived signals.

| Hyperparameter | Value |
|---|---|
| GNN layers | 2 |
| Hidden dimension | 48 |
| Dropout | 0.5 |
| Residual connections | ✓ |
| Loss | Weighted cross-entropy |
| Class weights | [9.0, 1.0] |
| Optimizer | Adam |
| Learning rate | 8.54 × 10⁻³ |
| Training epochs | 170 (train + validation) |
| Decision threshold | 0.5 (fixed) |

**Test: BACC = 0.8029 | ROC-AUC = 0.8483 | recall class 0 = 0.62 | precision class 0 = 0.83**

---

### RGCN

**Notebook:** `Experiments/Fusion_RGCNConv_Final.ipynb`

RGCN (Schlichtkrull et al., 2018) handles multiple edge types and learns separate transformation parameters per relation. It operates on a homogeneous node tensor obtained by concatenating projected image and patient representations, with three relations:

- `0`: image → patient
- `1`: patient → image (inverse)
- `2`: patient ↔ patient — a k-NN graph built on all available clinical and genomic features

The value of *k* was selected by Optuna (40 trials, 5-fold stratified cross-validation on the training set), yielding **k = 20** with a cosine-symmetric construction.

| Hyperparameter | Value |
|---|---|
| GNN layers | 2 |
| Hidden dimension | 128 |
| Dropout | 0.1 |
| Residual connections | ✗ |
| Loss | Focal Loss (γ = 5.0, α computed dynamically from the class distribution) |
| Optimizer | Adam |
| Learning rate | 5.01 × 10⁻⁴ |
| Weight decay | 3.83 × 10⁻⁶ |
| Training epochs | 170 |
| k (patient–patient k-NN) | 20 |
| Decision threshold | 0.485 |

The threshold was calibrated after training by sweeping 99 values between 0.01 and 0.99 on the combined train + validation set and retaining the value maximizing BACC. It was then applied to the test set without further adjustment.

**Test: BACC = 0.7428 | ROC-AUC = 0.8281 | recall class 0 = 0.5625 | recall class 1 = 0.9231**

---

### GAT

**Notebook:** `Experiments/Fusion_GATConv_Final.ipynb`

GAT (Veličković et al., 2018) lets each patient node learn attention coefficients over its slide nodes, weighting diagnostically relevant slides more heavily and down-weighting less informative ones. Edge dropout randomly masks a fraction of image→patient edges at each forward pass, preventing the model from relying on any fixed subset of slides. A residual connection follows each attention layer.

| Hyperparameter | Value |
|---|---|
| GNN layers | 1 |
| Hidden dimension | 64 |
| Dropout | 0.14 |
| Edge dropout | 0.016 |
| Residual connections | ✓ |
| Loss | Cross-entropy (no class weighting) |
| Optimizer | AdamW |
| Learning rate | 6.65 × 10⁻⁴ |
| Weight decay | 9.11 × 10⁻³ |
| Gradient clipping | norm 2.0 |
| Scheduler | ReduceLROnPlateau |
| Training epochs | 300 |
| Decision threshold | 0.82 |

The threshold was calibrated by sweeping 99 values between 0.01 and 0.99 across 5-fold cross-validation folds repeated over 3 seeds on the training set, retaining the median threshold. The high value reflects the model's tendency to assign high probabilities to the majority class; shifting the threshold upward forces a survival prediction only when the model is very confident, improving sensitivity to the minority class.

**Test: BACC = 0.7212 | ROC-AUC = 0.7806 | recall class 0 = 0.50 | recall class 1 = 0.94**

---

## 6. Results

### Embedding separability (before classification)

Class separability was assessed with intra-class and inter-class distances (cosine and Euclidean). The objective is inter-class distance **greater** than intra-class distance.

**Original ResNet-50 embeddings**

| Class + split | Intra cosine | Inter cosine | Intra Euclidean | Inter Euclidean |
|---|---|---|---|---|
| Train class 0 | 0.127 ± 0.043 | 0.135 ± 0.042 | 0.498 ± 0.082 | 0.514 ± 0.078 |
| Train class 1 | 0.140 ± 0.043 | — | 0.523 ± 0.081 | — |
| Test class 0 | 0.126 ± 0.040 | 0.134 ± 0.040 | 0.496 ± 0.081 | 0.512 ± 0.075 |
| Test class 1 | 0.136 ± 0.045 | — | 0.515 ± 0.086 | — |

Intra- and inter-class distances are nearly identical: the raw embeddings capture global morphological features but carry almost no signal separating survival classes. This motivates the metric-learning stage.

**After supervised contrastive learning**

| Class + split | Intra cosine | Inter cosine | Intra Euclidean | Inter Euclidean |
|---|---|---|---|---|
| Train class 0 | 0.006 ± 0.001 | 0.285 ± 0.041 | 0.101 ± 0.018 | 0.752 ± 0.084 |
| Train class 1 | 0.004 ± 0.001 | — | 0.083 ± 0.016 | — |
| Test class 0 | 0.099 ± 0.031 | 0.080 ± 0.027 | 0.360 ± 0.071 | 0.303 ± 0.065 |
| Test class 1 | 0.014 ± 0.006 | — | 0.129 ± 0.034 | — |

On the training set the reorganization is dramatic. On the **test** set, however, class 0 shows a **negative gap** (intra 0.099 > inter 0.080): embeddings of the opposite class are closer than embeddings of the same class. This reflects the small size and the histopathological heterogeneity of the minority class, and is the main limitation of this representation.

**After triplet loss**

| Class + split | Intra cosine | Inter cosine | Intra Euclidean | Inter Euclidean |
|---|---|---|---|---|
| Train class 0 | 0.649 ± 0.058 | 0.582 ± 0.067 | 0.963 ± 0.092 | 0.961 ± 0.088 |
| Train class 1 | 0.477 ± 0.051 | — | 0.838 ± 0.081 | — |
| Test class 0 | 0.489 ± 0.072 | 0.497 ± 0.064 | 0.851 ± 0.104 | 0.874 ± 0.096 |
| Test class 1 | 0.431 ± 0.063 | — | 0.769 ± 0.089 | — |

The survivor class (1) is consistently more compact than the non-survivor class (0). On the training set, class 0 does not achieve a compact intra-class structure, indicating instability in triplet formation. The test set does show the expected ordering for both classes.

### Comparative performance (test set)

| Model | Modalities | BACC | ROC-AUC |
|---|---|---|---|
| Baseline — Mota et al. (2024) | WSI only | 0.53 | n/a |
| Triplet loss + patient-level MLP | WSI only | 0.6274 | 0.59 |
| Supervised contrastive + patient-level MLP | WSI only | 0.6799 | 0.7228 |
| **GraphSAGE fusion** | **WSI + clinical + genomic** | **0.8029** | **0.8483** |
| RGCN fusion | WSI + clinical + genomic | 0.7428 | 0.8281 |
| GAT fusion | WSI + clinical + genomic | 0.7212 | 0.7806 |

> **Primary metric: Balanced Accuracy** — `BACC = 0.5 × (recall_class0 + recall_class1)` — chosen over raw accuracy because of the ~7:1 class imbalance. ROC-AUC is reported as a threshold-independent complement.

### Key observations

- **Metric learning validates Hypothesis 1.** Supervised contrastive learning lifts WSI-only BACC from the 0.53 reference to 0.6799, a 15-point gain obtained purely by restructuring the latent space before classification.
- **Fusion validates Hypothesis 2, and dominates.** GraphSAGE gains a further **+12 BACC points** over the best unimodal model (0.8029 vs 0.6799). Structured multimodal fusion contributes more than representation refinement alone.
- **Simplicity wins on small data.** The simplest architecture — GraphSAGE with a single unidirectional relation and residual connections — outperforms both the multi-relational RGCN and the attention-based GAT. With 472 training patients, the additional parameters required by three relation types or by attention weights do not receive enough gradient signal to be reliably learned.
- **Contrastive learning outperforms triplet loss.** Beyond the BACC gap (0.6799 vs 0.6274), the AUC gap is larger (0.7228 vs 0.59): the triplet model concentrates its discriminative capacity around a single operating point rather than producing well-calibrated probabilities across thresholds. Clinically, this is a meaningful limitation.
- **Minority-class recall remains the bottleneck.** Every model, unimodal or fused, achieves markedly higher recall on survivors than on non-survivors. Improving class-0 recall is the primary avenue for future work.

### Limitations

- Class imbalance affects decision thresholds and latent representation learning, despite systematic mitigation.
- The cohort (617 patients, 75 non-survivors) is small, which constrains the more complex fusion architectures and produces wide uncertainty on minority-class metrics.
- The validation set is small (25 patients, 3 non-survivors), limiting the reliability of hyperparameter and threshold selection performed on it.
- WSI representation quality is bounded by the use of a single ImageNet-pretrained backbone.

### Future work

- Replace ResNet-50 with domain-adaptive histopathology encoders (UNI, CONCH) for richer visual representations without additional labelled data
- Cross-modal attention / transformer-based fusion for finer-grained interactions than graph aggregation
- Graph-level balancing strategies and dynamic graph construction from learned similarity metrics
- Extend beyond binary 12-month survival to continuous survival modelling (Cox-based deep learning)

---

## 7. Exploratory Directions Not Reported in the Thesis

The following directions were explored during the project but are **not part of the results reported in the thesis**. They are documented here for transparency and to guide future work.

**7.1 — CLIP + LLM for WSI description.** An attempt to enrich WSI features by generating textual descriptions of patches with a CLIP-based model combined with an LLM, conditioned on survival labels. The generated descriptions added no discriminative signal over the visual embeddings, most likely because of the domain gap between natural-image CLIP representations and histopathology.

**7.2 — Vision-Language Models for WSI.** Building on medical VLMs was inconclusive: such models require large-scale supervision to fine-tune effectively, and a 617-patient cohort is insufficient to prevent overfitting.

**7.3 — Mean pooling.** Mean-pooled slide representations were computed and are retained in `WSI_features_MeanPooling/`. All experiments reported in the thesis use max pooling, in keeping with the reference pipeline of Mota et al. (2024). A systematic max-vs-mean ablation across the three GNN architectures remains open work.

---

## 8. Repository Structure

```
multimodal-ccRCC-prediction/
│
├── Feature extraction & preparation/
│   ├── Features_Extraction.ipynb            # Step 1: tissue segmentation + ResNet-50 patch features
│   ├── Mean_Max_Pooling.ipynb               # Step 2: aggregation to slide level
│   ├── Constractive_learning_Cleaned.ipynb  # Step 3a: SupCon + patient-level MLP
│   ├── Triplet_loss_model_Max.ipynb         # Step 3b: triplet loss + patient-level MLP
│   └── distribution.ipynb                   # Split validation (ANOVA / chi-square), label balance
│
├── Experiments/
│   ├── Fusion_SAGEConv_Final.ipynb          # Step 4: GraphSAGE fusion (best model)
│   ├── Fusion_RGCNConv_Final.ipynb          # Step 4: RGCN fusion with k-NN patient graph
│   └── Fusion_GATConv_Final.ipynb           # Step 4: GAT fusion
│
├── medical_genomic_data/
│   └── clinical+genomic_split.csv           # 617 patients: clinical, genomic, labels, split assignment
│
├── Presentation/                            # Thesis defence material
│
├── features/                                # Raw ResNet-50 patch embeddings (.npz), one per patient
├── WSI_features_MaxPooling/                 # Max-pooled slide embeddings (.npy) — used in all experiments
├── WSI_features_MeanPooling/                # Mean-pooled slide embeddings (.npy) — not used in the thesis
│
├── SVS original files/                      # Original WSI slides (.svs) — not tracked in git
│
├── triplet_model_final_trainval.pth         # Trained triplet embedding network
├── final_mlp_optuna_custom_score_train_only.pth
├── optuna_best_config_custom_score.json     # Best configuration (triplet patient-level MLP)
├── optuna_hyperparam_search_custom_score.csv
│
└── README.md
```

---

## 9. Installation & Usage

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (≥ 16 GB VRAM recommended for WSI processing)
- OpenSlide installed at system level

```bash
# Ubuntu/Debian
sudo apt-get install openslide-tools python3-openslide
```

### Setup

```bash
git clone https://github.com/yona1997/multimodal-ccRCC-prediction.git
cd multimodal-ccRCC-prediction

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### Data setup

Place the clinical/genomic CSV at `medical_genomic_data/clinical+genomic_split.csv`. It must contain `case_id`, `vital_status_12`, `Split`, and all clinical and genomic feature columns listed in [Section 3](#3-dataset).

Place the `.svs` slides in the folder configured as `base_folder` inside `Features_Extraction.ipynb`.

### Reproducing the pipeline

**Step 1 — Feature extraction**
```bash
jupyter nbconvert --to notebook --execute \
  "Feature extraction & preparation/Features_Extraction.ipynb"
# Output: features/<patient_id>_features.npz
```

**Step 2 — Aggregation**
```bash
jupyter nbconvert --to notebook --execute \
  "Feature extraction & preparation/Mean_Max_Pooling.ipynb"
# Output: WSI_features_MaxPooling/ and WSI_features_MeanPooling/
```

**Step 3a / 3b — Metric learning (WSI-only baselines, independent of each other)**
```bash
jupyter nbconvert --to notebook --execute \
  "Feature extraction & preparation/Constractive_learning_Cleaned.ipynb"

jupyter nbconvert --to notebook --execute \
  "Feature extraction & preparation/Triplet_loss_model_Max.ipynb"
```

**Step 4 — Multimodal GNN fusion**
```bash
# Best model
jupyter nbconvert --to notebook --execute "Experiments/Fusion_SAGEConv_Final.ipynb"

# Alternatives
jupyter nbconvert --to notebook --execute "Experiments/Fusion_RGCNConv_Final.ipynb"
jupyter nbconvert --to notebook --execute "Experiments/Fusion_GATConv_Final.ipynb"
```

> Steps 3a and 3b are not prerequisites for Step 4: the fusion models consume the 2048-dim slide embeddings produced at Step 2.

---

## 10. Requirements

```
torch>=2.0.0
torchvision>=0.15.0
torch-geometric>=2.3.0
openslide-python>=1.3.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
optuna>=3.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
opencv-python>=4.8.0
tqdm>=4.65.0
```

> **torch-geometric:** install the build matching your CUDA and PyTorch versions — see the [official installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

Neural network components (including the ResNet-50 backbone) are implemented in **PyTorch**; graph models in **PyTorch Geometric**; classical models and preprocessing in **scikit-learn**; hyperparameter optimization with **Optuna**; slide reading with **OpenSlide** and tissue segmentation with **OpenCV**.

---

## 11. Reproducibility

- All experiments use a **fixed random seed of 42**.
- The **test set was held out** and used only for the final evaluation of each model; no model was readjusted after seeing test results.
- Decision thresholds were calibrated on train + validation (RGCN) or by cross-validation on the training set (GAT), never on the test set. GraphSAGE uses a fixed threshold of 0.5.
- Hyperparameters were selected on the validation set with Optuna (MedianPruner), or by random / manual search where indicated per model.
- Final models were retrained on the combined train + validation sets before test evaluation.

---

## 12. Citation / Contact

Master's thesis submitted in partial fulfillment of the requirements for the degree of Master of Science in Industrial Engineering and Management, Ariel University.

**Title:** *Pipeline optimization of multimodal systems (model updates and data fusion strategies)*
**Author:** Yona Elbar — <yona.elbar@gmail.com>
**Supervisor:** Prof. Chen Hajaj
**Repository:** <https://github.com/yona1997/multimodal-ccRCC-prediction>

### Data sources

- TCGA-KIRC: <https://portal.gdc.cancer.gov/projects/TCGA-KIRC>
- CPTAC-CCRCC: <https://proteomics.cancer.gov/programs/cptac>

### Key references

- Mota T. et al. *MMIST-ccRCC: A Real World Medical Dataset for the Development of Multi-Modal Systems.* CVPR Workshops, 2024. — reference pipeline and WSI-only baseline
- Hamilton W. et al. *Inductive Representation Learning on Large Graphs* (GraphSAGE). NeurIPS 2017.
- Veličković P. et al. *Graph Attention Networks* (GAT). ICLR 2018.
- Schlichtkrull M. et al. *Modeling Relational Data with Graph Convolutional Networks* (RGCN). ESWC 2018.
- Khosla P. et al. *Supervised Contrastive Learning* (SupCon). NeurIPS 2020.
- Ghojogh B. et al. *Fisher Discriminant Triplet and Contrastive Losses for Training Siamese Networks.* IJCNN 2020.
- Ren C. et al. *Multimodal Graph Neural Networks for Integrative Analysis of Cancer Data.* Briefings in Bioinformatics, 2021.
- Fatemi B. et al. *Graph Neural Networks Use Graphs When They Shouldn't.* arXiv:2309.04332, 2023.
