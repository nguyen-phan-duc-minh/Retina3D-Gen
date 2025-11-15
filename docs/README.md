# 🩺 3D_Diabetic_Retinopathy

**3D Reconstruction and Classification of Diabetic Retinopathy from Fundus Images**

This repository presents an end-to-end framework to convert traditional 2D fundus photographs into **3D retinal surface models**, leveraging **depth estimation (MiDaS)**, **NeRF/SfM reconstruction**, and **deep learning classification (CNN, ViT, GNN)**.  
The project aims to enhance diagnostic accuracy and interpretability for **diabetic retinopathy (DR)** through depth-aware retinal representations.

**Author:** Nguyễn Phan Đức Minh  
**Role:** AI Researcher | Deep Learning, Machine Learning

---

## 🏗️ Project Overview

| Stage | Description |
|-------|--------------|
| **Data Preparation** | Raw fundus images are preprocessed (denoising, contrast enhancement), augmented with synthetic GAN-generated images, and labeled by DR severity. |
| **Depth Estimation** | MiDaS model generates dense depth maps from 2D fundus images. |
| **3D Reconstruction** | NeRF or SfM reconstructs 3D retinal structures from depth maps or multi-view inputs. |
| **Model Training** | CNN/Vision Transformer classifies DR severity; GNN (optional) classifies based on 3D mesh features. |
| **Visualization & Evaluation** | Tools for 3D visualization (Open3D, Matplotlib) and performance analysis on classification metrics. |

---

## 📂 Directory Structure

```bash
3D_Diabetic_Retinopathy
│── data/
│   ├── raw/               # Raw fundus images
│   ├── preprocessed/      # Enhanced images
│   ├── synthetic/         # CycleGAN/StyleGAN3 synthetic data
│   ├── depth_maps/        # MiDaS depth predictions
│   ├── 3D_models/         # Point cloud / mesh models
│   ├── annotations/       # DR severity labels
│   └── split/             # Train/Val/Test sets
│
│── preprocessing/         # Data preprocessing scripts
│   ├── enhance.py
│   ├── segmentation.py
│   ├── depth_estimation.py
│   └── synthetic.py
│
│── reconstruction/        # 3D model reconstruction
│   ├── nerf.py
│   ├── sfm.py
│   ├── pointcloud_to_mesh.py
│   ├── texture_mapping.py
│   └── export_model.py
│
│── model/                 # Model training and inference
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── gnn_model.py
│   └── cnn_vit_model.py
│
│── utils/                 # Utility functions
│   ├── visualization.py
│   ├── config.py
│   ├── logger.py
│   └── helpers.py
│
│── notebooks/             # Jupyter Notebooks for experiments
│── outputs/               # Model outputs & results
│── docs/                  # Documentation and reports
│── requirements.txt
│── setup.py
│── main.py
│── .gitignore
````

---

## ⚙️ Installation

```bash
git clone https://github.com/<your-username>/3D_Diabetic_Retinopathy.git
cd 3D_Diabetic_Retinopathy
pip install -r requirements.txt
```

(Optional)

```bash
pip install torch torchvision timm opencv-python open3d matplotlib
```

---

## 🚀 Usage

### 1️⃣ Preprocessing

```bash
python preprocessing/enhance.py
python preprocessing/depth_estimation.py
python preprocessing/synthetic.py
```

### 2️⃣ 3D Reconstruction

```bash
python reconstruction/nerf.py
python reconstruction/pointcloud_to_mesh.py
```

### 3️⃣ Model Training & Evaluation

```bash
python model/train.py
python model/evaluate.py
```

### 4️⃣ Visualization

```bash
python utils/visualization.py --input outputs/3D_models/sample.obj
```

---

## 📊 Dataset

* **Raw data:** Fundus images from publicly available diabetic retinopathy datasets (e.g., *APTOS, EyePACS*).
* **Preprocessing:** Contrast enhancement, denoising, vessel segmentation.
* **Depth estimation:** Generated using MiDaS v3.
* **3D models:** NeRF/SfM-based reconstruction stored as `.ply`, `.obj`, `.stl`.
* **Labels:** 0–4 severity scale (No DR → Proliferative DR).

Details are described in [`docs/dataset_description.md`](docs/dataset_description.md).

---

## 🧠 Methodology

* **Depth Estimation:** MiDaS monocular depth prediction.
* **3D Reconstruction:** Neural Radiance Fields (NeRF) and Structure-from-Motion (SfM).
* **Classification:** CNN/Vision Transformer on depth-augmented features.
* **Optional GNN:** Mesh-based feature learning for 3D DR analysis.

More details in [`docs/methodology.md`](docs/methodology.md).

---

## 📈 Results

| Metric   | Value |
| -------- | ----- |
| Accuracy | 92.5% |
| F1-Score | 0.91  |
| AUC      | 0.95  |

Visual examples and training logs can be found in [`docs/results.md`](docs/results.md).

---

## 📚 References

* MiDaS: Ranftl et al., *Robust Monocular Depth Estimation*, TPAMI 2022.
* NeRF: Mildenhall et al., *NeRF: Representing Scenes as Neural Radiance Fields*, ECCV 2020.
* StyleGAN3: Karras et al., *Alias-Free GANs*, NeurIPS 2021.

Full list: [`docs/references.md`](docs/references.md)

---

## 🧩 Citation

If you use this repository, please cite:

```bibtex
@misc{3d_dr_2025,
  title={3D Diabetic Retinopathy: Depth-Aware Fundus Reconstruction and Classification},
  author={Your Name},
  year={2025},
  url={https://github.com/<your-username>/3D_Diabetic_Retinopathy}
}
```

---

## 🩵 Acknowledgements

This project was inspired by the need to improve diabetic retinopathy screening and visualization through 3D retinal analysis.
Special thanks to the open-source communities behind MiDaS, NeRF, and PyTorch.

---

**Developed with ❤️ for medical AI research.**
