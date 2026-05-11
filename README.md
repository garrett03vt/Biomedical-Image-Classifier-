# MedMNIST v2 CNN Benchmark — Final Project

This repository trains, evaluates, and analyzes 2D and 3D CNN classifiers on
all 18 datasets in the MedMNIST v2 benchmark suite (Yang et al., *Scientific
Data*, 2023).

## Project structure

```
.
├── src/
│   ├── cnn.py                  # CNN2D and CNN3D architectures + training loop
│   ├── features.py             # 2D/3D dimensionality detection + HOG/PCA helpers
│   ├── noise.py                # Gaussian / salt-and-pepper / speckle noise
│   ├── utils.py                # MedMNIST data loading helpers
│   ├── train.py                # Master training script — runs everything
│   ├── train2d.py              # 2D CNN training pipeline (12 datasets)
│   ├── train3d.py              # 3D CNN training pipeline (6 datasets)
│   │
│   ├── analysis/               # Post-hoc analysis (no training, fast)
│   │   ├── analyze_results_2d.py        # AUC/accuracy/time tables for 2D
│   │   ├── analyze_results_3d.py        # AUC/accuracy/time tables for 3D
│   │   ├── bias_analysis_2d.py          # Class-imbalance plots for 2D
│   │   ├── bias_analysis_3d.py          # Class-imbalance plots for 3D
│   │   ├── overfit_diagnostics.py       # Train-vs-val loss curves + diagnoses
│   │   └── generate_report_figures.py   # Master script — all report figures
│   │
│   ├── evaluation/             # Evaluation pipelines
│   │   ├── test_evaluation.py           # Test-set eval + confusion matrices
│   │   ├── noise_evaluation_2d.py       # Per-dataset noise robustness (2D)
│   │   ├── noise_evaluation_3d.py       # Per-dataset noise robustness (3D)
│   │   └── run_all_noise_evaluations.py # Noise eval across all datasets
│   │
│   └── viewers/                # Interactive data viewers + sample renders
│       ├── 2Dviewer.py                       # 2D image grid viewer
│       ├── 3Dviewer.py                       # 3D volume slice-by-slice viewer
│       ├── visualize_noise_samples_2d.py     # Saves clean-vs-noisy PNG (2D, no model)
│       └── visualize_noise_samples_3d.py     # Saves clean-vs-noisy PNG (3D, no model)
│
├── tests/
│   └── test_pipeline.py        # Pytest suite — 26 tests, runs in ~10 s
│
├── models_2d/                  # Saved 2D training results (.joblib)
├── models_3d/                  # Saved 3D training results (.joblib)
├── report_figures/             # Generated tables and plots for the paper
├── test_results/               # Test-set evaluation outputs
├── noise_results/              # Cross-dataset noise evaluation outputs
└── README.md                   # This file
```

## Setup

```bash
# Create environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch torchvision numpy scikit-learn scikit-image \
            matplotlib pytest joblib tqdm pillow medmnist
```

GPU strongly recommended — full 2D + 3D training takes about 4–6 hours on an
RTX 4090. CPU-only training is feasible but will take 24+ hours.

## Reproducing the paper

The full pipeline runs as four steps. Each later step depends on the previous
one's outputs. If you wish to reproduce the models, you will need to delete the 
saved models in "models_2d" and "models_3d"

### 1. Sanity check (recommended first step)

```bash
pytest tests/
```

26 tests covering noise functions, CNN forward passes, gradient flow,
and the data preparation pipeline. Runs on synthetic data so no MedMNIST
download is required. Should take under 15 seconds and pass cleanly.

### 2. Train all models

```bash
# Trains every 2D and 3D dataset, saves results to models_2d/ and models_3d/
python src/train.py                                                 # ALTERNATIVE:  py -3.11 src/train.py
```

This is the long-running step. Per-dataset hyperparameters (epochs, learning
rate, augmentation, weight decay) are defined inline in `train2d.py` /
`train3d.py` and were tuned per-dataset based on training-curve diagnostics.
The script is idempotent — re-running skips datasets that already have a
saved `.joblib` result. Pass `force_retrain=True` to override.

### 3. Generate report figures

```bash
# Reads models_2d/*.joblib and models_3d/*.joblib, produces:
#   report_figures/combined_results_table.csv
#   report_figures/combined_results_table.tex 
#   report_figures/comparison_vs_medmnist.csv
#   report_figures/auc_comparison_2d.png
#   report_figures/auc_comparison_3d.png
#   report_figures/accuracy_comparison_2d.png
#   report_figures/accuracy_comparison_3d.png
#   report_figures/dataset_size_vs_auc.png
#   report_figures/task_difficulty.png
python src/analysis/generate_report_figures.py                      # ALTERNATIVE:  py -3.11 src/analysis/generate_report_figures.py

# Loss curves and overfit/underfit diagnoses across all datasets
python src/analysis/overfit_diagnostics.py                          # ALTERNATIVE:  py -3.11 src/analysis/overfit_diagnostics.py

# Per-dataset class-imbalance plots
python src/analysis/bias_analysis_2d.py                             # ALTERNATIVE:  py -3.11 -m src.analysis.bias_analysis_2d
python src/analysis/bias_analysis_3d.py                             # ALTERNATIVE:  py -3.11 -m src.analysis.bias_analysis_3d
```

### 4. Test-set evaluation + confusion matrices

```bash
# Evaluates each model on the held-out TEST split (not validation) using
# medmnist.Evaluator — produces canonical AUC/ACC numbers comparable to
# Tables 3 and 4 of Yang et al. 2023.
python src/evaluation/test_evaluation.py            # all datasets  # ALTERNATIVE:  py -3.11 src/evaluation/test_evaluation.py
python src/evaluation/test_evaluation.py --mode 2d  # 2D only       # ALTERNATIVE:  py -3.11 src/evaluation/test_evaluation.py --mode 2d
python src/evaluation/test_evaluation.py --mode 3d  # 3D only       # ALTERNATIVE:  py -3.11 src/evaluation/test_evaluation.py --mode 3d

# Outputs:
#   test_results/test_evaluation_summary.csv
#   test_results/confusion_matrices/<dataset>_confusion_matrix.png
```

This re-trains each model from scratch because the original training pipeline
saves metrics only — not weights. Budget similar wall-clock time to step 2.

### 5. Noise robustness experiments

```bash
# Runs the noise pipeline across ALL datasets and produces a single
# heatmap showing AUC degradation under each noise type.
python src/evaluation/run_all_noise_evaluations.py --epochs 10      # ALTERNATIVE:  py -3.11 src/evaluation/run_all_noise_evaluations.py --epochs 10

# Outputs:
#   noise_results/all_noise_results.csv
#   noise_results/auc_degradation_heatmap.png
```

For *one* dataset at full epochs you can use the original per-task scripts:

```bash
python src/evaluation/noise_evaluation_2d.py                        # ALTERNATIVE:  py -3.11 src/evaluation/noise_evaluation_2d.py
python src/evaluation/noise_evaluation_3d.py                        # ALTERNATIVE:  py -3.11 src/evaluation/noise_evaluation_3d.py
```

## Comparing against published baselines

The MedMNIST v2 paper (Yang et al., *Scientific Data* 10:41, 2023) publishes
ResNet-18 and ResNet-50 benchmark numbers in its Tables 3 (2D) and 4 (3D),
available at https://www.nature.com/articles/s41597-022-01721-8.

`generate_report_figures.py` produces a CSV template
(`report_figures/comparison_vs_medmnist.csv`) where the "Ours" columns are
filled in automatically. To complete the comparison, copy the
ResNet-18 (28×28 resolution) AUC and ACC numbers from Tables 3–4 of the
paper into the corresponding "MedMNIST ResNet-18 (28)" columns. This serves
as the published baseline required by the project rubric.

## Hardware

Original results were produced on:

- GPU: NVIDIA RTX 4090 (24 GB)
- CPU: AMD Ryzen 9 5900X (12C/24T)
- RAM: 128 GB

Training is GPU-bound and uses mixed-precision (AMP) for 2D models. 3D
models use full FP32 because GroupNorm + AMP combinations were
numerically unstable on the smaller 3D datasets.

## Notes on design choices (briefly)

- **2D model**: ResNet-style with 3 residual stages (64→128→256 channels),
  BatchNorm2d, dropout 0.5 in the classifier, OneCycleLR.
- **3D model**: ResNet-style with 3 residual stages (32→64→128 channels) but
  GroupNorm (not BatchNorm3d) and cosine annealing — see comments in `cnn.py`
  for why these choices fix the train/eval mismatch on small 3D datasets.
- **Multi-label** (chestmnist): Plain BCE-with-logits + threshold 0.5 +
  per-label mean accuracy (matches the official MedMNIST evaluator). Earlier
  attempts with pos_weight + per-label F1 threshold tuning improved F1 but
  hurt the metric we were being graded on.
- **Laterality-aware augmentation** (organmnist3d): Random flips restricted
  to the depth axis only — flipping left-right would silently swap
  `kidney-left` and `kidney-right` labels.

See header comments in `train2d.py`, `train3d.py`, and `cnn.py` for full
rationale on each per-dataset hyperparameter choice.

## License

Code: MIT. MedMNIST v2 data: CC BY 4.0 (see https://medmnist.com).

## Citation

If you use this code, please cite the original MedMNIST v2 paper:

```
@article{yang2023medmnist,
  title   = {MedMNIST v2 -- A large-scale lightweight benchmark for 2D and
             3D biomedical image classification},
  author  = {Yang, Jiancheng and Shi, Rui and Wei, Donglai and Liu, Zequan
             and Zhao, Lin and Ke, Bilian and Pfister, Hanspeter and Ni,
             Bingbing},
  journal = {Scientific Data},
  volume  = {10},
  number  = {1},
  pages   = {41},
  year    = {2023},
  publisher = {Nature Publishing Group}
}
```
