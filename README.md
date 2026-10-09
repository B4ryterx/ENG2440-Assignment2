# ENG 2440 — Assignment 2: Tumor Segmentation in MRI Volumes

Whole-tumor segmentation of MSD Task01_BrainTumour (4-channel 3D MRI) with a lightweight
3D U-Net trained from patches in PyTorch, comparing uniform vs tumor-aware patch sampling,
with overlapping sliding-window reconstruction and evaluation in physical units.
**Learning exercise, not a clinical tool.**

## Repository contents
| File | Purpose |
|---|---|
| `Assignment2_Student.ipynb` | Fully executed notebook, Sections A–G + both bonus questions |
| `seg_utils.py` | Supporting code: preprocessing, patch Dataset/samplers, augmentation, 3D U-Net, loss, training loop, sliding-window inference, metrics (Dice/IoU/sensitivity/HD95), postprocessing, plotting helpers |
| `assignment2_split.csv` | Supplied case-level split (used unchanged) |
| `environment.yml` / `requirements.txt` | Environment |
| `make_report.py` | Builds the PDF report from `outputs/` (edit name/ID/repo at the top, then `python make_report.py`) |
| `download_data.sh` | Downloads + extracts Task01_BrainTumour (~7.6 GB) |
| `outputs/` | Figures, per-case metric CSVs, training histories, checkpoints (`*_best.pt`, ~6 MB), `run_info.json` |

The dataset and the preprocessed cache (`outputs/cache/`, ~16 GB) are **not** committed (`.gitignore`).

## Setup
```bash
conda env create -f environment.yml
conda activate eng2440
python -m ipykernel install --user --name eng2440 --display-name "Python (eng2440)"
./download_data.sh            # or point DATA_ROOT at an existing copy
```
Expected layout:
```
Task01_BrainTumour/{imagesTr,labelsTr,dataset.json}
assignment2_split.csv
Assignment2_Student.ipynb
seg_utils.py
```

## Exact run order
1. Edit **`DATA_ROOT`** in the first code cell of `Assignment2_Student.ipynb` (only path to change).
2. Select kernel *Python (eng2440)* → **Run All** (top to bottom). Or headless:
   ```bash
   jupyter nbconvert --to notebook --execute Assignment2_Student.ipynb --output Assignment2_Student.ipynb --ExecutePreprocessor.timeout=-1
   ```
3. Section C3 preprocesses all 484 cases into `outputs/cache/` (parallel, ~3 min, reused on re-runs).
4. Sections D3/D4 train the two models. With `RETRAIN = False`, existing checkpoints in `outputs/` are reused.

## Reproducibility
| Item | Value |
|---|---|
| Random seed | 42 (Python, NumPy, PyTorch; every training patch uses RNG seeded by (42, patch index)) |
| Initialisation | identical for both samplers (`outputs/init_weights.pt`) |
| Hardware | Apple M4 Max (14-core CPU, 36 GB RAM), PyTorch MPS backend |
| Software | Python 3.11, PyTorch 2.13 (conda-forge), nibabel 5.4, NumPy 2.4, SciPy |
| Patch / batch / steps | 64³ / 4 / 2000 steps per sampler (8000 patches), Adam lr 1e-3, poly decay |
| Inference | sliding window 64³, 50% overlap (stride 32), Gaussian blending, threshold 0.5 |
| Training time | 37.4 min (uniform) + 33.3 min (tumor-aware), incl. validation monitoring |
| Total notebook runtime | 117 min end-to-end (Run All, incl. preprocessing, both trainings, evaluation and both bonus questions) |

MPS kernels are not bit-wise deterministic, so re-runs can differ slightly in the last decimals.

## Data and references
- Medical Segmentation Decathlon, Task01_BrainTumour — Antonelli et al., *Nature Communications* 13:4128 (2022); data CC-BY-SA 4.0.
- Menze et al., The BRATS benchmark, *IEEE TMI* 34(10):1993–2024 (2015).
- Çiçek et al., 3D U-Net, MICCAI 2016 (architecture pattern); Isensee et al., nnU-Net, *Nature Methods* 18:203–211 (2021) (foreground oversampling, Gaussian sliding-window blending, poly LR — design ideas, re-implemented here, no code copied).
