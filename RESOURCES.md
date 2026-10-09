# Resources — ENG 2440 Assignment 2

## Data
- MSD Task01_BrainTumour (MONAI mirror, ~7.6 GB): https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar
- Official MSD site: https://medicaldecathlon.com/
- Note: MSD stores the 4 modalities as one 4D NIfTI (H×W×D×4). Check `dataset.json` → `"modality"` for channel order
  (FLAIR, T1w, T1gd, T2w) and `"labels"` (0 background, 1 edema, 2 non-enhancing tumor, 3 enhancing tumor).

## Papers to cite
- Antonelli et al. (2022) *The Medical Segmentation Decathlon*. Nat. Commun. 13:4128. doi:10.1038/s41467-022-30695-9
- Menze et al. (2015) *The BRATS benchmark*. IEEE TMI 34(10):1993–2024. doi:10.1109/TMI.2014.2377694
- Çiçek et al. (2016) *3D U-Net*. MICCAI. arXiv:1606.06650
- Isensee et al. (2021) *nnU-Net*. Nat. Methods 18:203–211 — reference for patch sampling (⅓ foreground oversampling), sliding window with Gaussian blending
- Maier-Hein, Reinke et al. (2024) *Metrics Reloaded*. Nat. Methods — why Dice alone is insufficient; HD95 vs volume error
- Milletari et al. (2016) *V-Net* (Dice loss). arXiv:1606.04797

## Library docs (by section)
- **A / C – NIfTI geometry:** nibabel coordinate systems https://nipy.org/nibabel/coordinate_systems.html ;
  `nib.aff2axcodes`, `img.header.get_zooms()`, `nib.as_closest_canonical`
- **C – Resampling:** SimpleITK `ResampleImageFilter` (sitkLinear / sitkNearestNeighbor),
  `scipy.ndimage.zoom` (order=1 vs order=0)
- **D – Data pipeline:** PyTorch Dataset/DataLoader tutorial https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
- **D – Model:** `torch.nn.Conv3d`, `InstanceNorm3d`, `ConvTranspose3d`
- **F – Metrics/postproc:** `scipy.ndimage.label`, `scipy.ndimage.distance_transform_edt` (pass `sampling=spacing` for mm),
  `scipy.ndimage.binary_erosion` for surface extraction
- **Reference implementations to compare against (cite if used):**
  MONAI `sliding_window_inference`, `RandCropByPosNegLabeld`, `HausdorffDistanceMetric` — https://docs.monai.io/
  MONAI BraTS tutorial: https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/brats_segmentation_3d.ipynb

## Practical tips
- Volumes are 240×240×155 at 1 mm isotropic — check whether resampling is actually needed before doing it.
- Volume (mL) = voxel count × prod(spacing in mm) / 1000.
- HD95 is undefined when prediction or reference is empty — decide and document how you handle it.
- On a laptop (CPU/MPS), keep patches small (e.g. 64³–96³) and the base channel count low; cache preprocessed volumes as `.npy` to avoid re-decompressing .nii.gz every epoch.
