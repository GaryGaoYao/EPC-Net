# Public-release code changes

The three scripts preserve the original computational stages and principal hyperparameters while improving release quality.

## Stage 1

- Replaced fixed Windows paths with command-line arguments.
- Replaced order-based `zip(images, labels)` pairing with strict case-ID matching.
- Added seeded random train/validation splitting and a saved split manifest.
- Added complete checkpoints, resume support, atomic writes, CSV logs, AMP controls, and optional gradient clipping.
- Preserved SegResNet, percentile intensity normalization, 1-mm spacing, random positive/negative crops, Dice loss, and sliding-window validation.

## Stage 2

- Fixed checkpoint resume so that it resumes an existing run rather than checking a newly created empty directory.
- Applied the configured `(0.5, 0.5, 1.0)`-mm spacing directly to the corresponding MONAI spatial dimensions when computing physical eye distances.
- Replaced voxel-iteration dilation with distance-based dilation in millimetres.
- Normalized weighted cross-entropy by the sum of spatial weights.
- Added strict three-way matching among CT, label, and eye-prediction files.
- Added complete scheduler/checkpoint/RNG restoration and consistent manifests/logging.
- Preserved the two-channel SegResNet and eye-weighted Dice + cross-entropy formulation.

## Stage 3

- Added a headless CLI while retaining an optional GUI.
- Replaced the Python per-voxel VTK copy loop with NumPy-to-VTK conversion.
- Corrected voxel flattening to preserve x-fastest VTK ordering from SimpleITK `[z, y, x]` arrays.
- Extracted the selected integer label as a binary mask before Marching Cubes.
- Applied image direction through an explicit physical-coordinate transform.
- Added mesh cleaning, binary STL output, input validation, and clearer errors.
