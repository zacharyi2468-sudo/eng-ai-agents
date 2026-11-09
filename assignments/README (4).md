# Assignment 3: Anti-UAV Visual Tracking System

**Author:** Zachary Ibrahim
**Submitted:** 2026-11-08

Drone detection on the DUT Anti-UAV benchmark with a fine-tuned YOLO11n, and single-drone tracking with a
constant-velocity Kalman filter (`filterpy`) using χ² Mahalanobis gating, confidence-weighted
association, evidence-based track loss, and a reacquisition rule for lost tracks.

**Results (all 20 sequences):** success AUC 0.644 vs 0.624 for the detector alone, precision@20px
0.865 vs 0.834, absent-box rate 0.69 vs 1.00; video13 AUC 0.261 vs 0.122. Detector test mAP@0.5:
0.890 at 640, 0.906 at 1280.

## Contents

| File | What it is |
| - | - |
| `dut-anti-uav.ipynb` | The notebook, with every cell output saved |
| `video12_kalman.mp4` | Output video: ground truth (green), Kalman box (red) and its trajectory |
| `video03_kalman.mp4` | The sequence analyzed in question 4 (where the tracker trails the detector) |

## Running the notebook

1. Open `dut-anti-uav.ipynb` in Google Colab.
2. Runtime → Change runtime type → **T4 GPU**.
3. Runtime → Run all. The first cell installs `ultralytics`, `filterpy` and `datasets`; the data is
   downloaded from the Hugging Face dataset `aegean-ai/dut-anti-uav`.
4. The full run takes roughly 40–50 minutes on a T4: conversion of the detection images, 30 epochs of
   training, test-split evaluation at 640 and 1280, a detection pass over the 24,804 tracking frames,
   then tracker tuning, scoring, the noise sweep and the question cells (a few minutes, CPU only).
5. If the session disconnects, rerun all cells: converted images are reused if the runtime's files
   survive; training and the detection pass rerun.

The tracker's settings are chosen on the odd-numbered sequences by the tuning cell in the
"Tracking all 20 sequences" section; the even-numbered sequences are only used for checking.

## Tests

No separate tests. The notebook's own check cell ("Losing and reacquiring the drone") verifies the two
required conditions: video13 AUC at least that of the detector alone, and an absent-box rate below it
over all 20 sequences.
