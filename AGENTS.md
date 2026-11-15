# AGENTS.md

## Project

Computer-vision assignment: reconstruct 3D points from **two pre-calibrated** images and plot a point cloud.
- `main.py` is the deliverable and currently only contains placeholder comments (Step2–Step6).
- `dev/plan.md` is the detailed, already-validated implementation plan; follow it.
- `dev/introduce.md` is the assignment statement.
- No git repo, no tests, no build/lint config, no requirements file.

## Environment

- Windows PowerShell 5.1. The absolute path contains Chinese characters (`作业/计算机视觉`); always quote paths and prefer the `workdir` tool parameter over `cd`.
- Python 3.12 (`D:\DevTools\python3.12.10\python.exe`). Installed: `opencv-python` 4.11 (SIFT built in), `numpy` 2.2.6, `scipy`, `matplotlib`.
- Run with `python main.py`. Do not assume pytest/ruff/etc. exist.

## Data / calibration gotchas

- `data/templeRing/templeR_par.txt` and `data/dinoRing/dinoR_par.txt`: **first line is the view count**, not a camera. Each camera line has 22 tokens: `name` + `K`(9, row-major) + `R`(9, row-major) + `t`(3). `P = K @ [R|t]` (world→camera).
- Same intrinsics for all templeRing views: `fx=1520.4, fy=1525.9, cx=302.32, cy=246.87`.
- **Image size is inconsistent in templeRing**: `templeR0001.png` is 480×640; every other temple image and all dino images are 640×480. Avoid `0001` or rotate it, otherwise pixel coordinates won't match `K`.
- Calibration is given, so do **not** estimate pose for reconstruction: build `P1, P2` from the par file and triangulate directly. `findEssentialMat` / `recoverPose` is only optional verification against the known `R_rel = R2 @ R1.T`, `t_rel = t2 - R2 @ R1.T @ t1`.
- Reconstructed points are already in the dataset world frame, so they can be checked against the README bounding box. Do not rescale/normalize.
- `templeRing` is textured and works with SIFT. `dinoRing` is untextured/matte, so use a small baseline: `dinoR0026.png + dinoR0028.png` (15.7°) gives ~127 ratio-test matches and a good reconstruction; larger deltas drop sharply.

## Running

- `python main.py` reconstructs both datasets; `python main.py --dataset temple` or `--dataset dino` does one.
- Outputs: `output/reconstruction_temple.png/.ply`, `output/reconstruction_dino.png/.ply`. In headless runs set `MPLBACKEND=Agg` (the `plt.show()` warning is harmless).

## Reconstruction notes

- Pass `2×N` `float32` points to `cv2.triangulatePoints`; it returns `4×N` homogeneous — divide by the 4th row.
- Filter points by **positive depth in both cameras** (cheirality) **and** reprojection error < ~2 px.

## Session log convention

- After every reply, append a summary to `dev/<yyyy-mm-dd>/对话概要.md` (use today's date). Create the date folder if it does not exist.
