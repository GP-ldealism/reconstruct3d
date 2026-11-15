import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt

DATASETS = {
    "temple": {
        "dir": "data/templeRing",
        "par": "data/templeRing/templeR_par.txt",
        "img1": "templeR0002.png",
        "img2": "templeR0005.png",
        "prefix": "reconstruction_temple",
        "title": "Temple Ring 3D Reconstruction",
        "bbox_min": [-0.023121, -0.038009, -0.091940],
        "bbox_max": [0.078626, 0.121636, -0.017395],
    },
    "dino": {
        "dir": "data/dinoRing",
        "par": "data/dinoRing/dinoR_par.txt",
        "img1": "dinoR0026.png",
        "img2": "dinoR0028.png",
        "prefix": "reconstruction_dino",
        "title": "Dino Ring 3D Reconstruction",
        "bbox_min": [-0.021897, 0.021126, -0.017845],
        "bbox_max": [0.050897, 0.108227, 0.055495],
    },
}

RATIO = 0.7
REPROJ_THRESH = 2.0


def load_camera(par_path, image_name):
    """读取指定图像的内参和外参，并构造投影矩阵 P = K[R|t]。"""
    with open(par_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 22 and parts[0] == image_name:
                v = np.array([float(x) for x in parts[1:]])
                K = v[0:9].reshape(3, 3)
                R = v[9:18].reshape(3, 3)
                t = v[18:21].reshape(3, 1)
                P = K @ np.hstack([R, t])
                return K, R, t, P
    raise FileNotFoundError(f"Camera parameters not found for {image_name}")


def detect_and_match(gray1, gray2, ratio):
    """提取两幅灰度图的 SIFT 特征，并返回通过距离比检验的 2×N 像素坐标。"""
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    good = []
    if des1 is not None and des2 is not None:
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        for neighbors in matcher.knnMatch(des1, des2, k=2):
            if len(neighbors) == 2:
                best, second = neighbors
                if best.distance < ratio * second.distance:
                    good.append(best)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 2).T
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 2).T
    return len(kp1), len(kp2), pts1, pts2


def verify_pose(K1, R1, t1, R2, t2, pts1, pts2):
    """在两视图内参相同时，比较估计相对位姿与标定相对位姿。"""
    R_rel = R2 @ R1.T
    t_rel = (t2 - R2 @ R1.T @ t1).ravel()

    E_est, mask = cv2.findEssentialMat(
        pts1.T, pts2.T, K1, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R_est, t_est, _ = cv2.recoverPose(
        E_est, pts1.T, pts2.T, K1, mask=mask.copy())

    cos_r = (np.trace(R_est.T @ R_rel) - 1.0) / 2.0
    rot_err = np.degrees(np.arccos(np.clip(cos_r, -1.0, 1.0)))

    t_rel_dir = t_rel / np.linalg.norm(t_rel)
    t_est_dir = t_est.ravel() / np.linalg.norm(t_est)
    # 取绝对值比较平移轴的夹角，忽略方向的正负号。
    cos_t = np.abs(np.dot(t_rel_dir, t_est_dir))
    trans_err = np.degrees(np.arccos(np.clip(cos_t, -1.0, 1.0)))

    return int(mask.sum()), rot_err, trans_err


def triangulate(P1, P2, pts1, pts2, thresh):
    """线性三角化得到世界坐标，返回候选点、保留掩码和双视图最大重投影误差。"""
    X_h = cv2.triangulatePoints(P1, P2, pts1, pts2).astype(np.float64)
    scale = X_h[3]
    finite = np.isfinite(X_h).all(axis=0) & (np.abs(scale) > np.finfo(scale.dtype).eps)
    X = np.full((len(scale), 3), np.nan)
    X[finite] = (X_h[:3, finite] / scale[finite]).T

    # 统一令齐次坐标末项为 1，使投影第三项对应相机坐标系中的深度。
    X_euclidean_h = np.vstack([X.T, np.ones(len(X))])
    proj1 = P1 @ X_euclidean_h
    proj2 = P2 @ X_euclidean_h
    projectable = finite & (proj1[2] != 0) & (proj2[2] != 0)
    uv1 = (proj1[:2, projectable] / proj1[2, projectable]).T
    uv2 = (proj2[:2, projectable] / proj2[2, projectable]).T
    err1 = np.linalg.norm(uv1 - pts1[:, projectable].T, axis=1)
    err2 = np.linalg.norm(uv2 - pts2[:, projectable].T, axis=1)
    err = np.full(len(X), np.inf)
    err[projectable] = np.maximum(err1, err2)
    mask = (proj1[2] > 0) & (proj2[2] > 0) & (err < thresh)
    return X, mask, err


def save_ply(path, points, colors):
    """将世界坐标和归一化 RGB 颜色写为 ASCII PLY 点云。"""
    colors_u8 = (np.clip(colors, 0.0, 1.0) * 255).astype(np.uint8)
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(points, colors_u8):
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")


def visualize(points, colors, image_path, title):
    """按三轴坐标范围设置显示比例，绘制并保存彩色三维散点图。"""
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=3)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)

    ranges = np.ptp(points, axis=0)
    ranges[ranges == 0] = 1e-6
    ax.set_box_aspect(ranges)
    ax.view_init(elev=20, azim=-60)

    plt.tight_layout()
    plt.savefig(image_path, dpi=200)
    plt.show()


def process_dataset(name, cfg):
    """运行一个图像对的双目重构，输出检查指标及彩色点云文件。"""
    print(f"\n=== {name} ===")
    K1, R1, t1, P1 = load_camera(cfg["par"], cfg["img1"])
    K2, R2, t2, P2 = load_camera(cfg["par"], cfg["img2"])

    img1 = cv2.imread(f"{cfg['dir']}/{cfg['img1']}")
    img2 = cv2.imread(f"{cfg['dir']}/{cfg['img2']}")
    if img1 is None or img2 is None:
        raise FileNotFoundError("Failed to load input images")
    if img1.shape != img2.shape:
        raise ValueError(f"Image size mismatch: {img1.shape} vs {img2.shape}")
    print(f"Images: {cfg['img1']} + {cfg['img2']} {img1.shape[1]}x{img1.shape[0]}")

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    n_kp1, n_kp2, pts1, pts2 = detect_and_match(gray1, gray2, RATIO)
    print(f"Keypoints: {n_kp1} / {n_kp2}, good matches: {pts1.shape[1]}")
    if pts1.shape[1] < 8:
        print("Too few matches, skipping")
        return

    try:
        inliers, rot_err, trans_err = verify_pose(
            K1, R1, t1, R2, t2, pts1, pts2)
        print(f"Essential-matrix inliers: {inliers}, "
              f"rotation error: {rot_err:.3f} deg, "
              f"translation direction error: {trans_err:.3f} deg")
    except cv2.error as exc:
        print(f"Pose verification failed: {exc}")

    X, mask, err = triangulate(P1, P2, pts1, pts2, REPROJ_THRESH)
    X_in = X[mask]
    if len(X_in) == 0:
        print(f"Triangulated: {len(X)}, kept: 0 (0.0%). No point cloud to save.")
        return
    print(f"Triangulated: {len(X)}, kept: {len(X_in)} "
          f"({100.0 * len(X_in) / len(X):.1f}%), "
          f"median reprojection error: {np.median(err[mask]):.3f} px")

    rec_min = X_in.min(axis=0)
    rec_max = X_in.max(axis=0)
    true_min = np.array(cfg["bbox_min"])
    true_max = np.array(cfg["bbox_max"])
    print("Reconstructed bbox min:", np.round(rec_min, 4),
          "max:", np.round(rec_max, 4))
    print("Ground-truth  bbox min:", true_min,
          "max:", true_max)
    bbox_deviations = np.abs(np.stack([rec_min - true_min, rec_max - true_max]))
    print(f"Max bbox deviation: {bbox_deviations.max():.4f}")

    # 用第一幅图像中对应位置的 RGB 值为三维点着色。
    inlier_pts1 = pts1[:, mask]
    rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    xs = np.clip(inlier_pts1[0].astype(int), 0, rgb.shape[1] - 1)
    ys = np.clip(inlier_pts1[1].astype(int), 0, rgb.shape[0] - 1)
    colors = rgb[ys, xs] / 255.0

    ply_path = f"{cfg['prefix']}.ply"
    png_path = f"{cfg['prefix']}.png"
    save_ply(ply_path, X_in, colors)
    print(f"Saved {ply_path} and {png_path}")
    visualize(X_in, colors, png_path, cfg["title"])


def main():
    """解析数据集选项，依次运行选定的双目重构实验。"""
    parser = argparse.ArgumentParser(description="Two-view 3D reconstruction")
    parser.add_argument("--dataset", choices=["temple", "dino", "all"],
                        default="all")
    args = parser.parse_args()

    names = DATASETS.keys() if args.dataset == "all" else [args.dataset]
    for name in names:
        process_dataset(name, DATASETS[name])


if __name__ == '__main__':
    main()
