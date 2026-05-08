# 3Dviewer.py
# Interactive 3D volume viewer for all MedMNIST 3D datasets.
#
# On startup, generates one coronal GIF per label — a 20x20 grid of volumes
# sweeping through coronal(front/back) slices. If the GIF already exists, it is skipped.
# GIFs are saved to gifs/<dataset>/<label_name>_coronal_grid.gif
#
# After generating GIFs, you can interactively explore any volume slice-by-slice in axial/coronal/sagittal planes.
# Interactive controls:
#   ENTER / n  → next slice
#   p          → previous slice
#   a / c / s  → axial(top/bottom) / coronal(front/back) / sagittal(left/right)
#   q          → quit

import os
import io
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from tqdm import tqdm

from utils import load_dataset, dataset_to_arrays, get_dataset_info


DATASETS_3D = [
    "adrenalmnist3d",
    "fracturemnist3d",
    "nodulemnist3d",
    "organmnist3d",
    "synapsemnist3d",
    "vesselmnist3d",
]

GIF_DIR  = "gifs"
GIF_FPS  = 8
GRID_N   = 20   # 20×20 grid = up to 400 volumes per label GIF


def extract_volume(X, index):
    vol = np.asarray(X[index], dtype=np.float32)
    if vol.ndim == 4:
        if vol.shape[0] in (1, 3):
            vol = vol[0] if vol.shape[0] == 1 else vol.mean(axis=0)
        elif vol.shape[-1] in (1, 3):
            vol = vol[..., 0] if vol.shape[-1] == 1 else vol.mean(axis=-1)
    vmin, vmax = vol.min(), vol.max()
    if vmax > vmin:
        vol = (vol - vmin) / (vmax - vmin)
    return vol  # (D, H, W)


def get_slice(vol, plane, idx):
    if plane == "axial":    return vol[idx]
    if plane == "coronal":  return vol[:, idx, :]
    if plane == "sagittal": return vol[:, :, idx]


def plane_depth(vol, plane):
    if plane == "axial":    return vol.shape[0]
    if plane == "coronal":  return vol.shape[1]
    if plane == "sagittal": return vol.shape[2]


def label_gif_path(data_flag, label_name, plane="coronal"):
    folder = os.path.join(GIF_DIR, data_flag)
    os.makedirs(folder, exist_ok=True)
    safe = label_name.replace(" ", "_").replace("/", "-")
    return os.path.join(folder, f"{safe}_{plane}_grid.gif")


def label_gif_exists(data_flag, label_name, plane="coronal"):
    return os.path.exists(label_gif_path(data_flag, label_name, plane))


def render_grid_frame(volumes, slice_idx, plane, label_name, grid_n=GRID_N, cell_px=64):
    """
    Render one frame of the grid GIF: a grid_n × grid_n mosaic of coronal slices.
    volumes: list of (D, H, W) float32 arrays.
    Returns a PIL Image.
    """
    n_vols = len(volumes)
    total  = grid_n * grid_n

    fig_size = grid_n * cell_px / 80  # inches at 80 dpi
    fig, axes = plt.subplots(grid_n, grid_n, figsize=(fig_size, fig_size + 0.4), dpi=80)

    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < n_vols:
            vol = volumes[i]
            depth = plane_depth(vol, plane)
            si    = min(slice_idx, depth - 1)
            slc   = get_slice(vol, plane, si)
            ax.imshow(slc, cmap="gray", vmin=0, vmax=1, aspect="auto")

    fig.suptitle(f"{label_name}  |  {plane} slice {slice_idx}", fontsize=9, y=1.0)
    fig.tight_layout(pad=0.1)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def generate_label_gif(X, y, label_id, label_name, data_flag, plane="coronal"):
    """
    Build a single grid GIF for all volumes with label == label_id.
    The GIF sweeps through every coronal slice.
    Skips if the file already exists.
    """
    path = label_gif_path(data_flag, label_name, plane)
    if os.path.exists(path):
        print(f"  [skip] {os.path.basename(path)} already exists.")
        return path

    indices = [i for i in range(len(y)) if int(y[i]) == label_id]
    if not indices:
        print(f"  [skip] No volumes for label {label_id} ({label_name}).")
        return None

    # Cap at GRID_N^2 volumes, evenly sampled so the grid is representative
    max_vols = GRID_N * GRID_N
    if len(indices) > max_vols:
        step    = len(indices) / max_vols
        indices = [indices[int(i * step)] for i in range(max_vols)]

    print(f"  Generating grid GIF: {label_name}  ({len(indices)} volumes) ...")
    volumes = [extract_volume(X, i) for i in tqdm(indices, desc=f"  Loading {label_name}", leave=False)]

    # Use the minimum depth across all volumes so the slice index is always valid
    min_depth = min(plane_depth(v, plane) for v in volumes)
    frames    = []

    for si in tqdm(range(min_depth), desc=f"  Frames {label_name}", leave=False):
        frame = render_grid_frame(volumes, si, plane, label_name)
        frames.append(frame)

    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=int(1000 / GIF_FPS),
        optimize=False,
    )
    print(f"  Saved → {path}  ({len(frames)} frames, {len(indices)} volumes)")
    return path


def generate_all_label_gifs(X, y, info, data_flag, plane="coronal"):
    labels = info["label"]
    print(f"\n  Generating per-label {plane} grid GIFs for {data_flag} ...")
    for lid, lname in labels.items():
        generate_label_gif(X, y, int(lid), lname, data_flag, plane)
    print("  All label GIFs done.")


# Interactive viewer (uses TkAgg for display)

def view_volume_interactive(vol, label_name, vol_index, data_flag, plane="coronal"):
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt2

    depth     = plane_depth(vol, plane)
    slice_idx = depth // 2

    print(f"\n  Volume {vol_index}  |  Label: {label_name}")
    print(f"  Shape  : {vol.shape}  (D × H × W)")
    print(f"  Plane  : {plane}  ({depth} slices)")
    print("  Controls: ENTER/n = next  p = prev  a/c/s = plane  q = quit\n")

    while True:
        depth = plane_depth(vol, plane)
        slc   = get_slice(vol, plane, slice_idx)

        plt2.figure(figsize=(5, 5))
        plt2.imshow(slc, cmap="gray", vmin=0, vmax=1)
        plt2.title(f"{data_flag}  |  Vol {vol_index}  |  {label_name}\n{plane} {slice_idx}/{depth - 1}")
        plt2.axis("off")
        plt2.tight_layout()
        plt2.show()

        cmd = input("  Command [ENTER/n/p/a/c/s/q]: ").strip().lower()

        if cmd in ("", "n"):
            slice_idx = min(depth - 1, slice_idx + 1)
        elif cmd == "p":
            slice_idx = max(0, slice_idx - 1)
        elif cmd == "a":
            plane = "axial";    slice_idx = plane_depth(vol, plane) // 2
        elif cmd == "c":
            plane = "coronal";  slice_idx = plane_depth(vol, plane) // 2
        elif cmd == "s":
            plane = "sagittal"; slice_idx = plane_depth(vol, plane) // 2
        elif cmd == "q":
            print("  Exiting viewer."); break
        else:
            print("  Unknown command.")


def main():
    print("=" * 56)
    print("  MedMNIST 3D Viewer  +  Per-Label Coronal Grid GIFs")
    print("=" * 56)

    print("\nAvailable 3D datasets:")
    for i, flag in enumerate(DATASETS_3D):
        print(f"  [{i}] {flag}")

    choice = input("\nEnter number or dataset name: ").strip()
    if choice.isdigit() and int(choice) < len(DATASETS_3D):
        data_flag = DATASETS_3D[int(choice)]
    elif choice in DATASETS_3D:
        data_flag = choice
    else:
        print(f"  Defaulting to adrenalmnist3d.")
        data_flag = "adrenalmnist3d"

    print(f"\n  Loading {data_flag} ...")
    train_ds, _, _ = load_dataset(data_flag)
    X_train, y_train = dataset_to_arrays(train_ds, "train", data_flag)
    info = get_dataset_info(data_flag)

    print(f"  Loaded {len(X_train)} volumes  |  raw shape: {X_train.shape}")
    print("\n  Labels:")
    for lid, lname in info["label"].items():
        count = int((y_train == int(lid)).sum())
        print(f"    [{lid}] {lname}  (n={count})")

    # Generate one grid GIF per label (skips existing)
    generate_all_label_gifs(X_train, y_train, info, data_flag, plane="coronal")

    # Interactive viewer
    vol_idx = input(
        f"\n  Enter volume index to view [0–{len(X_train) - 1}] (or ENTER to skip): "
    ).strip()
    if vol_idx == "":
        print("  Skipping interactive viewer.")
        return

    vol_idx = max(0, min(int(vol_idx), len(X_train) - 1))
    lv      = int(y_train[vol_idx]) if not hasattr(y_train[vol_idx], "__len__") else int(y_train[vol_idx][0])
    lname   = info["label"].get(str(lv), str(lv))
    vol     = extract_volume(X_train, vol_idx)

    view_volume_interactive(vol, lname, vol_idx, data_flag, plane="coronal")


if __name__ == "__main__":
    main()