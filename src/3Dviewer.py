# 3Dviewer.py
# Interactive 3D volume viewer for all MedMNIST 3D datasets.
#
# On startup it auto-generates a coronal GIF for every volume in the chosen (WILL CHANGE TO SELECT WHICH TO MAKE A GIF)
# dataset. GIFs are saved to gifs/<dataset>/  and skipped if they already exist.
#
# Interactive controls while browsing:
#   ENTER / n  → next slice
#   p          → previous slice
#   a / c / s  → axial / coronal / sagittal plane
#   q          → quit viewer

import os
import io
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless backend for GIF rendering (no popup per frame)
import matplotlib.pyplot as plt
matplotlib.use("TkAgg") # switch back to interactive backend for the slice viewer
import matplotlib
from PIL import Image
from tqdm import tqdm

from utils import load_dataset, dataset_to_arrays, get_dataset_info


# Dataset list

DATASETS_3D = [
    "adrenalmnist3d",
    "fracturemnist3d",
    "nodulemnist3d",
    "organmnist3d",
    "synapsemnist3d",
    "vesselmnist3d",
]

GIF_DIR = "gifs"
GIF_FPS = 10


# Volume helpers

def extract_volume(X, index):
    """
    Pull one volume as (D, H, W) float32 in [0, 1].
    Handles every MedMNIST 3D shape variant:
      (N, D, H, W)       — grayscale, no channel
      (N, D, H, W, 1)    — grayscale, channel last
      (N, D, H, W, 3)    — RGB, channel last
      (N, 1, D, H, W)    — grayscale, channel first
    """
    vol = np.asarray(X[index], dtype=np.float32)

    if vol.ndim == 4:
        if vol.shape[0] in (1, 3):          # (C, D, H, W)
            vol = vol[0] if vol.shape[0] == 1 else vol.mean(axis=0)
        elif vol.shape[-1] in (1, 3):       # (D, H, W, C)
            vol = vol[..., 0] if vol.shape[-1] == 1 else vol.mean(axis=-1)
    # ndim == 3 → already (D, H, W)

    vmin, vmax = vol.min(), vol.max()
    if vmax > vmin:
        vol = (vol - vmin) / (vmax - vmin)

    return vol  # (D, H, W)


def get_slice(vol, plane, idx):
    if plane == "axial":
        return vol[idx]
    elif plane == "coronal":
        return vol[:, idx, :]
    elif plane == "sagittal":
        return vol[:, :, idx]
    raise ValueError(f"Unknown plane: {plane}")


def plane_depth(vol, plane):
    if plane == "axial":    return vol.shape[0]
    if plane == "coronal":  return vol.shape[1]
    if plane == "sagittal": return vol.shape[2]


# GIF generator

def gif_path(data_flag, vol_index, label_name, plane="coronal"):
    folder = os.path.join(GIF_DIR, data_flag)
    os.makedirs(folder, exist_ok=True)
    safe_label = label_name.replace(" ", "_").replace("/", "-")
    return os.path.join(folder, f"vol{vol_index:04d}_{safe_label}_{plane}.gif")


def gif_exists(data_flag, vol_index, label_name, plane="coronal"):
    return os.path.exists(gif_path(data_flag, vol_index, label_name, plane))


def save_gif_for_volume(vol, data_flag, vol_index, label_name, plane="coronal"):
    """
    Render every slice along `plane` to a GIF and save it.
    Uses the Agg (non-interactive) backend so no windows pop up during batch export.
    Returns the saved path.
    """
    depth  = plane_depth(vol, plane)
    frames = []

    # Use Agg for headless frame rendering
    orig_backend = matplotlib.get_backend()
    matplotlib.use("Agg")

    for i in range(depth):
        slc = get_slice(vol, plane, i)
        fig, ax = plt.subplots(figsize=(4, 4), dpi=80)
        ax.imshow(slc, cmap="gray", vmin=0, vmax=1)
        ax.set_title(
            f"{data_flag}  vol {vol_index}\n"
            f"{label_name}  |  {plane} {i}/{depth - 1}",
            fontsize=8,
        )
        ax.axis("off")
        fig.tight_layout(pad=0.4)

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).copy())

    out_path = gif_path(data_flag, vol_index, label_name, plane)
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=int(1000 / GIF_FPS),
    )

    # Restore interactive backend
    matplotlib.use(orig_backend)

    return out_path


def generate_all_gifs(X, y, info, data_flag, plane="coronal"):
    """
    Walk every volume in X and generate a coronal GIF.
    Skips any volume whose GIF file already exists in gifs/<data_flag>/.
    """
    labels = info["label"]
    total  = len(X)
    skip   = 0

    print(f"\n  Generating {plane} GIFs for all {total} volumes in {data_flag} ...")
    print(f"  Saving to: {os.path.abspath(os.path.join(GIF_DIR, data_flag))}/")

    # Count how many already exist
    for i in range(total):
        lv   = y[i]
        lv   = int(lv) if not hasattr(lv, "__len__") else int(lv[0])
        lname = labels.get(str(lv), str(lv))
        if gif_exists(data_flag, i, lname, plane):
            skip += 1

    if skip == total:
        print(f"  All {total} GIFs already exist — nothing to generate.")
        return

    print(f"  {skip} already exist, generating {total - skip} new GIF(s)...")

    for i in tqdm(range(total), desc="GIFs", unit="vol"):
        lv    = y[i]
        lv    = int(lv) if not hasattr(lv, "__len__") else int(lv[0])
        lname = labels.get(str(lv), str(lv))

        if gif_exists(data_flag, i, lname, plane):
            continue  # already on disk — skip

        vol  = extract_volume(X, i)
        path = save_gif_for_volume(vol, data_flag, i, lname, plane)

    print(f"  Done. GIFs saved to {os.path.join(GIF_DIR, data_flag)}/")


# Interactive slice viewer

def view_volume_interactive(vol, label_name, vol_index, data_flag, plane="coronal"):
    """
    Keyboard-driven slice browser (interactive matplotlib window).

    Controls:
      ENTER / n  → next slice
      p          → previous slice
      a          → axial
      c          → coronal
      s          → sagittal
      q          → quit
    """
    depth     = plane_depth(vol, plane)
    slice_idx = depth // 2

    print(f"\n  Volume {vol_index}  |  Label: {label_name}")
    print(f"  Shape  : {vol.shape}  (D × H × W)")
    print(f"  Plane  : {plane}  ({depth} slices)")
    print()
    print("  Controls:")
    print("    ENTER / n  → next slice")
    print("    p          → previous slice")
    print("    a / c / s  → axial / coronal / sagittal")
    print("    q          → quit\n")

    while True:
        depth = plane_depth(vol, plane)
        slc   = get_slice(vol, plane, slice_idx)

        plt.figure(figsize=(5, 5))
        plt.imshow(slc, cmap="gray", vmin=0, vmax=1)
        plt.title(
            f"{data_flag}  |  Vol {vol_index}  |  {label_name}\n"
            f"{plane.capitalize()} slice {slice_idx}/{depth - 1}"
        )
        plt.axis("off")
        plt.tight_layout()
        plt.show()

        cmd = input("  Command [ENTER/n/p/a/c/s/q]: ").strip().lower()

        if cmd in ("", "n"):
            slice_idx = min(depth - 1, slice_idx + 1)
        elif cmd == "p":
            slice_idx = max(0, slice_idx - 1)
        elif cmd == "a":
            plane     = "axial"
            slice_idx = plane_depth(vol, plane) // 2
            print(f"  → axial ({plane_depth(vol, plane)} slices)")
        elif cmd == "c":
            plane     = "coronal"
            slice_idx = plane_depth(vol, plane) // 2
            print(f"  → coronal ({plane_depth(vol, plane)} slices)")
        elif cmd == "s":
            plane     = "sagittal"
            slice_idx = plane_depth(vol, plane) // 2
            print(f"  → sagittal ({plane_depth(vol, plane)} slices)")
        elif cmd == "q":
            print("  Exiting viewer.")
            break
        else:
            print("  Unknown command.")


# Main

def main():
    print("=" * 56)
    print("  MedMNIST 3D Viewer  +  Coronal GIF Export")
    print("=" * 56)

    # Dataset selection
    print("\nAvailable 3D datasets:")
    for i, flag in enumerate(DATASETS_3D):
        print(f"  [{i}] {flag}")

    choice = input("\nEnter number or dataset name: ").strip()
    if choice.isdigit() and int(choice) < len(DATASETS_3D):
        data_flag = DATASETS_3D[int(choice)]
    elif choice in DATASETS_3D:
        data_flag = choice
    else:
        print(f"  '{choice}' not recognised — defaulting to adrenalmnist3d.")
        data_flag = "adrenalmnist3d"

    # load 
    print(f"\n  Loading {data_flag} ...")
    train_ds, _, _ = load_dataset(data_flag)
    X_train, y_train = dataset_to_arrays(train_ds, "train", data_flag)
    info = get_dataset_info(data_flag)

    print(f"  Loaded {len(X_train)} volumes  |  raw shape: {X_train.shape}")

    print("\n  Labels:")
    for lid, lname in info["label"].items():
        count = int((y_train == int(lid)).sum())
        print(f"    [{lid}] {lname}  (n={count})")

    # Auto-generate coronal GIFs for the whole dataset (if not already on disk)
    # NOTE: THIS MAKES WAY TOO MANY GIFS (one per volume) SO IT'S COMMENTED OUT BY DEFAULT. UNCOMMENT TO USE.
    # generate_all_gifs(X_train, y_train, info, data_flag, plane="coronal")

    # iInteractive viewer 
    vol_idx = input(
        f"\n  Enter volume index to view [0 – {len(X_train) - 1}] "
        f"(or ENTER to skip): "
    ).strip()

    if vol_idx == "":
        print("  Skipping interactive viewer.")
        return

    vol_idx = max(0, min(int(vol_idx), len(X_train) - 1))

    lv    = y_train[vol_idx]
    lv    = int(lv) if not hasattr(lv, "__len__") else int(lv[0])
    lname = info["label"].get(str(lv), str(lv))

    vol = extract_volume(X_train, vol_idx)

    # Switch to interactive backend for the slice viewer
    matplotlib.use("TkAgg")

    view_volume_interactive(vol, lname, vol_idx, data_flag, plane="coronal")


if __name__ == "__main__":
    main()