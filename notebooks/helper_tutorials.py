from collections import defaultdict

def find_repeated_indices(arr):
    
    index_map = defaultdict(list)
    for idx, val in enumerate(arr):
        index_map[val].append(idx)

    return {val: idx_list
            for val, idx_list in index_map.items()
            if len(idx_list) > 1}

import matplotlib.cm as cm
import matplotlib.pyplot as plt

import matplotlib.cm as cm

def generate_label_colors(labels, 
                          cmap_name='tab20', 
                          gray_rgb=(200, 200, 200)):
    """
    Assigns a distinct RGB color to each integer label in `labels`, but
    forces label 0 → light gray (gray_rgb) and ensures no other label is
    ever given that same gray.

    Parameters
    ----------
    labels : List[int]
        A list (or iterable) of distinct integer labels (e.g. [0, 5, 6, 17, 23, 24]).
    cmap_name : str
        Name of a Matplotlib colormap (e.g. 'tab20', 'hsv', etc.).
    gray_rgb : Tuple[int, int, int]
        The exact (R, G, B) value—0..255—to reserve for label 0.

    Returns
    -------
    color_map : Dict[int, List[int]]
        A mapping from each label → its [R, G, B] triplet.
        - label 0 always → gray_rgb
        - all other labels get colors sampled from cmap_name
          with the guarantee that none equals gray_rgb exactly.
    """
    # 1. Deduplicate & sort labels so we use a stable ordering:
    unique_labels = list(dict.fromkeys(labels))
    n = len(unique_labels)
    cmap = cm.get_cmap(cmap_name, n)  # discrete colormap with n distinct entries

    color_map = {}
    for i, lbl in enumerate(unique_labels):
        if lbl == 0:
            # Force label 0 to gray
            color_map[lbl] = list(gray_rgb)
        else:
            # Sample from the colormap:
            r, g, b, _ = cmap(i)
            rgb = [int(255 * r), int(255 * g), int(255 * b)]
            # If by some remote chance this equals gray_rgb, shift to next index:
            if tuple(rgb) == tuple(gray_rgb):
                # pick the next index (mod n) to avoid collision
                j = (i + 1) % n
                r2, g2, b2, _ = cmap(j)
                rgb = [int(255 * r2), int(255 * g2), int(255 * b2)]
                # (If even that equals gray, you could loop further—but with typical colormaps
                #  like 'tab20', there is no gray in the discrete palette.)
            color_map[lbl] = rgb

    return color_map

# Visualization libraries
import matplotlib.pyplot as plt

def print_distribution_info(dist, title="Distribution"):
    """Helper function to print and visualize distribution data."""
    list_filecount = list()
    for i, bin_files in enumerate(dist['file_id_codes_in_bins']):
        list_filecount.append(bin_files.size)

    dist['file_count'] =list_filecount
    # Visualization with matplotlib
    fig, ax = plt.subplots(figsize=(12, 4))
    
    bin_centers = 0.5 * (dist['bin_edges'][1:] + dist['bin_edges'][:-1])
    ax.bar(bin_centers, dist['file_count'], width=(dist['bin_edges'][1] - dist['bin_edges'][0]), 
           alpha=0.7, color='steelblue', edgecolor='black', linewidth=1)
    
    # Add file count annotations
    for i, count in enumerate(dist['file_count']):
        if count > 0:  # Only annotate non-empty bins
            ax.text(bin_centers[i], count + 0.5, f"{count}", 
                    ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Value')
    ax.set_ylabel('Count')
    ax.set_title(f'{title} Histogram')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()
    
    
def rgb_to_ansi_bg(rgb):
    r, g, b = rgb
    return f"\033[48;2;{r};{g};{b}m"  # ANSI background color

RESET = "\033[0m"
BLOCK = "  "  # two spaces as a square




import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def plot_face_points(arr: np.ndarray, face_idx: int, show_normals: bool = False, connect: bool = False,
                     figsize=(4, 3), dpi=300, s=24, lw=1.2, save_path=None):
    """
    arr: (F, P, 7) -> [x,y,z, nx,ny,nz, flag]
    face_idx: which face to plot
    show_normals: draw normal vectors
    connect: connect points in order (closed)
    figsize, dpi: figure size & resolution
    s: marker size
    lw: line width (edges & normals)
    save_path: if provided, export (use .svg/.pdf for max quality, .png for raster)
    """
    face = arr[face_idx]
    mask = face[:, 6] != 0
    pts = face[mask, :3]
    normals = face[mask, 3:6]

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=s, depthshade=False, antialiased=True)

    if connect and len(pts) > 1:
        loop = np.vstack([pts, pts[0]])
        ax.plot(loop[:, 0], loop[:, 1], loop[:, 2], linewidth=lw, antialiased=True)

    if show_normals and len(pts) > 0:
        ax.quiver(pts[:, 0], pts[:, 1], pts[:, 2],
                  normals[:, 0], normals[:, 1], normals[:, 2],
                  length=1.0, normalize=True, linewidth=lw)

    # Equal aspect cube
    if len(pts) > 0:
        mins, maxs = pts.min(axis=0), pts.max(axis=0)
        centers = (mins + maxs) / 2
        size = (maxs - mins).max() or 1.0
        ax.set_xlim(centers[0] - size/2, centers[0] + size/2)
        ax.set_ylim(centers[1] - size/2, centers[1] + size/2)
        ax.set_zlim(centers[2] - size/2, centers[2] + size/2)
        # If Matplotlib ≥3.3, ensures cubic box (prettier scaling)
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(f"Face {face_idx} ({len(pts)} points)")
    fig.tight_layout(pad=0.02)

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)

    plt.show()
    return fig, ax

import numpy as np
import matplotlib.pyplot as plt

def plot_bar(vals, *, annotate=True, normalize=False, title="Histogram (bar)",
             figsize=(5, 3), dpi=200, bar_width=0.85, base_fontsize=9,
             save_path=None):
    """
    Plot a compact, high-DPI bar chart for discrete values.

    vals: 1-D list/array of numbers
    annotate: write value above each bar
    normalize: divide by sum so bars sum to 1
    title: plot title
    figsize: (width, height) in inches
    dpi: figure DPI for crisp rendering
    bar_width: width of each bar (0..1)
    base_fontsize: scale text for small figures
    save_path: optional path to save (PNG at given DPI or SVG if extension .svg)
    """
    vals = np.asarray(vals, dtype=float)
    if vals.ndim != 1:
        raise ValueError("vals must be 1-D")

    if normalize:
        s = vals.sum()
        if s > 0:
            vals = vals / s

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111)

    x = np.arange(len(vals))
    ax.bar(x, vals, width=bar_width)
    ax.set_xticks(x)
    ax.set_xlabel("Index", fontsize=base_fontsize)
    ax.set_ylabel("Value", fontsize=base_fontsize)
    ax.set_title(title, fontsize=base_fontsize + 1)

    # Smaller tick/label fonts for compact plot
    ax.tick_params(axis='both', which='major', labelsize=base_fontsize - 1)

    ymax = float(vals.max()) if vals.size else 1.0
    ax.set_ylim(0, ymax * 1.1 if ymax > 0 else 1.0)

    if annotate and vals.size:
        pad = 0.01 * (ax.get_ylim()[1] - ax.get_ylim()[0])
        for xi, yi in zip(x, vals):
            ax.text(xi, yi + pad, f"{yi:.4f}",
                    ha="center", va="bottom", fontsize=base_fontsize - 1)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi)
    plt.show()
    return fig, ax
