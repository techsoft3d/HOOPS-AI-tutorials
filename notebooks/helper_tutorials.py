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


