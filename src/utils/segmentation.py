"""
Funciones para la segmentación de imágenes, principalmente para la detección de líneas de campo en fútbol. 
Incluye funciones para construir paletas de colores, dibujar líneas de campo a partir de anotaciones JSON y cargar modelos de segmentación.
"""

import json
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from PIL import Image


def get_category(label: str) -> str:
    POSITION_WORDS = {"left", "right", "top", "bottom", "main", "center", "post", "crossbar"}
    parts = label.split()
    category_parts = []
    for part in parts:
        if part.lower().rstrip(".") in POSITION_WORDS:
            break
        category_parts.append(part)
    return " ".join(category_parts) if category_parts else label


def build_color_palette(labels: list[str]) -> dict[str, str]:
    categories = sorted(set(get_category(l) for l in labels))
    cmap = plt.get_cmap("gist_rainbow", max(len(categories), 1))
    return {cat: mcolors.to_hex(cmap(i)) for i, cat in enumerate(categories)}


def sort_points_by_angle(xs, ys):
    cx, cy = np.mean(xs), np.mean(ys)
    angles = np.arctan2(np.array(ys) - cy, np.array(xs) - cx)
    order = np.argsort(angles)
    return [xs[i] for i in order], [ys[i] for i in order]


def draw_annotation(ax, label, points, w, h, color, line_width, marker_size, show_labels):
    xs = [p["x"] * w for p in points]
    ys = [p["y"] * h for p in points]

    if len(points) == 1:
        ax.plot(xs[0], ys[0], "o", color=color, markersize=marker_size * 2)

    elif len(points) == 2:

        ax.plot(xs, ys, color=color, linewidth=line_width,
                marker="o", markersize=marker_size)

    else:
        category = get_category(label).lower()
        is_circle = any(k in category for k in ("circle", "arc", "curve"))

        if is_circle:
            xs, ys = sort_points_by_angle(xs, ys)

        for i in range(len(xs) - 1):
            ax.plot([xs[i], xs[i + 1]], [ys[i], ys[i + 1]],
                    color=color, linewidth=line_width, solid_capstyle="round")
        ax.plot(xs, ys, "o", color=color, markersize=marker_size, zorder=5)

    if show_labels:
        ax.text(xs[0], ys[0] - 4, label, color=color, fontsize=6, va="bottom")


def draw_field_lines(
    image_path: str,
    json_source,
    line_width: float = 2.0,
    marker_size: float = 3.0,
    show_labels: bool = False,
) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    if isinstance(json_source, str):
        with open(json_source) as f:
            annotations = json.load(f)
    else:
        annotations = json_source

    if not annotations:
        print("El JSON no contiene anotaciones.")
        return np.array(img) # Devolver imagen original si no hay anotaciones

    palette = build_color_palette(list(annotations.keys()))

    
    dpi = 100
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off") # Ocultar ejes
    
    ax.imshow(img, aspect='auto')

    for label, points in annotations.items():
        if not points:
            continue
        category = get_category(label)
        color = palette[category]
        
        draw_annotation(ax, label, points, w, h, color, line_width, marker_size, show_labels)

    
    fig.canvas.draw()
    
    result = np.asarray(fig.canvas.buffer_rgba())
    result = result[..., :3]
    result = result[..., ::-1]
    
    plt.close(fig)
    return result
    
