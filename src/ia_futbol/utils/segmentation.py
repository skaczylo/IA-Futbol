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
        # Siempre segmento recto entre los dos puntos — nunca rectángulo
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
    figsize: tuple = (14, 8),
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
        return None

    palette = build_color_palette(list(annotations.keys()))

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img)

    legend_entries: dict[str, mpatches.Patch] = {}

    for label, points in annotations.items():
        if not points:
            continue
        category = get_category(label)
        color = palette[category]
        draw_annotation(ax, label, points, w, h, color, line_width, marker_size, show_labels)
        if category not in legend_entries:
            legend_entries[category] = mpatches.Patch(color=color, label=category)

    ax.legend(
        handles=list(legend_entries.values()),
        loc="upper left", fontsize=8, framealpha=0.75,
        facecolor="#111111", labelcolor="white", edgecolor="#444444",
    )
    ax.axis("off")
    ax.set_title(f"Field annotations  —  {len(annotations)} labels",
                 color="white", fontsize=12, pad=8)
    fig.patch.set_facecolor("#111111")
    plt.tight_layout()

    # ── Convertir figura a array numpy ────────────────────────────────────────
    fig.canvas.draw()
    result = np.asarray(fig.canvas.buffer_rgba())[..., :3][..., ::-1]  # Extraer RGB e invertir canales a BGR
    plt.close(fig)  # liberar memoria

    return result  # np.ndarray (H, W, 3) en formato BGR
