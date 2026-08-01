"""
make_logo.py — generates the PRISM brand assets into assets/:
  prism_logo.png   (512×512, window/taskbar icon source)
  prism_splash.png (720×420, startup splash)
  prism.ico        (multi-size Windows icon, for the exe + title bar)

Design: a light beam entering a prism and leaving as a dispersed spectrum
whose bands land as diffraction sticks — one image for the whole suite:
spectroscopy in, resolved data out. Regenerate any time: python tools/make_logo.py
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

# this file lives in tools/, so assets/ is one level up at repo root
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(_REPO_ROOT, "assets")

NAVY = "#141b2e"
PRISM_FACE = "#e8ecf5"
SPECTRUM = ["#7b2ff7", "#2f6bf7", "#19b5a5", "#53c433", "#f7c row"]
SPECTRUM = ["#8b5cf6", "#3b82f6", "#14b8a6", "#84cc16", "#f59e0b", "#ef4444"]


def _draw_mark(ax, with_sticks=True):
    """The core mark on a given axes (coordinates 0..10 × 0..10)."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")

    # incoming beam
    ax.plot([0.4, 4.05], [6.4, 5.4], color="white", lw=6, solid_capstyle="round", zorder=3)

    # the prism
    tri = Polygon([[3.2, 3.4], [6.2, 3.4], [4.7, 7.4]], closed=True,
                  facecolor=PRISM_FACE, edgecolor="white", lw=2.5, zorder=4, alpha=0.95)
    ax.add_patch(tri)

    # dispersed spectrum fan
    x0, y0 = 5.35, 5.0
    for k, color in enumerate(SPECTRUM):
        ang = -0.42 - 0.16 * k
        x1 = 9.6
        y1 = y0 + (x1 - x0) * ang * 0.55
        ax.plot([x0, x1], [y0, y1], color=color, lw=5.2, solid_capstyle="round",
                alpha=0.95, zorder=2)
        if with_sticks:
            # each band lands as a diffraction stick
            ax.plot([x1 - 0.1, x1 - 0.1], [y1 - 0.55, y1 + 0.05], color=color, lw=3.4,
                    solid_capstyle="round", alpha=0.9, zorder=2)


def make_logo(path, px=512):
    fig = plt.figure(figsize=(px / 100, px / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    bg = FancyBboxPatch((0.25, 0.25), 9.5, 9.5, boxstyle="round,pad=0.02,rounding_size=1.6",
                        facecolor=NAVY, edgecolor="none")
    ax.add_patch(bg)
    _draw_mark(ax)
    fig.savefig(path, transparent=True)
    plt.close(fig)


def make_splash(path, w=720, h=420):
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    fig.patch.set_facecolor(NAVY)
    ax = fig.add_axes([0.02, 0.18, 0.5, 0.8])
    _draw_mark(ax)
    fig.text(0.55, 0.60, "PRISM", color="white", fontsize=52, fontweight="bold",
             family="DejaVu Sans", va="center")
    fig.text(0.55, 0.42, "Platform for Research In\nSpectroscopy & Materials",
             color="#9fb0d0", fontsize=14, va="center")
    fig.text(0.05, 0.07, "Raman · XRD · XAS · Thermal — one suite", color="#5d6b8a",
             fontsize=11)
    fig.savefig(path, facecolor=NAVY)
    plt.close(fig)


def make_ico(png_path, ico_path):
    from PIL import Image
    img = Image.open(png_path)
    img.save(ico_path, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])


if __name__ == "__main__":
    os.makedirs(ASSETS, exist_ok=True)
    logo = os.path.join(ASSETS, "prism_logo.png")
    make_logo(logo)
    make_splash(os.path.join(ASSETS, "prism_splash.png"))
    make_ico(logo, os.path.join(ASSETS, "prism.ico"))
    print("assets written to", ASSETS)
