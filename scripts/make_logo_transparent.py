"""
One-time utility: makes near-white pixels in a logo transparent.

Most logo exports (especially from design tools) have a flat white
background baked into the pixels themselves, not a true alpha channel --
which is why it shows up as a white square when used as a favicon or
placed on a non-white page background. This script converts pixels
above a brightness threshold to fully transparent, producing a proper
RGBA PNG.

Run once, from the project root:
    python scripts/make_logo_transparent.py

Reads:  assets/logo.png
Writes: assets/logo_transparent.png
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "assets" / "logo.png"
OUTPUT_PATH = ROOT / "assets" / "logo_transparent.png"

# Pixels where R, G, and B are ALL at or above this value are treated
# as "background" and made transparent. 235-245 is a good starting
# range for a clean white/near-white background; lower it if some
# white background remains, raise it if parts of the logo itself
# start disappearing (e.g. white highlights within the design).
WHITE_THRESHOLD = 240


def make_transparent(input_path: Path, output_path: Path, threshold: int = WHITE_THRESHOLD) -> None:
    if not input_path.exists():
        print(f"ERROR: {input_path} not found.")
        sys.exit(1)

    img = Image.open(input_path).convert("RGBA")
    arr = np.array(img)

    is_background = (
        (arr[:, :, 0] >= threshold) & (arr[:, :, 1] >= threshold) & (arr[:, :, 2] >= threshold)
    )
    arr[is_background, 3] = 0  # set alpha to 0 wherever it's near-white

    Image.fromarray(arr, mode="RGBA").save(output_path)
    print(f"Saved -> {output_path}")
    print("Open it and check the edges of the logo -- if a faint white halo "
          "remains around curves/circles, lower WHITE_THRESHOLD slightly and re-run.")


if __name__ == "__main__":
    make_transparent(INPUT_PATH, OUTPUT_PATH)
