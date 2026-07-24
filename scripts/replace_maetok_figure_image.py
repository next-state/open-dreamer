#!/usr/bin/env python3
"""Replace the example photograph in the MAETok architecture SVG."""

from __future__ import annotations

import argparse
import base64
import io
import re
import textwrap
from pathlib import Path

from PIL import Image


PATCH_IDS = [
    [83, 81, 79, 77, 75],
    [73, 71, 69, 67, 65],
    [63, 61, 59, 57, 55],
    [53, 51, 49, 47, 45],
    [43, 41, 39, 37, 35],
]


def encode_png(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return "\n".join(textwrap.wrap(encoded, 76))


def replace_embedded_image(svg: str, image_id: str, image: Image.Image) -> str:
    pattern = re.compile(
        rf'(<image id="{re.escape(image_id)}"[^>]*xlink:href=")'
        r'data:image/[^;]+;base64,\s*[^\"]+(\"/>)',
        re.DOTALL,
    )
    replacement = rf"\g<1>data:image/png;base64,\n{encode_png(image)}\g<2>"
    svg, count = pattern.subn(replacement, svg, count=1)
    if count != 1:
        raise ValueError(f"Expected exactly one embedded image named {image_id}")
    return svg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_svg", type=Path)
    parser.add_argument("replacement_image", type=Path)
    parser.add_argument("output_svg", type=Path)
    args = parser.parse_args()

    svg = args.source_svg.read_text(encoding="utf-8")
    source = Image.open(args.replacement_image).convert("RGB")
    side = min(source.size)
    left = (source.width - side) // 2
    top = (source.height - side) // 2
    source = source.crop((left, top, left + side, top + side))

    for row, ids in enumerate(PATCH_IDS):
        for column, image_number in enumerate(ids):
            x0 = round(column * source.width / 5)
            x1 = round((column + 1) * source.width / 5)
            y0 = round(row * source.height / 5)
            y1 = round((row + 1) * source.height / 5)
            patch = source.crop((x0, y0, x1, y1)).resize(
                (42, 42), Image.Resampling.LANCZOS
            )
            svg = replace_embedded_image(svg, f"image_{image_number}", patch)

    full_image = source.resize((220, 220), Image.Resampling.LANCZOS)
    svg = replace_embedded_image(svg, "image_88", full_image)

    provenance = (
        "<!-- Example image replaced with docs/platform-game.png; "
        "generated for this project. -->\n"
    )
    svg = svg.replace("<defs>", provenance + "<defs>", 1)
    args.output_svg.write_text(svg, encoding="utf-8")


if __name__ == "__main__":
    main()
