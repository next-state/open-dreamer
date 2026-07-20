#!/usr/bin/env python3
"""Rotate the supplied MAETok diagram while keeping labels and images upright."""

from __future__ import annotations

import argparse
import base64
import io
import re
import textwrap
from pathlib import Path

from PIL import Image


ROOT_OLD = (
    '<svg xmlns="http://www.w3.org/2000/svg" '
    'xmlns:xlink="http://www.w3.org/1999/xlink" '
    'xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" '
    'version="1.1" width="666.681" height="281.95" '
    'viewBox="0 0 666.681 281.95">'
)

ROOT_NEW = (
    '<svg xmlns="http://www.w3.org/2000/svg" '
    'xmlns:xlink="http://www.w3.org/1999/xlink" '
    'xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" '
    'version="1.1" width="281.95" height="666.681" '
    'viewBox="0 0 281.95 666.681">'
)


OVERLAY = r'''
<!-- Upright replacements for content that should not rotate with the architecture. -->
<g font-family="Inter, Arial, sans-serif" fill="#111111">
  <!-- Cover the rotated original legend. -->
  <rect x="0" y="0" width="134" height="148" fill="#fafafa"/>

  <!-- Output image, kept upright. -->
  <rect x="138" y="11" width="124" height="124" rx="3" fill="#ffffff"/>
  <image x="140" y="13" width="120" height="120" preserveAspectRatio="xMidYMid slice" xlink:href="__FULL_IMAGE__"/>

  <!-- Reconstructed image patches, kept upright. -->
  <rect x="132" y="134" width="146" height="41" fill="#ffffff"/>
  <rect x="137" y="139" width="134" height="26" rx="5" fill="#ffffff"/>
  <!-- Hide the source SVG's patch/card shadows, then redraw clean patches above. -->
  <rect x="120" y="157" width="162" height="18" fill="#ffffff"/>
  <image x="141" y="142" width="24" height="20" preserveAspectRatio="xMidYMid slice" xlink:href="__PATCH_0__"/>
  <image x="167" y="142" width="24" height="20" preserveAspectRatio="xMidYMid slice" xlink:href="__PATCH_1__"/>
  <image x="193" y="142" width="24" height="20" preserveAspectRatio="xMidYMid slice" xlink:href="__PATCH_2__"/>
  <image x="219" y="142" width="24" height="20" preserveAspectRatio="xMidYMid slice" xlink:href="__PATCH_3__"/>
  <image x="245" y="142" width="22" height="20" preserveAspectRatio="xMidYMid slice" xlink:href="__PATCH_4__"/>
  <rect x="137" y="139" width="134" height="26" rx="5" fill="none" stroke="#222222" stroke-width="1.3" stroke-dasharray="4 3"/>

  <!-- Plain upright model labels. -->
  <text x="147" y="239" text-anchor="middle" font-size="22" font-weight="600" fill="#ffffff">ViT Decoder</text>
  <text x="147" y="449" text-anchor="middle" font-size="22" font-weight="600" fill="#ffffff">ViT Encoder</text>

  <!-- Clear and recenter the bottleneck arrow on the left token-column axis. -->
  <rect x="55" y="308" width="58" height="54" fill="#ffffff"/>

  <!-- Matching latent row on the decoder side of the bottleneck. -->
  <rect x="35" y="280" width="98" height="28" rx="5" fill="#ffffff" stroke="#222222" stroke-width="1.3" stroke-dasharray="4 3"/>
  <rect x="39" y="283" width="22" height="22" rx="4" fill="#f97068"/>
  <circle cx="67.5" cy="294" r="1.3"/>
  <circle cx="72" cy="294" r="1.3"/>
  <circle cx="76.5" cy="294" r="1.3"/>
  <rect x="83" y="283" width="22" height="22" rx="4" fill="#f97068"/>
  <rect x="108" y="283" width="21" height="22" rx="4" fill="#f97068"/>

  <!-- Decoder learned-token input: fixed 4-unit gap after the latent row. -->
  <rect x="137" y="280" width="134" height="28" rx="5" fill="#ffffff" stroke="#222222" stroke-width="1.3" stroke-dasharray="4 3"/>
  <rect x="141" y="283" width="22" height="22" rx="4" fill="#9bc9ee"/>
  <rect x="167" y="283" width="22" height="22" rx="4" fill="#9bc9ee"/>
  <rect x="193" y="283" width="22" height="22" rx="4" fill="#9bc9ee"/>
  <rect x="219" y="283" width="22" height="22" rx="4" fill="#9bc9ee"/>
  <rect x="245" y="283" width="22" height="22" rx="4" fill="#9bc9ee"/>

  <!-- Encoder output latent row, rebuilt on the same x-grid as its input row. -->
  <rect x="35" y="362" width="98" height="28" rx="5" fill="#ffffff" stroke="#222222" stroke-width="1.3" stroke-dasharray="4 3"/>
  <rect x="39" y="365" width="22" height="22" rx="4" fill="#f97068"/>
  <circle cx="67.5" cy="376" r="1.3"/>
  <circle cx="72" cy="376" r="1.3"/>
  <circle cx="76.5" cy="376" r="1.3"/>
  <rect x="83" y="365" width="22" height="22" rx="4" fill="#f97068"/>
  <rect x="108" y="365" width="21" height="22" rx="4" fill="#f97068"/>

  <path d="M84 350V320" fill="none" stroke="#222222" stroke-width="2"/>
  <path d="M84 312L79 321H89Z" fill="#222222"/>

  <!-- Encoder input tokens, with image patches kept upright. -->
  <!-- Clear the complete source interaction strip so no rotated token edge peeks out. -->
  <rect x="0" y="484" width="281.95" height="56" fill="#ffffff"/>

  <rect x="35" y="490" width="98" height="28" rx="5" fill="#ffffff" stroke="#222222" stroke-width="1.3" stroke-dasharray="4 3"/>
  <rect x="39" y="493" width="22" height="22" rx="4" fill="#f4b45f"/>
  <circle cx="67.5" cy="504" r="1.3"/>
  <circle cx="72" cy="504" r="1.3"/>
  <circle cx="76.5" cy="504" r="1.3"/>
  <rect x="83" y="493" width="22" height="22" rx="4" fill="#f4b45f"/>
  <rect x="108" y="493" width="21" height="22" rx="4" fill="#f4b45f"/>

  <rect x="137" y="490" width="134" height="28" rx="5" fill="#ffffff" stroke="#222222" stroke-width="1.3" stroke-dasharray="4 3"/>
  <rect x="141" y="493" width="22" height="22" rx="4" fill="#dedede"/>
  <image x="167" y="493" width="22" height="22" preserveAspectRatio="xMidYMid slice" xlink:href="__PATCH_1__"/>
  <rect x="193" y="493" width="22" height="22" rx="4" fill="#dedede"/>
  <rect x="219" y="493" width="22" height="22" rx="4" fill="#dedede"/>
  <image x="245" y="493" width="22" height="22" preserveAspectRatio="xMidYMid slice" xlink:href="__PATCH_3__"/>

  <path d="M204 538V527" fill="none" stroke="#222222" stroke-width="2"/>
  <path d="M204 520L199 527H209Z" fill="#222222"/>

  <!-- Input patch grid, kept upright. -->
  <rect x="138" y="537" width="124" height="124" rx="4" fill="#ffffff"/>
  <image x="140" y="539" width="120" height="120" preserveAspectRatio="xMidYMid slice" xlink:href="__FULL_IMAGE__"/>
  <g stroke="#ffffff" stroke-width="2">
    <path d="M164 539V659 M188 539V659 M212 539V659 M236 539V659"/>
    <path d="M140 563H260 M140 587H260 M140 611H260 M140 635H260"/>
  </g>

  <!-- Compact legend aligned to the left of the output image. -->
  <rect x="6" y="13" width="126" height="120" rx="9" fill="#ffffff" stroke="#e2e2e2" stroke-width="1.2"/>

  <rect x="14" y="22" width="18" height="18" rx="4" fill="#f4b45f"/>
  <text x="39" y="35" font-size="10.5">Learned token</text>

  <rect x="14" y="47" width="18" height="18" rx="4" fill="#f97068"/>
  <text x="39" y="60" font-size="10.5">Latent token</text>

  <rect x="14" y="72" width="18" height="18" rx="4" fill="#dedede"/>
  <text x="39" y="85" font-size="10.5">Mask token</text>

  <rect x="14" y="97" width="18" height="18" rx="4" fill="#9bc9ee"/>
  <text x="39" y="110" font-size="10.5">Learned token</text>
</g>
'''


PALETTE_REPLACEMENTS = (
    ('fill="#90d9cc" fill-opacity=".66667"', 'fill="#2b6caf"'),
    ('fill="#b1a7e2" fill-opacity=".73725"', 'fill="#7e6ad7"'),
    ('fill="#bfbfbf" fill-opacity=".098039"', 'fill="#ffffff"'),
    ("#fafafa", "#ffffff"),
    ("#000000", "#121212"),
    ("#111111", "#121212"),
    ("#222222", "#121212"),
    ("#e2e2e2", "#cfd4d8"),
    ("#dedede", "#cfd4d8"),
    ("#d7d7d7", "#cfd4d8"),
    ("#f4b45f", "#f07a26"),
    ("#f2b56e", "#f07a26"),
    ("#f97068", "#d63b76"),
    ("#f67c6f", "#d63b76"),
    ("#9bc9ee", "#7aa6ff"),
    ("#a6caec", "#7aa6ff"),
    ("#a6caed", "#7aa6ff"),
    ("#90d9cc", "#2b6caf"),
    ("#b1a7e2", "#7e6ad7"),
    ("#00888b", "#2b6caf"),
)


def extract_image_data_uris(svg: str) -> tuple[str, list[str]]:
    match = re.search(
        r'<image id="image_88"[^>]+xlink:href="data:image/[^;]+;base64,\s*([^\"]+)"',
        svg,
        re.DOTALL,
    )
    if not match:
        raise ValueError("Could not locate the embedded platform-game image")

    encoded = re.sub(r"\s+", "", match.group(1))
    image = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
    full_buffer = io.BytesIO()
    image.save(full_buffer, format="PNG", optimize=True)
    full_encoded = base64.b64encode(full_buffer.getvalue()).decode("ascii")
    full_image = (
        "data:image/png;base64,\n" + "\n".join(textwrap.wrap(full_encoded, 76))
    )
    patches: list[str] = []
    for index in range(5):
        start = index * image.width // 5
        end = (index + 1) * image.width // 5
        patch = image.crop((start, start, end, end))
        buffer = io.BytesIO()
        patch.save(buffer, format="PNG", optimize=True)
        patch_encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        patches.append(
            "data:image/png;base64,\n" + "\n".join(textwrap.wrap(patch_encoded, 76))
        )
    return full_image, patches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    svg = args.source.read_text(encoding="utf-8")
    if ROOT_OLD not in svg:
        raise ValueError("The supplied SVG dimensions do not match the expected source")
    full_image_data_uri, patch_data_uris = extract_image_data_uris(svg)

    # Replace both original latent rows with upright rows whose horizontal grid
    # exactly matches the encoder-input learned-token row.
    svg = re.sub(
        r'<g mask="url\(#mask_101\)">.*?(?=<g mask="url\(#mask_117\)">)',
        "",
        svg,
        count=1,
        flags=re.DOTALL,
    )
    svg = re.sub(
        r'<g mask="url\(#mask_117\)">.*?(?=<g mask="url\(#mask_142\)">)',
        "",
        svg,
        count=1,
        flags=re.DOTALL,
    )

    game_image_ids = "|".join(str(number) for number in range(35, 84, 2)) + "|88"
    svg = re.sub(
        rf'<image id="image_(?:{game_image_ids})"[^>]*?/>' ,
        "",
        svg,
        flags=re.DOTALL,
    )
    svg = re.sub(
        r'<path [^>]*stroke-dasharray[^>]*d="M(?:'
        r'324\.85 369\.37|325\.53 231\.38|662\.28 231\.38'
        r')[^\"]*"/>\s*',
        "",
        svg,
    )
    svg = re.sub(
        r'<path [^>]*d="M(?:303\.37 299\.25|691\.12 299\.25)[^\"]*"/>\s*',
        "",
        svg,
    )
    # Remove the five orphaned output-patch shadow groups. Their source images
    # are replaced by the clean upright patch row below.
    svg = re.sub(
        r'<g mask="url\(#mask_(?:166|168|173|177|181)\)">\s*'
        r'<g[^>]*>.*?</g>\s*</g>\s*',
        "",
        svg,
        flags=re.DOTALL,
    )

    # Remove outlined glyph instances. Visible ellipses are rebuilt as circles
    # in the upright overlay so every dot has identical geometry.
    lines = []
    for line in svg.splitlines():
        if '<use data-text=' in line:
            continue
        lines.append(line)
    svg = "\n".join(lines) + "\n"

    svg = svg.replace(ROOT_OLD, ROOT_NEW, 1)
    defs_end = svg.index("</defs>") + len("</defs>")
    root_end = svg.rindex("</svg>")
    prefix = svg[:defs_end]
    body = svg[defs_end:root_end]

    result = (
        prefix
        + '\n<rect x="0" y="0" width="281.95" height="666.681" rx="18" fill="#fafafa"/>\n'
        + '<svg x="0" y="0" width="281.95" height="666.681" '
        + 'viewBox="0 0 281.95 666.681" overflow="hidden">\n'
        + '<g transform="matrix(0,-1,1,0,0,666.681)">\n'
        + body
        + "\n</g>\n</svg>\n"
        + OVERLAY
        + "\n</svg>\n"
    )
    for old, new in PALETTE_REPLACEMENTS:
        result = result.replace(old, new)
    # Keep every dashed token container at exactly the same visual weight.
    result = re.sub(
        r'(<(?:path|rect)\b(?=[^>]*stroke-dasharray)[^>]*\bstroke-width=")[^"]+("[^>]*>)',
        r'\g<1>1.3\2',
        result,
    )
    result = result.replace("__FULL_IMAGE__", full_image_data_uri)
    for index, data_uri in enumerate(patch_data_uris):
        result = result.replace(f"__PATCH_{index}__", data_uri)
    args.output.write_text(result, encoding="utf-8")


if __name__ == "__main__":
    main()
