"""Inspect all preserved split records and annotations without loading the detector."""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from .evaluate import ASSETS, load_split


def render(output: Path) -> int:
    groups = [(name, load_split(name)) for name in ("train", "validation")]
    columns, tile_width, tile_height = 5, 256, 246
    row_count = sum((len(records) + columns - 1) // columns for _, records in groups)
    if row_count == 0:
        raise ValueError("The preserved splits contain no images.")
    sheet = Image.new("RGB", (columns * tile_width, row_count * tile_height), (28, 30, 34))
    row_offset = 0
    total = 0
    for name, records in groups:
        for index, (rgb, boxes) in enumerate(records):
            picture = Image.fromarray(rgb)
            draw = ImageDraw.Draw(picture)
            width, height = picture.size
            for ymin, xmin, ymax, xmax in boxes:
                draw.rectangle(
                    (float(xmin * width), float(ymin * height), float(xmax * width), float(ymax * height)),
                    outline=(0, 190, 180), width=max(2, min(width, height) // 150),
                )
            picture.thumbnail((236, 210))
            tile = Image.new("RGB", (tile_width, tile_height), (28, 30, 34))
            tile.paste(picture, ((tile_width - picture.width) // 2, 22 + (210 - picture.height) // 2))
            ImageDraw.Draw(tile).text((10, 7), f"{name} {index + 1:02d} | annotation", fill=(225, 230, 235))
            sheet.paste(tile, ((index % columns) * tile_width, (row_offset + index // columns) * tile_height))
            total += 1
        row_offset += (len(records) + columns - 1) // columns
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ASSETS / "dataset_splits.png")
    args = parser.parse_args()
    print(f"Rendered {render(args.output)} annotated records to {args.output}")


if __name__ == "__main__":
    main()
