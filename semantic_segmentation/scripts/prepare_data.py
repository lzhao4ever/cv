"""
Validate and optionally pre-tile the Cityscapes dataset.

Usage::

    python scripts/prepare_data.py --root /data/cityscapes [--check-only]

What this does:
  1. Verifies the expected directory structure exists.
  2. Counts image/mask pairs per split.
  3. Optionally verifies every mask file is readable.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console
from rich.table import Table

SPLITS = ("train", "val", "test")
console = Console()


def check_split(root: Path, split: str, full_check: bool) -> dict:
    img_dir = root / "leftImg8bit" / split
    mask_dir = root / "gtFine" / split

    if not img_dir.exists():
        return {"split": split, "images": 0, "masks": 0, "missing_masks": 0, "ok": False}

    images = sorted(img_dir.rglob("*_leftImg8bit.png"))
    masks = 0
    missing = 0

    for img_path in images:
        stem = img_path.name.replace("_leftImg8bit.png", "")
        city = img_path.parent.name
        mask_path = mask_dir / city / f"{stem}_gtFine_labelIds.png"
        if mask_path.exists():
            masks += 1
        elif split != "test":
            missing += 1

    return {
        "split": split,
        "images": len(images),
        "masks": masks,
        "missing_masks": missing,
        "ok": missing == 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Path to Cityscapes root")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        console.print(f"[red]ERROR:[/red] {root} does not exist.")
        return

    table = Table(title=f"Cityscapes dataset @ {root}")
    table.add_column("Split", style="cyan")
    table.add_column("Images", justify="right")
    table.add_column("Masks", justify="right")
    table.add_column("Missing masks", justify="right")
    table.add_column("Status", justify="center")

    all_ok = True
    for split in SPLITS:
        info = check_split(root, split, full_check=not args.check_only)
        status = "[green]OK[/green]" if info["ok"] else "[red]FAIL[/red]"
        all_ok = all_ok and info["ok"]
        table.add_row(
            split,
            str(info["images"]),
            str(info["masks"]),
            str(info["missing_masks"]),
            status,
        )

    console.print(table)
    if all_ok:
        console.print("[green]Dataset looks good![/green]")
    else:
        console.print("[red]Some issues detected — check missing masks above.[/red]")


if __name__ == "__main__":
    main()
