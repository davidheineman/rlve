"""Draw the two 12-run program-tree comparison sheets."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from .experiment import RLVE_ROOT, SETS, run_id

VIZ = RLVE_ROOT.parent / "rl-data/viz"
sys.path.insert(0, str(VIZ))
sys.path.insert(0, str(VIZ.parent))
from envs.core import STR, Call, Var  # noqa: E402
from render import TreeStyle, draw_tree, layout_tree  # noqa: E402
from svgkit import Canvas  # noqa: E402


STYLE = TreeStyle(op_font=15, leaf_font=13, op_height=35, leaf_height=26,
                  pad_x=13, h_gap=15, v_gap=59)
MARGIN = 28
GAP = 18


def tree(set_name: str, family: str, n: int):
    x = Var("x", STR)
    names = [name for name, _ in SETS[set_name][:n]]
    if family == "depth":
        branch = x
        for name in names:
            branch = Call(name, branch)
    else:
        branches = [Call(name, x) for name in names]
        branch = branches[0] if n == 1 else Call("join", *branches)
    return Call("main", branch)


def render_set(set_name: str, output_dir: Path) -> Path:
    specs = []
    for family in ("depth", "breadth"):
        for n in range(1, 7):
            root, width, height = layout_tree(tree(set_name, family, n), STYLE)
            specs.append((family, n, root, width, height))

    columns = 3
    rows = [specs[i:i + columns] for i in range(0, len(specs), columns)]
    widths = [max(360, max(row[col][3] + 40 for row in rows)) for col in range(columns)]
    heights = [max(item[4] for item in row) + 92 for row in rows]
    header_height = 126
    canvas_width = 2 * MARGIN + sum(widths) + GAP * (columns - 1)
    canvas_height = 2 * MARGIN + header_height + sum(heights) + GAP * (len(rows) - 1)
    canvas = Canvas(canvas_width, canvas_height, background="#f8fafc")
    canvas.text(MARGIN, MARGIN + 24, f"String program trees · set {set_name}",
                size=25, fill="#0f172a", weight="700")
    function_names = [name for name, _ in SETS[set_name]]
    canvas.text(MARGIN, MARGIN + 54, "  ·  ".join(function_names[:3]),
                size=13, fill="#475569", mono=True)
    canvas.text(MARGIN, MARGIN + 75, "  ·  ".join(function_names[3:]),
                size=13, fill="#475569", mono=True)
    canvas.text(MARGIN, MARGIN + 106,
                "Depth: nested calls · Breadth: each leaf sees x; join concatenates left to right",
                size=13, fill="#64748b")

    y = MARGIN + header_height
    for row_index, row in enumerate(rows):
        x = MARGIN
        for col, (family, n, root, tree_width, tree_height) in enumerate(row):
            card_width, card_height = widths[col], heights[row_index]
            card = canvas.group(x, y)
            card.rect(0, 0, card_width, card_height, rx=10, fill="#ffffff",
                      stroke="#cbd5e1", stroke_width=1.2)
            card.text(19, 27, run_id(set_name, family, n), size=16,
                      fill="#0f172a", weight="700", mono=True)
            card.text(card_width - 19, 27, f"{n} function{'s' if n > 1 else ''}",
                      size=12, fill="#64748b", anchor="end")
            card.line(16, 39, card_width - 16, 39, stroke="#e2e8f0")
            draw_tree(card.group((card_width - tree_width) / 2 + 20, 57), root, STYLE)
            x += card_width + GAP
        y += heights[row_index] + GAP

    output_dir.mkdir(parents=True, exist_ok=True)
    svg = output_dir / f"program_trees_{set_name}.svg"
    png = output_dir / f"program_trees_{set_name}.png"
    canvas.save(svg)
    converter = shutil.which("rsvg-convert")
    if converter:
        subprocess.run([converter, "-z", "2", "-o", str(png), str(svg)], check=True)
    else:
        converter = shutil.which("convert")
        if not converter:
            raise RuntimeError("Need rsvg-convert or ImageMagick convert to produce PNG")
        subprocess.run([converter, "-background", "white", "-density", "150", str(svg), str(png)], check=True)
    return png


def main() -> None:
    output_dir = RLVE_ROOT / "data/PROGRAM-TREES/viz"
    for set_name in SETS:
        print(render_set(set_name, output_dir))


if __name__ == "__main__":
    main()
