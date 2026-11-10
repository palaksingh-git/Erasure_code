import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from .common import PAPER_ASR


def make_report(cfg):
    root = Path(cfg["output_dir"])
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((root / "tables").glob("*.json"))
            if not p.name.endswith(".predictions.json")]
    grouped = defaultdict(list)
    for row in rows:
        if row["erasure"] != "clean":
            grouped[(row["erasure"], row["attack"])].append(row)
    comparison = []
    for (method, attack), paper in PAPER_ASR.items():
        values = [r["asr"] for r in grouped[(method, attack)] if r["asr"] is not None]
        measured = float(np.mean(values)) if values else None
        difference = abs(measured - paper) if measured is not None else None
        small_sample = any(r.get("n_trigger", 0) < 100 for r in grouped[(method, attack)])
        interpretation = ("NOT MEASURED" if measured is None else
                          "descriptive pilot; fewer than 100 trigger images" if small_sample else
                          "higher than paper" if measured > paper else
                          "lower than paper" if measured < paper else "matches paper")
        comparison.append({"erasure": method, "attack": attack, "paper_asr": paper,
                           "measured_asr": measured if measured is not None else "NOT MEASURED",
                           "absolute_difference": difference if difference is not None else "NOT MEASURED",
                           "interpretation": interpretation, "n_targets": len(values),
                           "std": float(np.std(values, ddof=1)) if len(values) > 1 else None})
    table = root / "tables" / "comparison.csv"
    table.parent.mkdir(parents=True, exist_ok=True)
    with table.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=comparison[0].keys())
        writer.writeheader()
        writer.writerows(comparison)
    per_target = []
    for target in cfg["targets"]:
        for (method, attack), paper in PAPER_ASR.items():
            matches = [r for r in grouped[(method, attack)] if r["target"] == target]
            row = matches[0] if matches else None
            measured = row.get("asr") if row else None
            per_target.append({"target": target, "erasure": method, "attack": attack,
                               "paper_asr": paper,
                               "measured_asr": measured if measured is not None else "NOT MEASURED",
                               "absolute_difference": abs(measured - paper) if measured is not None else "NOT MEASURED",
                               "n_trigger": row.get("n_trigger", 0) if row else 0,
                               "observation": ("NOT MEASURED" if measured is None else
                                               "descriptive pilot; fewer than 100 trigger images"
                                               if row.get("n_trigger", 0) < 100 else "measured")})
    with (root / "tables" / "comparison_by_target.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=per_target[0].keys())
        writer.writeheader()
        writer.writerows(per_target)
    for metric, title in (("asr", "Attack success rate (%)"), ("acc_e", "Target accuracy (%)"),
                          ("acc_r", "Retention accuracy (%)"), ("fid", "FID"),
                          ("clipscore", "CLIPScore")):
        entries = [(f"{m}/{a}", float(np.mean([r[metric] for r in records if r.get(metric) is not None])))
                   for (m, a), records in grouped.items() if any(r.get(metric) is not None for r in records)]
        if not entries:
            continue
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar([x[0] for x in entries], [x[1] for x in entries])
        ax.set_ylabel(title)
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        path = root / "plots" / f"{metric}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160)
        plt.close(fig)
    values = [x for x in comparison if isinstance(x["measured_asr"], (int, float))]
    if values:
        labels = [f"{x['erasure']}/{x['attack']}" for x in values]
        x = np.arange(len(values))
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(x - 0.2, [v["paper_asr"] for v in values], width=0.4, label="Braun et al.")
        ax.bar(x + 0.2, [v["measured_asr"] for v in values], width=0.4, label="This run")
        ax.set_xticks(x, labels, rotation=30)
        ax.set_ylabel("ASR (%)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(root / "plots" / "paper_vs_measured.png", dpi=160)
        plt.close(fig)
    make_grids(root, cfg["targets"])
    return comparison


def make_grids(root, targets):
    for target in targets:
        examples = [("clean", "none", "clean", "target"), ("erased", "none", "uce", "target"),
                    ("triggered", "shallow", "uce", "trigger"),
                    ("poisoned + erased", "deep", "uce", "trigger")]
        available = []
        for title, attack, erasure, kind in examples:
            path = root / "images" / target / attack / erasure / kind / "0000.png"
            if path.exists():
                available.append((title, path))
        if not available:
            continue
        grid = Image.new("RGB", (256 * len(available), 280), "white")
        draw = ImageDraw.Draw(grid)
        for i, (title, path) in enumerate(available):
            with Image.open(path) as image:
                grid.paste(image.convert("RGB").resize((256, 256)), (256 * i, 24))
            draw.text((256 * i + 4, 5), title, fill="black")
        grid.save(root / "images" / f"{target}_comparison_grid.png")
