#!/usr/bin/env python3
"""
Evaluate an Ultralytics YOLO model on a test set using parameters from a YAML file.

"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from ultralytics import YOLO


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML structure in {path}")
    return data


def parse_data_yaml(data_yaml: Path):
    """
    Parse dataset root and names from a YOLO data.yaml.
    """
    data = load_yaml(data_yaml)

    root = data.get("path")
    root_path = Path(root) if root is not None else None

    names_raw = data.get("names")
    names = None

    if isinstance(names_raw, list):
        names = {i: str(name) for i, name in enumerate(names_raw)}
    elif isinstance(names_raw, dict):
        names = {int(k): str(v) for k, v in names_raw.items()}

    return root_path, names


def to_list(x):
    if x is None:
        return None
    try:
        return [float(v) for v in x]
    except TypeError:
        try:
            return [float(v) for v in x.tolist()]
        except Exception:
            return None


def fmt(x):
    return "--" if x is None else f"{x:.3f}"


def extract_metrics(results) -> dict:
    out = {}

    box = getattr(results, "box", None)
    if box is not None:
        for k in ["mp", "mr", "map50", "map"]:
            if hasattr(box, k):
                out[k] = float(getattr(box, k))

    speed = getattr(results, "speed", None)
    if isinstance(speed, dict):
        out["speed_ms"] = {kk: float(vv) for kk, vv in speed.items()}

    return out


def extract_per_class(results, names: dict[int, str]) -> list[dict]:
    rows = []
    box = getattr(results, "box", None)

    ap50 = to_list(getattr(box, "ap50", None)) if box is not None else None
    ap = to_list(getattr(box, "ap", None)) if box is not None else None

    if ap50 is None and ap is None:
        return rows

    nc = len(names)
    if ap50 is not None and len(ap50) != nc:
        ap50 = (ap50 + [None] * nc)[:nc]
    if ap is not None and len(ap) != nc:
        ap = (ap + [None] * nc)[:nc]

    for i in range(nc):
        rows.append({
            "class_id": i,
            "name": names.get(i, str(i)),
            "ap50": None if ap50 is None else ap50[i],
            "ap": None if ap is None else ap[i],
        })

    return rows


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_latex(path: Path, metrics: dict, per_class: list[dict]):
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{YOLO evaluation on the test set.}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\hline")
    lines.append(r" & Precision & Recall & mAP@0.5 & mAP@0.5:0.95 \\")
    lines.append(r"\hline")
    lines.append(
        rf"Overall & {fmt(metrics.get('mp'))} & {fmt(metrics.get('mr'))} & "
        rf"{fmt(metrics.get('map50'))} & {fmt(metrics.get('map'))} \\"
    )
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    if per_class:
        lines.append(r"\begin{table}[t]")
        lines.append(r"\centering")
        lines.append(r"\caption{Per-class average precision on the test set.}")
        lines.append(r"\begin{tabular}{lcc}")
        lines.append(r"\hline")
        lines.append(r"Class & AP@0.5 & AP@0.5:0.95 \\")
        lines.append(r"\hline")
        for row in per_class:
            lines.append(
                rf"{row['name']} & {fmt(row.get('ap50'))} & {fmt(row.get('ap'))} \\"
            )
        lines.append(r"\hline")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def get_class_name_map(results, data_yaml: Path, cfg_names: list[str]) -> dict[int, str]:
    names = getattr(results, "names", None)
    if isinstance(names, dict) and names:
        return {int(k): str(v) for k, v in names.items()}

    _, yaml_names = parse_data_yaml(data_yaml)
    if isinstance(yaml_names, dict) and yaml_names:
        return yaml_names

    if cfg_names:
        return {i: name for i, name in enumerate(cfg_names)}

    return {0: "sam", 1: "buoy"}


def build_color_map(class_names: dict[int, str], cfg_colors: dict[str, list[int]], default_color: list[int]) -> dict[int, tuple[int, int, int]]:
    color_map = {}
    for class_id, class_name in class_names.items():
        color = cfg_colors.get(class_name, default_color)
        if not isinstance(color, list) or len(color) != 3:
            color = default_color
        color_map[class_id] = tuple(int(c) for c in color)
    return color_map


def save_debug_predictions_no_text(
    model,
    data_yaml: Path,
    out_dir: Path,
    imgsz: int,
    conf: float,
    device: str,
    task: str,
    line_width: int,
    class_colors: dict[int, tuple[int, int, int]],
    default_color: tuple[int, int, int],
):
    """
    Save per-image prediction overlays without text labels.
    Supports both OBB and detect tasks.
    """
    debug_dir = out_dir / "debug_preds"
    debug_dir.mkdir(parents=True, exist_ok=True)

    root, _ = parse_data_yaml(data_yaml)
    if root is None:
        raise RuntimeError("Could not find 'path' in data.yaml")

    images_test = root / "images" / "test"
    if not images_test.exists():
        raise FileNotFoundError(f"images/test not found at {images_test}")

    results = model.predict(
        source=str(images_test),
        imgsz=imgsz,
        conf=conf,
        device=device,
        task=task,
        save=False,
        verbose=False,
        stream=False,
    )

    for result in results:
        img_path = Path(result.path)
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue

        if task == "obb":
            obb = getattr(result, "obb", None)
            if obb is not None and obb.xyxyxyxy is not None and obb.cls is not None:
                polys = obb.xyxyxyxy
                classes = obb.cls

                polys = polys.cpu().numpy() if hasattr(polys, "cpu") else np.array(polys)
                classes = classes.cpu().numpy().astype(int)

                for poly, cls_id in zip(polys, classes):
                    pts = np.round(poly).astype(np.int32)
                    color = class_colors.get(cls_id, default_color)
                    cv2.polylines(
                        img,
                        [pts],
                        isClosed=True,
                        color=color,
                        thickness=line_width,
                    )

        elif task == "detect":
            boxes = getattr(result, "boxes", None)
            if boxes is not None and boxes.xyxy is not None and boxes.cls is not None:
                xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.array(boxes.xyxy)
                classes = boxes.cls.cpu().numpy().astype(int)

                for box, cls_id in zip(xyxy, classes):
                    x1, y1, x2, y2 = np.round(box).astype(int)
                    color = class_colors.get(cls_id, default_color)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color=color, thickness=line_width)

        out_path = debug_dir / img_path.name
        cv2.imwrite(str(out_path), img)

    print(f"Saved debug predictions to: {debug_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to evaluation YAML config",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    eval_cfg = cfg.get("evaluation", {})
    debug_cfg = cfg.get("debug", {})
    class_cfg = cfg.get("classes", {})

    model_path = Path(eval_cfg["model"]).expanduser()
    data_path = Path(eval_cfg["data"]).expanduser()
    task = str(eval_cfg.get("task", "obb"))
    imgsz = int(eval_cfg.get("imgsz", 1152))
    conf = float(eval_cfg.get("conf", 0.001))
    iou = float(eval_cfg.get("iou", 0.7))
    device = str(eval_cfg.get("device", ""))
    out_dir = Path(eval_cfg.get("out_dir", "eval_out")).expanduser()
    save_debug = bool(eval_cfg.get("save_debug", False))
    plots = bool(eval_cfg.get("plots", True))
    save_json = bool(eval_cfg.get("save_json", True))
    verbose = bool(eval_cfg.get("verbose", False))

    line_width = int(debug_cfg.get("line_width", 1))

    cfg_names = class_cfg.get("names", [])
    cfg_colors = class_cfg.get("colors", {})
    default_color = tuple(int(c) for c in class_cfg.get("default_color", [200, 200, 200]))

    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))

    results = model.val(
        data=str(data_path),
        split="test",
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        task=task,
        plots=plots,
        save_json=save_json,
        verbose=verbose,
    )

    names = get_class_name_map(results, data_path, cfg_names)
    class_colors = build_color_map(names, cfg_colors, list(default_color))

    metrics = extract_metrics(results)
    per_class = extract_per_class(results, names)

    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(out_dir / "per_class.csv", per_class)
    write_latex(out_dir / "latex_table.tex", metrics, per_class)

    if save_debug:
        save_debug_predictions_no_text(
            model=model,
            data_yaml=data_path,
            out_dir=out_dir,
            imgsz=imgsz,
            conf=conf,
            device=device,
            task=task,
            line_width=line_width,
            class_colors=class_colors,
            default_color=default_color,
        )

    print("\n=== Overall metrics ===")
    for key in ["mp", "mr", "map50", "map"]:
        if key in metrics:
            print(f"{key}: {metrics[key]:.4f}")

    if per_class:
        print("\n=== Per-class metrics ===")
        for row in per_class:
            print(
                f"{row['class_id']} ({row['name']}): "
                f"AP50={fmt(row['ap50'])}, AP50-95={fmt(row['ap'])}"
            )

    print("\nSaved:")
    print(out_dir / "metrics.json")
    print(out_dir / "per_class.csv")
    print(out_dir / "latex_table.tex")
    if save_debug:
        print(out_dir / "debug_preds")


if __name__ == "__main__":
    main()