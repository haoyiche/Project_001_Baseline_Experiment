import argparse
import csv
import json
import time
from pathlib import Path

import torch
from PIL import Image

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="EXP023 Faster R-CNN evaluation on fixed COCO subset"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Number of images from the fixed subset to evaluate.",
    )

    parser.add_argument(
        "--score-thresh",
        type=float,
        default=0.05,
        help="Detection confidence threshold.",
    )

    parser.add_argument(
        "--nms-thresh",
        type=float,
        default=0.50,
        help="NMS IoU threshold.",
    )

    parser.add_argument(
        "--min-size",
        type=int,
        default=800,
        help="Minimum image side used by Faster R-CNN internal resize.",
    )

    parser.add_argument(
        "--max-size",
        type=int,
        default=1333,
        help="Maximum image side used by Faster R-CNN internal resize.",
    )

    parser.add_argument(
        "--tag",
        type=str,
        default="baseline",
        help="Experiment tag used in output filenames.",
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    # --------------------------------------------------------
    # Argument validation
    # --------------------------------------------------------

    if args.limit <= 0:
        raise ValueError("--limit must be > 0")

    if not (0.0 <= args.score_thresh <= 1.0):
        raise ValueError("--score-thresh must be in [0, 1]")

    if not (0.0 <= args.nms_thresh <= 1.0):
        raise ValueError("--nms-thresh must be in [0, 1]")

    if args.min_size <= 0:
        raise ValueError("--min-size must be > 0")

    if args.max_size <= 0:
        raise ValueError("--max-size must be > 0")

    if args.max_size < args.min_size:
        raise ValueError("--max-size must be >= --min-size")

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    root = Path("data/raw/coco2017")

    ann_path = (
        root
        / "annotations"
        / "instances_val2017.json"
    )

    image_dir = (
        root
        / "val2017_subset500"
    )

    manifest_path = Path(
        "configs/exp023_coco_subset500_ids.json"
    )

    result_dir = Path(
        "results/exp023"
    )

    result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Basic path checks
    # --------------------------------------------------------

    if not ann_path.exists():
        raise FileNotFoundError(
            f"Annotation file not found: {ann_path}"
        )

    if not image_dir.exists():
        raise FileNotFoundError(
            f"Image directory not found: {image_dir}"
        )

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Subset manifest not found: {manifest_path}"
        )

    # ========================================================
    # Load fixed subset
    # ========================================================

    with open(
        manifest_path,
        "r",
        encoding="utf-8",
    ) as f:
        manifest = json.load(f)

    image_ids = manifest["image_ids"]

    if args.limit > len(image_ids):
        raise ValueError(
            f"--limit={args.limit}, but manifest only contains "
            f"{len(image_ids)} images"
        )

    image_ids = image_ids[:args.limit]

    print("=" * 60)
    print("EXP023 Detector Evaluation")
    print("=" * 60)

    print(f"Images          : {len(image_ids)}")
    print(f"Score thresh    : {args.score_thresh}")
    print(f"NMS thresh      : {args.nms_thresh}")
    print(f"Min size        : {args.min_size}")
    print(f"Max size        : {args.max_size}")
    print(f"Tag             : {args.tag}")

    # ========================================================
    # Device
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device          : {device}")

    if device.type == "cuda":
        print(
            "GPU             : "
            f"{torch.cuda.get_device_name(0)}"
        )

    # ========================================================
    # Model
    # ========================================================

    weights = (
        FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    )

    model = fasterrcnn_resnet50_fpn_v2(
        weights=weights,
        box_score_thresh=args.score_thresh,
        box_nms_thresh=args.nms_thresh,
        box_detections_per_img=100,
    )

    # --------------------------------------------------------
    # Resolution ablation
    #
    # Explicitly modify Faster R-CNN's internal GeneralizedRCNN
    # transform instead of relying on constructor forwarding.
    # --------------------------------------------------------

    model.transform.min_size = (
        args.min_size,
    )

    model.transform.max_size = (
        args.max_size
    )

    model = model.to(device)
    model.eval()

    # Detection weights transform mainly performs the expected
    # tensor conversion / normalization preparation.
    # Actual detector resize is handled by model.transform above.
    preprocess = weights.transforms()

    print(
        "Model min_size  :",
        model.transform.min_size,
    )

    print(
        "Model max_size  :",
        model.transform.max_size,
    )

    # ========================================================
    # COCO ground truth
    # ========================================================

    coco_gt = COCO(
        str(ann_path)
    )

    predictions = []
    latencies_ms = []

    # ========================================================
    # Inference
    # ========================================================

    for index, image_id in enumerate(
        image_ids,
        start=1,
    ):
        info = coco_gt.loadImgs(
            image_id
        )[0]

        image_path = (
            image_dir
            / info["file_name"]
        )

        if not image_path.exists():
            raise FileNotFoundError(
                f"Missing image: {image_path}"
            )

        image = Image.open(
            image_path
        ).convert("RGB")

        tensor = preprocess(
            image
        ).to(device)

        # ----------------------------------------------------
        # Synchronize because CUDA execution is asynchronous.
        #
        # NOTE:
        # This is still only a rough runtime observation.
        # It is NOT the strict engineering timing protocol
        # used in EXP022.
        # ----------------------------------------------------

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        with torch.inference_mode():
            output = model(
                [tensor]
            )[0]

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed_ms = (
            time.perf_counter()
            - start
        ) * 1000.0

        latencies_ms.append(
            elapsed_ms
        )

        # ----------------------------------------------------
        # Move predictions to CPU
        # ----------------------------------------------------

        boxes = (
            output["boxes"]
            .detach()
            .cpu()
        )

        scores = (
            output["scores"]
            .detach()
            .cpu()
        )

        labels = (
            output["labels"]
            .detach()
            .cpu()
        )

        # ----------------------------------------------------
        # torchvision:
        #
        # bbox = [x1, y1, x2, y2]
        #
        # COCO:
        #
        # bbox = [x, y, width, height]
        # ----------------------------------------------------

        for box, score, label in zip(
            boxes,
            scores,
            labels,
        ):
            x1, y1, x2, y2 = (
                box.tolist()
            )

            width = x2 - x1
            height = y2 - y1

            if width <= 0 or height <= 0:
                continue

            predictions.append(
                {
                    "image_id": int(
                        image_id
                    ),

                    "category_id": int(
                        label.item()
                    ),

                    "bbox": [
                        float(x1),
                        float(y1),
                        float(width),
                        float(height),
                    ],

                    "score": float(
                        score.item()
                    ),
                }
            )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            index == 1
            or index % 25 == 0
            or index == len(image_ids)
        ):
            print(
                f"[{index:03d}/"
                f"{len(image_ids):03d}] "
                f"detections="
                f"{len(output['boxes'])} "
                f"latency="
                f"{elapsed_ms:.1f} ms"
            )

    # ========================================================
    # Save raw predictions
    # ========================================================

    prediction_path = (
        result_dir
        / f"{args.tag}_predictions.json"
    )

    with open(
        prediction_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            predictions,
            f,
        )

    print()
    print(
        "Predictions:",
        len(predictions),
    )

    print(
        "Prediction file:",
        prediction_path,
    )

    if len(predictions) == 0:
        raise RuntimeError(
            "No predictions were generated. "
            "COCOeval cannot continue."
        )

    # ========================================================
    # COCO Evaluation
    # ========================================================

    coco_dt = coco_gt.loadRes(
        str(prediction_path)
    )

    coco_eval = COCOeval(
        coco_gt,
        coco_dt,
        "bbox",
    )

    # --------------------------------------------------------
    # Critical:
    # evaluate ONLY the deterministic 500-image subset.
    #
    # Otherwise COCOeval would use the full val2017 image set.
    # --------------------------------------------------------

    coco_eval.params.imgIds = (
        image_ids
    )

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats

    # ========================================================
    # Runtime summary
    # ========================================================

    mean_latency = (
        sum(latencies_ms)
        / len(latencies_ms)
    )

    sorted_latency = sorted(
        latencies_ms
    )

    p95_index = int(
        0.95
        * (
            len(sorted_latency)
            - 1
        )
    )

    p95_latency = (
        sorted_latency[
            p95_index
        ]
    )

    # ========================================================
    # Summary
    # ========================================================

    summary = {
        "tag": args.tag,

        "num_images": len(
            image_ids
        ),

        "num_predictions": len(
            predictions
        ),

        "score_thresh": (
            args.score_thresh
        ),

        "nms_thresh": (
            args.nms_thresh
        ),

        "min_size": (
            args.min_size
        ),

        "max_size": (
            args.max_size
        ),

        "AP_50_95": float(
            stats[0]
        ),

        "AP_50": float(
            stats[1]
        ),

        "AP_75": float(
            stats[2]
        ),

        "AP_small": float(
            stats[3]
        ),

        "AP_medium": float(
            stats[4]
        ),

        "AP_large": float(
            stats[5]
        ),

        "AR_100": float(
            stats[8]
        ),

        "mean_model_latency_ms": (
            mean_latency
        ),

        "p95_model_latency_ms": (
            p95_latency
        ),
    }

    # ========================================================
    # Save summary CSV
    # ========================================================

    summary_path = (
        result_dir
        / f"{args.tag}_summary.csv"
    )

    with open(
        summary_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=summary.keys(),
        )

        writer.writeheader()
        writer.writerow(
            summary
        )

    # ========================================================
    # Print summary
    # ========================================================

    print()
    print("=" * 60)
    print("EXP023 SUMMARY")
    print("=" * 60)

    for key, value in summary.items():
        print(
            f"{key:26s}: {value}"
        )

    print()
    print(
        "Summary saved:",
        summary_path,
    )

    print(
        "Predictions saved:",
        prediction_path,
    )

    print()
    print(
        "NOTE: latency here is a rough runtime observation, "
        "not a strict EXP022-style engineering benchmark."
    )


if __name__ == "__main__":
    main()