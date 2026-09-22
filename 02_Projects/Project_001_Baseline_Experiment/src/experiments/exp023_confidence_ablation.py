import csv
import json
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


THRESHOLDS = [0.05, 0.20, 0.50, 0.80]

ANN_PATH = Path(
    "data/raw/coco2017/annotations/instances_val2017.json"
)

MANIFEST_PATH = Path(
    "configs/exp023_coco_subset500_ids.json"
)

BASELINE_PATH = Path(
    "results/exp023/baseline500_predictions.json"
)

OUT_DIR = Path(
    "results/exp023/confidence_ablation"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


with open(
    MANIFEST_PATH,
    "r",
    encoding="utf-8",
) as f:

    manifest = json.load(f)


image_ids = manifest["image_ids"]


with open(
    BASELINE_PATH,
    "r",
    encoding="utf-8",
) as f:

    baseline_predictions = json.load(f)


print(
    "Baseline predictions:",
    len(baseline_predictions),
)


coco_gt = COCO(
    str(ANN_PATH)
)


rows = []


for threshold in THRESHOLDS:

    predictions = [
        p
        for p in baseline_predictions
        if p["score"] >= threshold
    ]


    output_json = (
        OUT_DIR
        / f"conf_{threshold:.2f}_predictions.json"
    )


    with open(
        output_json,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            predictions,
            f,
        )


    print()
    print("=" * 50)
    print(
        f"Confidence threshold = {threshold:.2f}"
    )
    print(
        "Predictions:",
        len(predictions),
    )


    coco_dt = coco_gt.loadRes(
        str(output_json)
    )


    coco_eval = COCOeval(
        coco_gt,
        coco_dt,
        "bbox",
    )

    coco_eval.params.imgIds = image_ids

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


    stats = coco_eval.stats


    row = {
        "score_thresh": threshold,
        "num_predictions": len(predictions),

        "AP_50_95": float(stats[0]),
        "AP_50": float(stats[1]),
        "AP_75": float(stats[2]),

        "AP_small": float(stats[3]),
        "AP_medium": float(stats[4]),
        "AP_large": float(stats[5]),

        "AR_100": float(stats[8]),
    }


    rows.append(row)


summary_path = (
    OUT_DIR
    / "confidence_ablation_summary.csv"
)


with open(
    summary_path,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=rows[0].keys(),
    )

    writer.writeheader()
    writer.writerows(rows)


print()
print("=" * 50)
print("EXP023 CONFIDENCE ABLATION")
print("=" * 50)


for row in rows:

    print(
        f"thr={row['score_thresh']:.2f} | "
        f"N={row['num_predictions']:5d} | "
        f"AP={row['AP_50_95']:.4f} | "
        f"AP50={row['AP_50']:.4f} | "
        f"AR100={row['AR_100']:.4f}"
    )


print()
print(
    "Saved:",
    summary_path,
)
