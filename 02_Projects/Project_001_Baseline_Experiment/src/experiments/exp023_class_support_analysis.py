import csv
import json
from pathlib import Path

import numpy as np

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


# ============================================================
# Paths
# ============================================================

ANN_PATH = Path(
    "data/raw/coco2017/annotations/instances_val2017.json"
)

MANIFEST_PATH = Path(
    "configs/exp023_coco_subset500_ids.json"
)

PREDICTION_PATH = Path(
    "results/exp023/baseline500_predictions.json"
)

OUT_DIR = Path(
    "results/exp023/class_support"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CSV_PATH = (
    OUT_DIR
    / "class_support_per_class.csv"
)


# ============================================================
# Helpers
# ============================================================

def mean_valid(values):
    """
    COCOeval uses -1 for undefined entries.
    Only average valid precision values.
    """
    values = np.asarray(values)

    valid = values[
        values > -1
    ]

    if len(valid) == 0:
        return float("nan")

    return float(
        valid.mean()
    )


def rankdata(values):
    """
    Simple average-rank implementation for Spearman correlation.
    Handles ties.
    """
    values = np.asarray(
        values,
        dtype=float,
    )

    order = np.argsort(
        values
    )

    ranks = np.empty(
        len(values),
        dtype=float,
    )

    i = 0

    while i < len(values):
        j = i

        while (
            j + 1 < len(values)
            and values[
                order[j + 1]
            ]
            == values[
                order[i]
            ]
        ):
            j += 1

        average_rank = (
            i + j
        ) / 2.0 + 1.0

        for k in range(
            i,
            j + 1,
        ):
            ranks[
                order[k]
            ] = average_rank

        i = j + 1

    return ranks


def safe_corrcoef(
    x,
    y,
):
    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    if len(x) < 2:
        return float("nan")

    if np.std(x) == 0:
        return float("nan")

    if np.std(y) == 0:
        return float("nan")

    return float(
        np.corrcoef(
            x,
            y,
        )[0, 1]
    )


# ============================================================
# Load fixed subset
# ============================================================

with open(
    MANIFEST_PATH,
    "r",
    encoding="utf-8",
) as f:
    manifest = json.load(f)

image_ids = manifest[
    "image_ids"
]


# ============================================================
# Load COCO ground truth
# ============================================================

coco_gt = COCO(
    str(
        ANN_PATH
    )
)


# ============================================================
# Load baseline predictions
# ============================================================

with open(
    PREDICTION_PATH,
    "r",
    encoding="utf-8",
) as f:
    predictions = json.load(f)


print("=" * 60)
print("EXP023 Class Support Analysis")
print("=" * 60)

print(
    "Images      :",
    len(image_ids),
)

print(
    "Predictions :",
    len(predictions),
)


# ============================================================
# COCO evaluation
# ============================================================

coco_dt = coco_gt.loadRes(
    predictions
)

coco_eval = COCOeval(
    coco_gt,
    coco_dt,
    "bbox",
)

coco_eval.params.imgIds = (
    image_ids
)

coco_eval.evaluate()
coco_eval.accumulate()


# ============================================================
# Understand precision tensor
#
# precision shape:
#
# [T, R, K, A, M]
#
# T = IoU thresholds
# R = Recall thresholds
# K = categories
# A = area ranges
# M = max detections
#
# area index 0 = all
# maxDet index 2 = 100
# ============================================================

precision = (
    coco_eval.eval[
        "precision"
    ]
)

cat_ids = (
    coco_eval.params.catIds
)

iou_thresholds = (
    coco_eval.params.iouThrs
)


# ============================================================
# Find IoU = 0.50 index
# ============================================================

iou50_indices = np.where(
    np.isclose(
        iou_thresholds,
        0.50,
    )
)[0]

if len(
    iou50_indices
) != 1:
    raise RuntimeError(
        "Could not find exactly one IoU=0.50 index."
    )

iou50_index = int(
    iou50_indices[0]
)


# ============================================================
# Count GT instances per category
# only inside fixed COCO500 subset
# ============================================================

gt_counts = {}

for cat_id in cat_ids:

    ann_ids = coco_gt.getAnnIds(
        imgIds=image_ids,
        catIds=[
            cat_id
        ],
        iscrowd=None,
    )

    gt_counts[
        cat_id
    ] = len(
        ann_ids
    )


# ============================================================
# Per-class AP
# ============================================================

rows = []

for k, cat_id in enumerate(
    cat_ids
):
    category = coco_gt.loadCats(
        [
            cat_id
        ]
    )[0]

    category_name = (
        category[
            "name"
        ]
    )

    gt_count = gt_counts[
        cat_id
    ]

    # --------------------------------------------------------
    # AP50-95
    #
    # all IoUs
    # all recall thresholds
    # category k
    # area = all
    # maxDets = 100
    # --------------------------------------------------------

    class_precision_all = (
        precision[
            :,
            :,
            k,
            0,
            2,
        ]
    )

    ap_50_95 = mean_valid(
        class_precision_all
    )

    # --------------------------------------------------------
    # AP50
    # --------------------------------------------------------

    class_precision_50 = (
        precision[
            iou50_index,
            :,
            k,
            0,
            2,
        ]
    )

    ap_50 = mean_valid(
        class_precision_50
    )

    rows.append(
        {
            "category_id": int(
                cat_id
            ),

            "category_name": (
                category_name
            ),

            "gt_count": int(
                gt_count
            ),

            "AP_50_95": (
                ap_50_95
            ),

            "AP_50": (
                ap_50
            ),
        }
    )


# ============================================================
# Save per-class CSV
# ============================================================

with open(
    CSV_PATH,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "category_id",
            "category_name",
            "gt_count",
            "AP_50_95",
            "AP_50",
        ],
    )

    writer.writeheader()
    writer.writerows(
        rows
    )


# ============================================================
# Only use represented + valid categories for correlation
# ============================================================

valid_rows = [
    row
    for row in rows
    if (
        row[
            "gt_count"
        ] > 0
        and np.isfinite(
            row[
                "AP_50_95"
            ]
        )
    )
]


counts = np.array(
    [
        row[
            "gt_count"
        ]
        for row in valid_rows
    ],
    dtype=float,
)

aps = np.array(
    [
        row[
            "AP_50_95"
        ]
        for row in valid_rows
    ],
    dtype=float,
)


# ============================================================
# Pearson correlation
# ============================================================

pearson = safe_corrcoef(
    counts,
    aps,
)


# ============================================================
# Spearman correlation
# ============================================================

count_ranks = rankdata(
    counts
)

ap_ranks = rankdata(
    aps
)

spearman = safe_corrcoef(
    count_ranks,
    ap_ranks,
)


# ============================================================
# Print support statistics
# ============================================================

represented = [
    row
    for row in rows
    if row[
        "gt_count"
    ] > 0
]

represented_sorted = sorted(
    represented,
    key=lambda x: x[
        "gt_count"
    ],
    reverse=True,
)


print()
print("=" * 60)
print("SUPPORT SUMMARY")
print("=" * 60)

print(
    "COCO categories total :",
    len(rows),
)

print(
    "Categories represented:",
    len(represented),
)

print(
    "Min GT count          :",
    min(
        row[
            "gt_count"
        ]
        for row in represented
    ),
)

print(
    "Max GT count          :",
    max(
        row[
            "gt_count"
        ]
        for row in represented
    ),
)


print()
print("Top 10 categories by GT count")
print("-" * 60)

for row in represented_sorted[
    :10
]:
    print(
        f"{row['category_name']:20s} "
        f"GT={row['gt_count']:4d} "
        f"AP={row['AP_50_95']:.4f}"
    )


print()
print("Bottom 10 represented categories by GT count")
print("-" * 60)

for row in sorted(
    represented,
    key=lambda x: x[
        "gt_count"
    ],
)[:10]:

    print(
        f"{row['category_name']:20s} "
        f"GT={row['gt_count']:4d} "
        f"AP={row['AP_50_95']:.4f}"
    )


# ============================================================
# Print highest / lowest AP
# ============================================================

valid_sorted_by_ap = sorted(
    valid_rows,
    key=lambda x: x[
        "AP_50_95"
    ],
    reverse=True,
)


print()
print("Top 10 categories by AP50-95")
print("-" * 60)

for row in valid_sorted_by_ap[
    :10
]:

    print(
        f"{row['category_name']:20s} "
        f"GT={row['gt_count']:4d} "
        f"AP={row['AP_50_95']:.4f}"
    )


print()
print("Bottom 10 categories by AP50-95")
print("-" * 60)

for row in valid_sorted_by_ap[
    -10:
]:

    print(
        f"{row['category_name']:20s} "
        f"GT={row['gt_count']:4d} "
        f"AP={row['AP_50_95']:.4f}"
    )


# ============================================================
# Correlation
# ============================================================

print()
print("=" * 60)
print("CORRELATION")
print("=" * 60)

print(
    f"Pearson GT count vs AP50-95 : "
    f"{pearson:.4f}"
)

print(
    f"Spearman GT count vs AP50-95: "
    f"{spearman:.4f}"
)


print()
print(
    "IMPORTANT:"
)

print(
    "This is evaluation support on the fixed "
    "COCO500 subset."
)

print(
    "Correlation does NOT prove that class frequency "
    "causes AP differences."
)

print()
print(
    "Saved:",
    CSV_PATH,
)