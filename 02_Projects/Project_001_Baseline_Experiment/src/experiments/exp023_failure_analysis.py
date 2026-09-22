import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

from pycocotools.coco import COCO


# ============================================================
# Configuration
# ============================================================

MATCH_IOU = 0.50

# Same-class prediction overlaps GT, but not enough to become TP.
LOCALIZATION_IOU_MIN = 0.10

# A GT is treated as "crowded / overlapping proxy" when another
# GT of the SAME category overlaps it by at least this IoU.
#
# This is only a proxy for dense-object analysis.
CROWDED_GT_IOU = 0.10


# ============================================================
# Paths
# ============================================================

ANN_PATH = Path(
    "data/raw/coco2017/annotations/instances_val2017.json"
)

MANIFEST_PATH = Path(
    "configs/exp023_coco_subset500_ids.json"
)

IMAGE_DIR = Path(
    "data/raw/coco2017/val2017_subset500"
)

RESULT_DIR = Path(
    "results/exp023"
)

OUT_DIR = (
    RESULT_DIR
    / "failure_analysis"
)

EXAMPLE_DIR = (
    OUT_DIR
    / "examples"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

EXAMPLE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Prediction sets
# ============================================================

PREDICTION_FILES = {
    "baseline": (
        RESULT_DIR
        / "baseline500_predictions.json"
    ),

    "conf080": (
        RESULT_DIR
        / "confidence_ablation"
        / "conf_0.80_predictions.json"
    ),

    "nms030": (
        RESULT_DIR
        / "nms030_predictions.json"
    ),

    "res400": (
        RESULT_DIR
        / "res400_predictions.json"
    ),
}


# ============================================================
# Geometry helpers
# ============================================================

def xywh_to_xyxy(box):
    x, y, w, h = box

    return [
        float(x),
        float(y),
        float(x + w),
        float(y + h),
    ]


def box_iou_xyxy(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(
        ax1,
        bx1,
    )

    inter_y1 = max(
        ay1,
        by1,
    )

    inter_x2 = min(
        ax2,
        bx2,
    )

    inter_y2 = min(
        ay2,
        by2,
    )

    inter_w = max(
        0.0,
        inter_x2 - inter_x1,
    )

    inter_h = max(
        0.0,
        inter_y2 - inter_y1,
    )

    inter_area = (
        inter_w
        * inter_h
    )

    area_a = max(
        0.0,
        ax2 - ax1,
    ) * max(
        0.0,
        ay2 - ay1,
    )

    area_b = max(
        0.0,
        bx2 - bx1,
    ) * max(
        0.0,
        by2 - by1,
    )

    union = (
        area_a
        + area_b
        - inter_area
    )

    if union <= 0:
        return 0.0

    return (
        inter_area
        / union
    )


# ============================================================
# COCO size helper
# ============================================================

def coco_size_name(area):
    if area < 32 ** 2:
        return "small"

    if area < 96 ** 2:
        return "medium"

    return "large"


# ============================================================
# Load metadata
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

image_id_set = set(
    image_ids
)


coco = COCO(
    str(
        ANN_PATH
    )
)


category_names = {
    int(cat["id"]): cat["name"]
    for cat in coco.loadCats(
        coco.getCatIds()
    )
}


# ============================================================
# Build GT database
#
# For failure taxonomy:
# - only fixed COCO500
# - ignore COCO crowd annotations
#
# Official AP is still provided by COCOeval elsewhere.
# ============================================================

gt_by_image = defaultdict(
    list
)

all_gt_records = []


ann_ids = coco.getAnnIds(
    imgIds=image_ids,
)

annotations = coco.loadAnns(
    ann_ids
)


for ann in annotations:

    if int(
        ann.get(
            "iscrowd",
            0,
        )
    ) == 1:
        continue

    bbox_xywh = ann[
        "bbox"
    ]

    bbox_xyxy = xywh_to_xyxy(
        bbox_xywh
    )

    area = float(
        ann.get(
            "area",
            bbox_xywh[2]
            * bbox_xywh[3],
        )
    )

    record = {
        "ann_id": int(
            ann["id"]
        ),

        "image_id": int(
            ann["image_id"]
        ),

        "category_id": int(
            ann["category_id"]
        ),

        "bbox_xyxy": (
            bbox_xyxy
        ),

        "area": area,

        "size": coco_size_name(
            area
        ),

        "crowded_proxy": False,
    }

    gt_by_image[
        int(
            ann["image_id"]
        )
    ].append(
        record
    )

    all_gt_records.append(
        record
    )


# ============================================================
# Mark overlapping same-category GT
#
# This is a proxy for dense / adjacent target situations.
# It is NOT an official COCO property.
# ============================================================

for image_id in image_ids:

    gts = gt_by_image[
        image_id
    ]

    for i in range(
        len(gts)
    ):
        for j in range(
            i + 1,
            len(gts),
        ):

            if (
                gts[i][
                    "category_id"
                ]
                !=
                gts[j][
                    "category_id"
                ]
            ):
                continue

            iou = box_iou_xyxy(
                gts[i][
                    "bbox_xyxy"
                ],
                gts[j][
                    "bbox_xyxy"
                ],
            )

            if (
                iou
                >=
                CROWDED_GT_IOU
            ):
                gts[i][
                    "crowded_proxy"
                ] = True

                gts[j][
                    "crowded_proxy"
                ] = True


# ============================================================
# Load predictions
# ============================================================

def load_predictions(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        predictions = json.load(f)

    predictions = [
        pred
        for pred in predictions
        if int(
            pred[
                "image_id"
            ]
        ) in image_id_set
    ]

    by_image = defaultdict(
        list
    )

    for pred in predictions:

        record = {
            "image_id": int(
                pred["image_id"]
            ),

            "category_id": int(
                pred["category_id"]
            ),

            "bbox_xyxy": xywh_to_xyxy(
                pred["bbox"]
            ),

            "score": float(
                pred["score"]
            ),
        }

        by_image[
            record[
                "image_id"
            ]
        ].append(
            record
        )

    # Greedy detector matching uses score order.
    for image_id in by_image:

        by_image[
            image_id
        ].sort(
            key=lambda x: x[
                "score"
            ],
            reverse=True,
        )

    return (
        predictions,
        by_image,
    )


# ============================================================
# Match one prediction set
# ============================================================

def analyze_prediction_set(
    name,
    prediction_path,
):

    raw_predictions, pred_by_image = (
        load_predictions(
            prediction_path
        )
    )

    counts = Counter()

    prediction_failures = []
    gt_results = []

    for image_id in image_ids:

        gts = gt_by_image[
            image_id
        ]

        preds = pred_by_image[
            image_id
        ]

        matched_gt_indices = set()

        for pred in preds:

            pred_box = pred[
                "bbox_xyxy"
            ]

            pred_class = pred[
                "category_id"
            ]

            # -----------------------------------------------
            # IoU against every GT in this image
            # -----------------------------------------------

            ious = [
                box_iou_xyxy(
                    pred_box,
                    gt[
                        "bbox_xyxy"
                    ],
                )
                for gt in gts
            ]

            same_class_indices = [
                i
                for i, gt
                in enumerate(
                    gts
                )
                if (
                    gt[
                        "category_id"
                    ]
                    ==
                    pred_class
                )
            ]

            unmatched_same_indices = [
                i
                for i in same_class_indices
                if (
                    i
                    not in
                    matched_gt_indices
                )
            ]

            # -----------------------------------------------
            # 1. True Positive
            # -----------------------------------------------

            if unmatched_same_indices:

                best_unmatched = max(
                    unmatched_same_indices,
                    key=lambda i: ious[i],
                )

                best_unmatched_iou = (
                    ious[
                        best_unmatched
                    ]
                )

                if (
                    best_unmatched_iou
                    >=
                    MATCH_IOU
                ):

                    matched_gt_indices.add(
                        best_unmatched
                    )

                    counts[
                        "true_positive"
                    ] += 1

                    continue

            # -----------------------------------------------
            # Best same-category GT
            # -----------------------------------------------

            best_same_index = None
            best_same_iou = 0.0

            if same_class_indices:

                best_same_index = max(
                    same_class_indices,
                    key=lambda i: ious[i],
                )

                best_same_iou = (
                    ious[
                        best_same_index
                    ]
                )

            # -----------------------------------------------
            # 2. Duplicate detection
            #
            # Same-category GT already matched by a higher
            # scoring prediction.
            # -----------------------------------------------

            if (
                best_same_index
                is not None
                and best_same_iou
                >=
                MATCH_IOU
            ):

                counts[
                    "duplicate_detection"
                ] += 1

                target_gt = gts[
                    best_same_index
                ]

                prediction_failures.append(
                    {
                        "variant": name,
                        "failure_type": (
                            "duplicate_detection"
                        ),
                        "image_id": image_id,
                        "score": pred[
                            "score"
                        ],
                        "pred_category_id": (
                            pred_class
                        ),
                        "pred_bbox": (
                            pred_box
                        ),
                        "gt_category_id": (
                            target_gt[
                                "category_id"
                            ]
                        ),
                        "gt_bbox": (
                            target_gt[
                                "bbox_xyxy"
                            ]
                        ),
                        "iou": (
                            best_same_iou
                        ),
                    }
                )

                continue

            # -----------------------------------------------
            # Best different-class GT
            # -----------------------------------------------

            different_class_indices = [
                i
                for i, gt
                in enumerate(
                    gts
                )
                if (
                    gt[
                        "category_id"
                    ]
                    !=
                    pred_class
                )
            ]

            best_diff_index = None
            best_diff_iou = 0.0

            if different_class_indices:

                best_diff_index = max(
                    different_class_indices,
                    key=lambda i: ious[i],
                )

                best_diff_iou = (
                    ious[
                        best_diff_index
                    ]
                )

            # -----------------------------------------------
            # 3. Classification error
            #
            # Prediction spatially matches a GT at IoU >= .5
            # but predicts a different category.
            # -----------------------------------------------

            if (
                best_diff_index
                is not None
                and best_diff_iou
                >=
                MATCH_IOU
            ):

                counts[
                    "classification_error"
                ] += 1

                target_gt = gts[
                    best_diff_index
                ]

                prediction_failures.append(
                    {
                        "variant": name,
                        "failure_type": (
                            "classification_error"
                        ),
                        "image_id": image_id,
                        "score": pred[
                            "score"
                        ],
                        "pred_category_id": (
                            pred_class
                        ),
                        "pred_bbox": (
                            pred_box
                        ),
                        "gt_category_id": (
                            target_gt[
                                "category_id"
                            ]
                        ),
                        "gt_bbox": (
                            target_gt[
                                "bbox_xyxy"
                            ]
                        ),
                        "iou": (
                            best_diff_iou
                        ),
                    }
                )

                continue

            # -----------------------------------------------
            # 4. Localization error
            #
            # Correct category, some overlap exists,
            # but IoU is below .50.
            # -----------------------------------------------

            if (
                best_same_index
                is not None
                and best_same_iou
                >=
                LOCALIZATION_IOU_MIN
            ):

                counts[
                    "localization_error"
                ] += 1

                target_gt = gts[
                    best_same_index
                ]

                prediction_failures.append(
                    {
                        "variant": name,
                        "failure_type": (
                            "localization_error"
                        ),
                        "image_id": image_id,
                        "score": pred[
                            "score"
                        ],
                        "pred_category_id": (
                            pred_class
                        ),
                        "pred_bbox": (
                            pred_box
                        ),
                        "gt_category_id": (
                            target_gt[
                                "category_id"
                            ]
                        ),
                        "gt_bbox": (
                            target_gt[
                                "bbox_xyxy"
                            ]
                        ),
                        "iou": (
                            best_same_iou
                        ),
                    }
                )

                continue

            # -----------------------------------------------
            # 5. Background FP
            #
            # Prediction does not sufficiently overlap any
            # same-category GT and does not spatially match a
            # different-category GT at IoU >= .50.
            # -----------------------------------------------

            counts[
                "background_fp"
            ] += 1

            prediction_failures.append(
                {
                    "variant": name,
                    "failure_type": (
                        "background_fp"
                    ),
                    "image_id": image_id,
                    "score": pred[
                        "score"
                    ],
                    "pred_category_id": (
                        pred_class
                    ),
                    "pred_bbox": (
                        pred_box
                    ),
                    "gt_category_id": None,
                    "gt_bbox": None,
                    "iou": 0.0,
                }
            )

        # ----------------------------------------------------
        # Remaining unmatched GT = missed GT
        # ----------------------------------------------------

        for gt_index, gt in enumerate(
            gts
        ):

            matched = (
                gt_index
                in
                matched_gt_indices
            )

            gt_results.append(
                {
                    "variant": name,
                    "image_id": image_id,
                    "ann_id": (
                        gt[
                            "ann_id"
                        ]
                    ),
                    "category_id": (
                        gt[
                            "category_id"
                        ]
                    ),
                    "bbox_xyxy": (
                        gt[
                            "bbox_xyxy"
                        ]
                    ),
                    "size": (
                        gt[
                            "size"
                        ]
                    ),
                    "area": (
                        gt[
                            "area"
                        ]
                    ),
                    "crowded_proxy": (
                        gt[
                            "crowded_proxy"
                        ]
                    ),
                    "matched": matched,
                }
            )

            if not matched:

                counts[
                    "missed_gt"
                ] += 1

    # ========================================================
    # Derived simple metrics
    # ========================================================

    total_gt = len(
        gt_results
    )

    total_predictions = len(
        raw_predictions
    )

    tp = counts[
        "true_positive"
    ]

    simplified_precision = (
        tp
        / total_predictions
        if total_predictions
        else 0.0
    )

    simplified_recall = (
        tp
        / total_gt
        if total_gt
        else 0.0
    )

    if (
        simplified_precision
        +
        simplified_recall
        > 0
    ):
        simplified_f1 = (
            2
            * simplified_precision
            * simplified_recall
            /
            (
                simplified_precision
                +
                simplified_recall
            )
        )
    else:
        simplified_f1 = 0.0

    summary = {
        "variant": name,
        "num_predictions": (
            total_predictions
        ),
        "num_gt": (
            total_gt
        ),
        "true_positive": (
            counts[
                "true_positive"
            ]
        ),
        "background_fp": (
            counts[
                "background_fp"
            ]
        ),
        "classification_error": (
            counts[
                "classification_error"
            ]
        ),
        "localization_error": (
            counts[
                "localization_error"
            ]
        ),
        "duplicate_detection": (
            counts[
                "duplicate_detection"
            ]
        ),
        "missed_gt": (
            counts[
                "missed_gt"
            ]
        ),
        "simplified_precision_iou50": (
            simplified_precision
        ),
        "simplified_recall_iou50": (
            simplified_recall
        ),
        "simplified_f1_iou50": (
            simplified_f1
        ),
    }

    return (
        summary,
        prediction_failures,
        gt_results,
    )


# ============================================================
# Analyze all four prediction sets
# ============================================================

all_summaries = []
all_prediction_failures = {}
all_gt_results = {}


for name, path in PREDICTION_FILES.items():

    print()
    print("=" * 60)
    print(
        f"Analyzing: {name}"
    )
    print("=" * 60)

    summary, failures, gt_results = (
        analyze_prediction_set(
            name,
            path,
        )
    )

    all_summaries.append(
        summary
    )

    all_prediction_failures[
        name
    ] = failures

    all_gt_results[
        name
    ] = gt_results

    for key, value in (
        summary.items()
    ):
        print(
            f"{key:30s}: {value}"
        )


# ============================================================
# Save variant summary
# ============================================================

SUMMARY_PATH = (
    OUT_DIR
    / "failure_summary_by_variant.csv"
)


with open(
    SUMMARY_PATH,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=(
            all_summaries[
                0
            ].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(
        all_summaries
    )


# ============================================================
# Baseline Top-5 failure modes
#
# NOTE:
# missed_gt is GT-side.
# Other four are prediction-side.
# These are useful failure categories but are not one mutually
# exclusive denominator.
# ============================================================

baseline_summary = next(
    row
    for row in all_summaries
    if (
        row[
            "variant"
        ]
        ==
        "baseline"
    )
)


failure_names = [
    "background_fp",
    "classification_error",
    "localization_error",
    "duplicate_detection",
    "missed_gt",
]


baseline_failure_rows = [
    {
        "failure_type": name,
        "count": int(
            baseline_summary[
                name
            ]
        ),
    }
    for name in failure_names
]


baseline_failure_rows.sort(
    key=lambda x: x[
        "count"
    ],
    reverse=True,
)


TOP5_PATH = (
    OUT_DIR
    / "baseline_top5_failure_modes.csv"
)


with open(
    TOP5_PATH,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "failure_type",
            "count",
        ],
    )

    writer.writeheader()
    writer.writerows(
        baseline_failure_rows
    )


# ============================================================
# Cause validation 1:
#
# confidence 0.05 -> 0.80
#
# Prediction:
# background FP decreases
# missed GT increases
# ============================================================

summary_by_name = {
    row[
        "variant"
    ]: row
    for row in all_summaries
}


baseline = summary_by_name[
    "baseline"
]

conf080 = summary_by_name[
    "conf080"
]


confidence_validation = {
    "baseline_background_fp": (
        baseline[
            "background_fp"
        ]
    ),

    "conf080_background_fp": (
        conf080[
            "background_fp"
        ]
    ),

    "background_fp_delta": (
        conf080[
            "background_fp"
        ]
        -
        baseline[
            "background_fp"
        ]
    ),

    "baseline_missed_gt": (
        baseline[
            "missed_gt"
        ]
    ),

    "conf080_missed_gt": (
        conf080[
            "missed_gt"
        ]
    ),

    "missed_gt_delta": (
        conf080[
            "missed_gt"
        ]
        -
        baseline[
            "missed_gt"
        ]
    ),
}


# ============================================================
# GT slice helpers
# ============================================================

def gt_slice_stats(
    gt_results,
    key,
    value,
):

    selected = [
        row
        for row in gt_results
        if (
            row[
                key
            ]
            ==
            value
        )
    ]

    total = len(
        selected
    )

    missed = sum(
        not row[
            "matched"
        ]
        for row in selected
    )

    miss_rate = (
        missed
        / total
        if total
        else float("nan")
    )

    return (
        total,
        missed,
        miss_rate,
    )


# ============================================================
# Cause validation 2:
#
# NMS=0.30 vs baseline.
#
# Compare crowded same-class GT proxy and isolated GT.
# ============================================================

nms_rows = []


for variant in [
    "baseline",
    "nms030",
]:

    gt_results = (
        all_gt_results[
            variant
        ]
    )

    for crowded_status in [
        True,
        False,
    ]:

        total, missed, rate = (
            gt_slice_stats(
                gt_results,
                "crowded_proxy",
                crowded_status,
            )
        )

        nms_rows.append(
            {
                "variant": variant,
                "slice": (
                    "overlapping_same_class_gt"
                    if crowded_status
                    else "other_gt"
                ),
                "total_gt": total,
                "missed_gt": missed,
                "miss_rate": rate,
            }
        )


# ============================================================
# Cause validation 3:
#
# Resolution baseline vs res400
# by COCO object size.
# ============================================================

resolution_rows = []


for variant in [
    "baseline",
    "res400",
]:

    gt_results = (
        all_gt_results[
            variant
        ]
    )

    for size_name in [
        "small",
        "medium",
        "large",
    ]:

        total, missed, rate = (
            gt_slice_stats(
                gt_results,
                "size",
                size_name,
            )
        )

        resolution_rows.append(
            {
                "variant": variant,
                "size": size_name,
                "total_gt": total,
                "missed_gt": missed,
                "miss_rate": rate,
            }
        )


# ============================================================
# Save validation CSVs
# ============================================================

CONF_PATH = (
    OUT_DIR
    / "cause_validation_confidence.csv"
)


with open(
    CONF_PATH,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=(
            confidence_validation.keys()
        ),
    )

    writer.writeheader()
    writer.writerow(
        confidence_validation
    )


NMS_PATH = (
    OUT_DIR
    / "cause_validation_nms.csv"
)


with open(
    NMS_PATH,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=nms_rows[
            0
        ].keys(),
    )

    writer.writeheader()
    writer.writerows(
        nms_rows
    )


RES_PATH = (
    OUT_DIR
    / "cause_validation_resolution.csv"
)


with open(
    RES_PATH,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=resolution_rows[
            0
        ].keys(),
    )

    writer.writeheader()
    writer.writerows(
        resolution_rows
    )


# ============================================================
# Failure examples
# ============================================================

def draw_box(
    draw,
    box,
    width=3,
):

    x1, y1, x2, y2 = box

    draw.rectangle(
        [
            x1,
            y1,
            x2,
            y2,
        ],
        width=width,
    )


def save_prediction_failure_example(
    record,
    output_path,
):

    image_info = coco.loadImgs(
        [
            record[
                "image_id"
            ]
        ]
    )[0]

    image_path = (
        IMAGE_DIR
        / image_info[
            "file_name"
        ]
    )

    image = Image.open(
        image_path
    ).convert("RGB")

    draw = ImageDraw.Draw(
        image
    )

    # Prediction box
    draw_box(
        draw,
        record[
            "pred_bbox"
        ],
        width=4,
    )

    pred_name = category_names.get(
        record[
            "pred_category_id"
        ],
        str(
            record[
                "pred_category_id"
            ]
        ),
    )

    label = (
        f"PRED {pred_name} "
        f"{record['score']:.2f} "
        f"{record['failure_type']}"
    )

    draw.text(
        (
            record[
                "pred_bbox"
            ][0],
            max(
                0,
                record[
                    "pred_bbox"
                ][1]
                - 15,
            ),
        ),
        label,
    )

    # Relevant GT, when available.
    if (
        record[
            "gt_bbox"
        ]
        is not None
    ):

        draw_box(
            draw,
            record[
                "gt_bbox"
            ],
            width=2,
        )

        gt_name = category_names.get(
            record[
                "gt_category_id"
            ],
            str(
                record[
                    "gt_category_id"
                ]
            ),
        )

        draw.text(
            (
                record[
                    "gt_bbox"
                ][0],
                record[
                    "gt_bbox"
                ][1],
            ),
            f"GT {gt_name}",
        )

    image.save(
        output_path
    )


def save_missed_gt_example(
    record,
    output_path,
):

    image_info = coco.loadImgs(
        [
            record[
                "image_id"
            ]
        ]
    )[0]

    image_path = (
        IMAGE_DIR
        / image_info[
            "file_name"
        ]
    )

    image = Image.open(
        image_path
    ).convert("RGB")

    draw = ImageDraw.Draw(
        image
    )

    draw_box(
        draw,
        record[
            "bbox_xyxy"
        ],
        width=4,
    )

    category_name = (
        category_names.get(
            record[
                "category_id"
            ],
            str(
                record[
                    "category_id"
                ]
            ),
        )
    )

    draw.text(
        (
            record[
                "bbox_xyxy"
            ][0],
            max(
                0,
                record[
                    "bbox_xyxy"
                ][1]
                - 15,
            ),
        ),
        (
            f"MISSED GT "
            f"{category_name} "
            f"{record['size']}"
        ),
    )

    image.save(
        output_path
    )


baseline_failures = (
    all_prediction_failures[
        "baseline"
    ]
)


# Highest-score example for each prediction-side failure.
for failure_type in [
    "background_fp",
    "classification_error",
    "localization_error",
    "duplicate_detection",
]:

    candidates = [
        row
        for row in baseline_failures
        if (
            row[
                "failure_type"
            ]
            ==
            failure_type
        )
    ]

    if not candidates:
        continue

    candidates.sort(
        key=lambda x: x[
            "score"
        ],
        reverse=True,
    )

    record = candidates[
        0
    ]

    save_prediction_failure_example(
        record,
        EXAMPLE_DIR
        / f"{failure_type}.png",
    )


# Prefer a missed small GT as the missed-GT example.
missed_baseline = [
    row
    for row in all_gt_results[
        "baseline"
    ]
    if not row[
        "matched"
    ]
]


if missed_baseline:

    missed_baseline.sort(
        key=lambda x: (
            x[
                "size"
            ]
            !=
            "small",
            x[
                "area"
            ]
            if "area" in x
            else 0,
        )
    )

    # We did not keep area inside gt_results, so choosing the
    # first small missed GT is sufficient for this visualization.
    small_missed = [
        row
        for row in missed_baseline
        if (
            row[
                "size"
            ]
            ==
            "small"
        )
    ]

    record = (
        small_missed[
            0
        ]
        if small_missed
        else missed_baseline[
            0
        ]
    )

    save_missed_gt_example(
        record,
        EXAMPLE_DIR
        / "missed_gt.png",
    )


# ============================================================
# Print final report
# ============================================================

print()
print("=" * 70)
print("BASELINE TOP-5 FAILURE MODES")
print("=" * 70)

for row in baseline_failure_rows:

    print(
        f"{row['failure_type']:25s}"
        f"{row['count']:8d}"
    )


print()
print("=" * 70)
print("CAUSE VALIDATION 1: CONFIDENCE")
print("=" * 70)

for key, value in (
    confidence_validation.items()
):

    print(
        f"{key:32s}: {value}"
    )


print()
print("=" * 70)
print("CAUSE VALIDATION 2: NMS / OVERLAPPING-GT PROXY")
print("=" * 70)

for row in nms_rows:

    print(
        f"{row['variant']:10s} | "
        f"{row['slice']:28s} | "
        f"GT={row['total_gt']:5d} | "
        f"miss={row['missed_gt']:5d} | "
        f"rate={row['miss_rate']:.4f}"
    )


print()
print("=" * 70)
print("CAUSE VALIDATION 3: RESOLUTION / OBJECT SIZE")
print("=" * 70)

for row in resolution_rows:

    print(
        f"{row['variant']:10s} | "
        f"{row['size']:8s} | "
        f"GT={row['total_gt']:5d} | "
        f"miss={row['missed_gt']:5d} | "
        f"rate={row['miss_rate']:.4f}"
    )


print()
print("=" * 70)
print("OUTPUTS")
print("=" * 70)

print(
    SUMMARY_PATH
)

print(
    TOP5_PATH
)

print(
    CONF_PATH
)

print(
    NMS_PATH
)

print(
    RES_PATH
)

print(
    EXAMPLE_DIR
)

print()
print(
    "IMPORTANT: this is a simplified IoU=0.50 failure taxonomy."
)

print(
    "It does not replace official COCOeval AP/AR because "
    "COCO crowd/ignore/area rules are more complex."
)