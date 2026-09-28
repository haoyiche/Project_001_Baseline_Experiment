import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModel,
)

from exp024_dinov3_resolution_ablation import (
    list_images,
    load_binary_gt,
    extract_patch_tokens,
    cosine_nn_anomaly_scores,
    evaluate_pixel_map,
    BATCH_SIZE_BY_SIZE,
    NN_QUERY_CHUNK,
)


MODEL_DIR = Path(
    r"D:\AI_Lab\models\dinov3"
    r"\dinov3-vits16-pretrain-lvd1689m"
)

DATA_ROOT = Path(
    "data/raw/mvtec_anomaly_detection"
)

CATEGORY = "bottle"
DEFECT_TYPE = "broken_small"

INPUT_SIZES = [224, 448]

OUTPUT_DIR = Path(
    "results/exp024/dinov3_resolution_dataset_eval"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PER_IMAGE_PATH = (
    OUTPUT_DIR
    / "per_image.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "summary.csv"
)

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


def build_memory(
    train_paths,
    model,
    processor,
    input_size,
):

    batch_size = (
        BATCH_SIZE_BY_SIZE[
            input_size
        ]
    )

    (
        normal_tokens,
        patch_h,
        patch_w,
    ) = extract_patch_tokens(
        train_paths,
        model,
        processor,
        input_size,
        batch_size,
    )

    hidden_size = (
        normal_tokens.shape[-1]
    )

    memory = normal_tokens.reshape(
        -1,
        hidden_size,
    )

    return (
        memory,
        patch_h,
        patch_w,
    )


def evaluate_one_image(
    test_path,
    gt_path,
    input_size,
    memory,
    patch_h,
    patch_w,
    model,
    processor,
):

    (
        query_tokens,
        query_h,
        query_w,
    ) = extract_patch_tokens(
        [test_path],
        model,
        processor,
        input_size,
        batch_size=1,
    )

    assert query_h == patch_h
    assert query_w == patch_w

    query = query_tokens[0]

    anomaly_scores = (
        cosine_nn_anomaly_scores(
            query,
            memory,
            NN_QUERY_CHUNK,
        )
    )

    anomaly_map = (
        anomaly_scores.reshape(
            patch_h,
            patch_w,
        )
    )

    gt_mask = load_binary_gt(
        gt_path
    )

    image = Image.open(
        test_path
    )

    assert (
        gt_mask.shape
        == (
            image.height,
            image.width,
        )
    )

    metrics = evaluate_pixel_map(
        anomaly_map,
        image.height,
        image.width,
        gt_mask,
    )

    return {
        "pixel_auroc":
            metrics["pixel_auroc"],

        "pixel_ap":
            metrics["pixel_ap"],

        "gt_positive_mean":
            metrics["positive_mean"],

        "gt_negative_mean":
            metrics["negative_mean"],
    }


def bootstrap_mean_ci(
    values,
    num_bootstrap=10000,
    seed=2026,
):

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    rng = np.random.default_rng(
        seed
    )

    means = np.empty(
        num_bootstrap,
        dtype=np.float64,
    )

    n = len(values)

    for i in range(
        num_bootstrap
    ):

        sample = rng.choice(
            values,
            size=n,
            replace=True,
        )

        means[i] = (
            sample.mean()
        )

    low = np.percentile(
        means,
        2.5,
    )

    high = np.percentile(
        means,
        97.5,
    )

    return (
        float(low),
        float(high),
    )


def paired_counts(
    deltas,
    eps=1e-12,
):

    wins = sum(
        value > eps
        for value in deltas
    )

    losses = sum(
        value < -eps
        for value in deltas
    )

    ties = (
        len(deltas)
        - wins
        - losses
    )

    return (
        wins,
        ties,
        losses,
    )


def main():

    print(
        "Device:",
        DEVICE
    )

    category_root = (
        DATA_ROOT
        / CATEGORY
    )

    train_paths = list_images(
        category_root
        / "train"
        / "good"
    )

    test_paths = list_images(
        category_root
        / "test"
        / DEFECT_TYPE
    )

    gt_dir = (
        category_root
        / "ground_truth"
        / DEFECT_TYPE
    )

    assert train_paths
    assert test_paths

    print(
        "Train good:",
        len(train_paths)
    )

    print(
        "Test images:",
        len(test_paths)
    )


    # --------------------------------
    # Model
    # --------------------------------

    processor = (
        AutoImageProcessor.from_pretrained(
            MODEL_DIR,
            local_files_only=True,
        )
    )

    model = AutoModel.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
    )

    model.eval()
    model.to(
        DEVICE
    )


    # --------------------------------
    # Build memory ONCE per resolution
    # --------------------------------

    memories = {}

    for input_size in INPUT_SIZES:

        print()
        print(
            "=" * 80
        )

        print(
            f"BUILD MEMORY {input_size}"
        )

        print(
            "=" * 80
        )

        (
            memory,
            patch_h,
            patch_w,
        ) = build_memory(
            train_paths,
            model,
            processor,
            input_size,
        )

        memories[
            input_size
        ] = {
            "memory":
                memory,

            "patch_h":
                patch_h,

            "patch_w":
                patch_w,
        }

        print(
            "Memory:",
            memory.shape
        )


    # --------------------------------
    # Paired per-image evaluation
    # --------------------------------

    rows = []

    for index, test_path in enumerate(
        test_paths
    ):

        gt_path = (
            gt_dir
            / f"{test_path.stem}_mask.png"
        )

        assert gt_path.exists()

        print()
        print(
            "=" * 80
        )

        print(
            f"IMAGE "
            f"{index + 1}/{len(test_paths)}: "
            f"{test_path.name}"
        )

        print(
            "=" * 80
        )

        per_resolution = {}

        for input_size in INPUT_SIZES:

            info = memories[
                input_size
            ]

            result = evaluate_one_image(
                test_path,
                gt_path,
                input_size,
                info["memory"],
                info["patch_h"],
                info["patch_w"],
                model,
                processor,
            )

            per_resolution[
                input_size
            ] = result

            print(
                f"{input_size}: "
                f"AP="
                f"{result['pixel_ap']:.6f}, "
                f"AUROC="
                f"{result['pixel_auroc']:.6f}"
            )


        r224 = per_resolution[
            224
        ]

        r448 = per_resolution[
            448
        ]

        delta_ap = (
            r448["pixel_ap"]
            - r224["pixel_ap"]
        )

        delta_auc = (
            r448["pixel_auroc"]
            - r224["pixel_auroc"]
        )

        row = {
            "image":
                test_path.name,

            "ap_224":
                r224["pixel_ap"],

            "ap_448":
                r448["pixel_ap"],

            "delta_ap":
                delta_ap,

            "auroc_224":
                r224["pixel_auroc"],

            "auroc_448":
                r448["pixel_auroc"],

            "delta_auroc":
                delta_auc,

            "gt_pos_mean_224":
                r224[
                    "gt_positive_mean"
                ],

            "gt_pos_mean_448":
                r448[
                    "gt_positive_mean"
                ],

            "gt_neg_mean_224":
                r224[
                    "gt_negative_mean"
                ],

            "gt_neg_mean_448":
                r448[
                    "gt_negative_mean"
                ],
        }

        rows.append(
            row
        )


    # --------------------------------
    # Save per-image
    # --------------------------------

    with open(
        PER_IMAGE_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=rows[
                0
            ].keys(),
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


    # --------------------------------
    # Paired summary
    # --------------------------------

    ap224 = np.array(
        [
            row["ap_224"]
            for row in rows
        ]
    )

    ap448 = np.array(
        [
            row["ap_448"]
            for row in rows
        ]
    )

    delta_ap = (
        ap448
        - ap224
    )

    auc224 = np.array(
        [
            row["auroc_224"]
            for row in rows
        ]
    )

    auc448 = np.array(
        [
            row["auroc_448"]
            for row in rows
        ]
    )

    delta_auc = (
        auc448
        - auc224
    )


    (
        ap_wins,
        ap_ties,
        ap_losses,
    ) = paired_counts(
        delta_ap
    )

    (
        ci_low,
        ci_high,
    ) = bootstrap_mean_ci(
        delta_ap
    )


    best_index = int(
        np.argmax(
            delta_ap
        )
    )

    worst_index = int(
        np.argmin(
            delta_ap
        )
    )


    summary = {
        "num_images":
            len(rows),

        "mean_ap_224":
            float(
                ap224.mean()
            ),

        "mean_ap_448":
            float(
                ap448.mean()
            ),

        "mean_delta_ap":
            float(
                delta_ap.mean()
            ),

        "median_delta_ap":
            float(
                np.median(
                    delta_ap
                )
            ),

        "std_delta_ap":
            float(
                delta_ap.std(
                    ddof=1
                )
            ),

        "ap_448_wins":
            ap_wins,

        "ap_ties":
            ap_ties,

        "ap_448_losses":
            ap_losses,

        "delta_ap_bootstrap_ci_low":
            ci_low,

        "delta_ap_bootstrap_ci_high":
            ci_high,

        "mean_auroc_224":
            float(
                auc224.mean()
            ),

        "mean_auroc_448":
            float(
                auc448.mean()
            ),

        "mean_delta_auroc":
            float(
                delta_auc.mean()
            ),

        "best_gain_image":
            rows[
                best_index
            ]["image"],

        "best_delta_ap":
            float(
                delta_ap[
                    best_index
                ]
            ),

        "worst_gain_image":
            rows[
                worst_index
            ]["image"],

        "worst_delta_ap":
            float(
                delta_ap[
                    worst_index
                ]
            ),
    }


    with open(
        SUMMARY_PATH,
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


    # --------------------------------
    # Print
    # --------------------------------

    print()
    print(
        "=" * 80
    )

    print(
        "BROKEN_SMALL PAIRED SUMMARY"
    )

    print(
        "=" * 80
    )

    for key, value in (
        summary.items()
    ):

        print(
            f"{key}: {value}"
        )

    print()
    print(
        "Saved:",
        PER_IMAGE_PATH
    )

    print(
        "Saved:",
        SUMMARY_PATH
    )


if __name__ == "__main__":
    main()