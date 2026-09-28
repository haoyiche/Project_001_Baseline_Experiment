from pathlib import Path
import csv

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image

from exp025_autoencoder_bottle_minimal import (
    CHECKPOINT_PATH,
    DEVICE,
    IMAGE_SIZE,
    TEST_DIR,
    ConvAutoencoder,
    load_gt,
)


# ============================================================
# Config
# ============================================================

OUT_DIR = Path(
    "results/exp025/autoencoder_edge_fp_analysis"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Sobel kernels
# ============================================================

SOBEL_X = torch.tensor(
    [
        [-1.0, 0.0, 1.0],
        [-2.0, 0.0, 2.0],
        [-1.0, 0.0, 1.0],
    ]
).view(
    1,
    1,
    3,
    3,
)


SOBEL_Y = torch.tensor(
    [
        [-1.0, -2.0, -1.0],
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 1.0],
    ]
).view(
    1,
    1,
    3,
    3,
)


# ============================================================
# Model
# ============================================================

def load_model():

    assert CHECKPOINT_PATH.exists(), (
        f"Checkpoint missing: "
        f"{CHECKPOINT_PATH}"
    )

    model = ConvAutoencoder().to(
        DEVICE
    )

    state_dict = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(
        state_dict
    )

    model.eval()

    print(
        "Loaded:",
        CHECKPOINT_PATH
    )

    return model


# ============================================================
# Image
# ============================================================

def load_image(
    path,
):

    image = Image.open(
        path
    ).convert(
        "RGB"
    )

    original_np = (
        np.asarray(
            image
        )
        .astype(
            np.float32
        )
        / 255.0
    )

    original_size = (
        image.height,
        image.width,
    )

    resized = image.resize(
        (
            IMAGE_SIZE,
            IMAGE_SIZE,
        ),
        Image.BILINEAR,
    )

    x = (
        torch.from_numpy(
            np.asarray(
                resized
            ).copy()
        )
        .permute(
            2,
            0,
            1,
        )
        .float()
        / 255.0
    )

    return (
        x,
        original_np,
        original_size,
    )


# ============================================================
# Gradient
# ============================================================

def compute_gradient(
    rgb_np,
):

    # --------------------------------------------------------
    # RGB -> grayscale
    # --------------------------------------------------------

    gray = (
        0.299
        * rgb_np[
            ...,
            0
        ]
        +
        0.587
        * rgb_np[
            ...,
            1
        ]
        +
        0.114
        * rgb_np[
            ...,
            2
        ]
    )


    gray = (
        torch.from_numpy(
            gray
        )
        .float()
        .unsqueeze(
            0
        )
        .unsqueeze(
            0
        )
    )


    sobel_x = SOBEL_X.to(
        gray.device
    )

    sobel_y = SOBEL_Y.to(
        gray.device
    )


    gx = F.conv2d(
        gray,
        sobel_x,
        padding=1,
    )


    gy = F.conv2d(
        gray,
        sobel_y,
        padding=1,
    )


    grad = torch.sqrt(
        gx ** 2
        + gy ** 2
    )


    return (
        grad[
            0,
            0
        ]
        .numpy()
    )


# ============================================================
# Score
# ============================================================

@torch.inference_mode()
def compute_score(
    model,
    x,
    original_size,
):

    x = (
        x
        .unsqueeze(
            0
        )
        .to(
            DEVICE
        )
    )


    x_hat = model(
        x
    )


    score = torch.abs(
        x - x_hat
    ).mean(
        dim=1,
        keepdim=True,
    )


    score = F.interpolate(
        score,
        size=original_size,
        mode="bilinear",
        align_corners=False,
    )


    return (
        score[
            0,
            0
        ]
        .cpu()
        .numpy()
    )


# ============================================================
# Analysis
# ============================================================

def analyze(
    score,
    gradient,
    gt,
):

    negative_mask = ~gt


    negative_scores = score[
        negative_mask
    ]


    negative_gradient = gradient[
        negative_mask
    ]


    # --------------------------------------------------------
    # Top 1% anomaly-score normal pixels
    # --------------------------------------------------------

    score_threshold = np.quantile(
        negative_scores,
        0.99,
    )


    top_fp_mask = (
        negative_mask
        & (
            score
            >= score_threshold
        )
    )


    top_fp_gradient = gradient[
        top_fp_mask
    ]


    # --------------------------------------------------------
    # Top 10% gradient among normal pixels
    # --------------------------------------------------------

    gradient_threshold = np.quantile(
        negative_gradient,
        0.90,
    )


    high_gradient_mask = (
        negative_mask
        & (
            gradient
            >= gradient_threshold
        )
    )


    # --------------------------------------------------------
    # Enrichment
    #
    # Random normal pixels:
    # by definition about 10% are top-gradient.
    #
    # If top-score FP strongly overlap high-gradient regions,
    # this ratio should be much larger than 0.10.
    # --------------------------------------------------------

    overlap = (
        top_fp_mask
        & high_gradient_mask
    )


    overlap_ratio = (
        overlap.sum()
        / top_fp_mask.sum()
    )


    enrichment = (
        overlap_ratio
        / 0.10
    )


    return {

        "normal_gradient_mean":
            float(
                negative_gradient.mean()
            ),

        "normal_gradient_median":
            float(
                np.median(
                    negative_gradient
                )
            ),

        "top1_fp_gradient_mean":
            float(
                top_fp_gradient.mean()
            ),

        "top1_fp_gradient_median":
            float(
                np.median(
                    top_fp_gradient
                )
            ),

        "gradient_mean_ratio":
            float(
                top_fp_gradient.mean()
                / negative_gradient.mean()
            ),

        "gradient_median_ratio":
            float(
                np.median(
                    top_fp_gradient
                )
                /
                np.median(
                    negative_gradient
                )
            ),

        "top1_fp_in_top10_gradient":
            float(
                overlap_ratio
            ),

        "top10_gradient_enrichment":
            float(
                enrichment
            ),
    }


# ============================================================
# Main
# ============================================================

def main():

    print(
        "Device:",
        DEVICE
    )

    print(
        "Checkpoint:",
        CHECKPOINT_PATH
    )

    print()


    model = load_model()


    rows = []


    for path in sorted(
        TEST_DIR.glob(
            "*.png"
        )
    ):

        (
            x,
            original,
            original_size,
        ) = load_image(
            path
        )


        gt = load_gt(
            path
        )


        score = compute_score(
            model,
            x,
            original_size,
        )


        gradient = compute_gradient(
            original
        )


        assert (
            score.shape
            == gradient.shape
            == gt.shape
        )


        stats = analyze(
            score,
            gradient,
            gt,
        )


        row = {
            "image":
                path.name,
        }

        row.update(
            stats
        )

        rows.append(
            row
        )


        print(
            path.name
        )

        print(
            "  normal grad mean:",
            stats[
                "normal_gradient_mean"
            ]
        )

        print(
            "  top1 FP grad mean:",
            stats[
                "top1_fp_gradient_mean"
            ]
        )

        print(
            "  mean ratio:",
            stats[
                "gradient_mean_ratio"
            ]
        )

        print(
            "  FP in top10% gradient:",
            stats[
                "top1_fp_in_top10_gradient"
            ]
        )

        print(
            "  enrichment:",
            stats[
                "top10_gradient_enrichment"
            ]
        )


    # ========================================================
    # Dataset summary
    # ========================================================

    summary = {

        "num_images":
            len(
                rows
            ),

        "mean_gradient_mean_ratio":
            float(
                np.mean(
                    [
                        x[
                            "gradient_mean_ratio"
                        ]
                        for x
                        in rows
                    ]
                )
            ),

        "median_gradient_mean_ratio":
            float(
                np.median(
                    [
                        x[
                            "gradient_mean_ratio"
                        ]
                        for x
                        in rows
                    ]
                )
            ),

        "mean_fp_in_top10_gradient":
            float(
                np.mean(
                    [
                        x[
                            "top1_fp_in_top10_gradient"
                        ]
                        for x
                        in rows
                    ]
                )
            ),

        "mean_top10_gradient_enrichment":
            float(
                np.mean(
                    [
                        x[
                            "top10_gradient_enrichment"
                        ]
                        for x
                        in rows
                    ]
                )
            ),
    }


    print()

    print(
        "=" * 80
    )

    print(
        "EDGE FP ANALYSIS SUMMARY"
    )

    print(
        "=" * 80
    )


    for key, value in summary.items():

        print(
            key,
            "=",
            value,
        )


    # ========================================================
    # Save
    # ========================================================

    per_image_path = (
        OUT_DIR
        / "per_image.csv"
    )


    with open(
        per_image_path,
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


    summary_path = (
        OUT_DIR
        / "summary.csv"
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


    print()

    print(
        "Saved:",
        per_image_path
    )

    print(
        "Saved:",
        summary_path
    )


if __name__ == "__main__":

    main()