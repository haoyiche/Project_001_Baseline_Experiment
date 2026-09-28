from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
from scipy.ndimage import distance_transform_edt
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

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

TARGET_IMAGES = [
    "015.png",
    "008.png",
    "005.png",
    "013.png",
    "000.png",
]

OUT_DIR = Path(
    "results/exp025/autoencoder_failure_diagnosis"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Model
# ============================================================

def load_model():

    assert CHECKPOINT_PATH.exists(), (
        f"Checkpoint not found: "
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
        "Loaded checkpoint:",
        CHECKPOINT_PATH,
    )

    return model


# ============================================================
# Image loading
# ============================================================

def load_image(
    image_path,
):

    image = Image.open(
        image_path
    ).convert(
        "RGB"
    )

    original_np = (
        np.array(
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
        resample=Image.BILINEAR,
    )

    x = (
        torch.from_numpy(
            np.array(
                resized
            )
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
# Forward
# ============================================================

@torch.inference_mode()
def get_reconstruction_and_score(
    model,
    image_path,
):

    (
        x,
        original_np,
        original_size,
    ) = load_image(
        image_path
    )

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

    # --------------------------------------------------------
    # Reconstruction at 224
    # --------------------------------------------------------

    recon_224 = (
        x_hat[
            0
        ]
        .permute(
            1,
            2,
            0,
        )
        .detach()
        .cpu()
        .numpy()
    )

    # --------------------------------------------------------
    # Resize reconstruction to original geometry
    # for visualization only
    # --------------------------------------------------------

    recon_full = F.interpolate(
        x_hat,
        size=original_size,
        mode="bilinear",
        align_corners=False,
    )

    recon_full = (
        recon_full[
            0
        ]
        .permute(
            1,
            2,
            0,
        )
        .detach()
        .cpu()
        .numpy()
    )

    # --------------------------------------------------------
    # Anomaly map:
    # mean RGB absolute reconstruction error
    # --------------------------------------------------------

    score_map = torch.abs(
        x - x_hat
    ).mean(
        dim=1,
        keepdim=True,
    )

    score_map = F.interpolate(
        score_map,
        size=original_size,
        mode="bilinear",
        align_corners=False,
    )

    score_map = (
        score_map[
            0,
            0,
        ]
        .detach()
        .cpu()
        .numpy()
    )

    return (
        original_np,
        recon_224,
        recon_full,
        score_map,
    )


# ============================================================
# Diagnostics
# ============================================================

def precision_at_gt_size(
    score_map,
    gt,
):

    scores = score_map.reshape(
        -1
    )

    labels = (
        gt
        .astype(
            np.uint8
        )
        .reshape(
            -1
        )
    )

    k = int(
        labels.sum()
    )

    if k <= 0:

        return float(
            "nan"
        )

    top_idx = np.argpartition(
        scores,
        -k,
    )[
        -k:
    ]

    precision = float(
        labels[
            top_idx
        ].mean()
    )

    return precision


def negative_tail_distance_stats(
    score_map,
    gt,
    fraction,
):

    negative_mask = ~gt

    negative_scores = score_map[
        negative_mask
    ]

    n = len(
        negative_scores
    )

    k = max(
        1,
        int(
            np.ceil(
                n
                * fraction
            )
        ),
    )

    threshold = np.partition(
        negative_scores,
        -k,
    )[
        -k
    ]

    high_fp_mask = (
        negative_mask
        & (
            score_map
            >= threshold
        )
    )

    # --------------------------------------------------------
    # Distance from every non-GT pixel
    # to nearest GT-positive pixel.
    #
    # distance_transform_edt(~gt):
    # values outside GT = distance to nearest GT.
    # --------------------------------------------------------

    distance_map = distance_transform_edt(
        ~gt
    )

    distances = distance_map[
        high_fp_mask
    ]

    return {
        "count":
            int(
                len(
                    distances
                )
            ),

        "mean_distance":
            float(
                distances.mean()
            ),

        "median_distance":
            float(
                np.median(
                    distances
                )
            ),

        "q95_distance":
            float(
                np.quantile(
                    distances,
                    0.95,
                )
            ),

        "max_distance":
            float(
                distances.max()
            ),

        "within_10px_ratio":
            float(
                np.mean(
                    distances
                    <= 10
                )
            ),

        "within_20px_ratio":
            float(
                np.mean(
                    distances
                    <= 20
                )
            ),

        "over_100px_ratio":
            float(
                np.mean(
                    distances
                    > 100
                )
            ),
    }


def analyze_sample(
    score_map,
    gt,
):

    y_score = score_map.reshape(
        -1
    )

    y_true = (
        gt
        .astype(
            np.uint8
        )
        .reshape(
            -1
        )
    )

    auroc = float(
        roc_auc_score(
            y_true,
            y_score,
        )
    )

    ap = float(
        average_precision_score(
            y_true,
            y_score,
        )
    )

    positive_scores = y_score[
        y_true == 1
    ]

    negative_scores = y_score[
        y_true == 0
    ]

    positive_median = float(
        np.median(
            positive_scores
        )
    )

    num_negative_above_positive_median = int(
        np.sum(
            negative_scores
            > positive_median
        )
    )

    max_flat_idx = int(
        np.argmax(
            score_map
        )
    )

    max_row, max_col = np.unravel_index(
        max_flat_idx,
        score_map.shape,
    )

    max_inside_gt = bool(
        gt[
            max_row,
            max_col,
        ]
    )

    p_at_gt_size = precision_at_gt_size(
        score_map,
        gt,
    )

    top_01 = negative_tail_distance_stats(
        score_map,
        gt,
        fraction=0.001,
    )

    top_1 = negative_tail_distance_stats(
        score_map,
        gt,
        fraction=0.01,
    )

    return {
        "pixel_auroc":
            auroc,

        "pixel_ap":
            ap,

        "gt_positive_mean":
            float(
                positive_scores.mean()
            ),

        "gt_negative_mean":
            float(
                negative_scores.mean()
            ),

        "gt_positive_median":
            positive_median,

        "negative_above_positive_median":
            num_negative_above_positive_median,

        "precision_at_gt_size":
            p_at_gt_size,

        "max_score":
            float(
                score_map[
                    max_row,
                    max_col,
                ]
            ),

        "max_row":
            int(
                max_row
            ),

        "max_col":
            int(
                max_col
            ),

        "max_inside_gt":
            max_inside_gt,

        "top01_count":
            top_01[
                "count"
            ],

        "top01_mean_distance":
            top_01[
                "mean_distance"
            ],

        "top01_median_distance":
            top_01[
                "median_distance"
            ],

        "top01_q95_distance":
            top_01[
                "q95_distance"
            ],

        "top01_max_distance":
            top_01[
                "max_distance"
            ],

        "top01_within_10px_ratio":
            top_01[
                "within_10px_ratio"
            ],

        "top01_within_20px_ratio":
            top_01[
                "within_20px_ratio"
            ],

        "top01_over_100px_ratio":
            top_01[
                "over_100px_ratio"
            ],

        "top1_count":
            top_1[
                "count"
            ],

        "top1_mean_distance":
            top_1[
                "mean_distance"
            ],

        "top1_median_distance":
            top_1[
                "median_distance"
            ],

        "top1_q95_distance":
            top_1[
                "q95_distance"
            ],

        "top1_max_distance":
            top_1[
                "max_distance"
            ],

        "top1_within_10px_ratio":
            top_1[
                "within_10px_ratio"
            ],

        "top1_within_20px_ratio":
            top_1[
                "within_20px_ratio"
            ],

        "top1_over_100px_ratio":
            top_1[
                "over_100px_ratio"
            ],
    }


# ============================================================
# Visualization
# ============================================================

def normalize_for_display(
    x,
):

    lo = float(
        np.percentile(
            x,
            1
        )
    )

    hi = float(
        np.percentile(
            x,
            99
        )
    )

    if hi <= lo:

        return np.zeros_like(
            x
        )

    y = (
        x - lo
    ) / (
        hi - lo
    )

    y = np.clip(
        y,
        0.0,
        1.0,
    )

    return y


def save_visualization(
    image_name,
    original,
    recon_full,
    score_map,
    gt,
):

    display_score = normalize_for_display(
        score_map
    )

    overlay = original.copy()

    # Red overlay for GT
    red = np.zeros_like(
        overlay
    )

    red[
        ...,
        0
    ] = 1.0

    alpha = (
        gt.astype(
            np.float32
        )[
            ...,
            None
        ]
        * 0.45
    )

    overlay = (
        overlay
        * (
            1.0 - alpha
        )
        + red
        * alpha
    )

    fig = plt.figure(
        figsize=(
            20,
            4,
        )
    )

    ax1 = fig.add_subplot(
        1,
        5,
        1,
    )

    ax1.imshow(
        original
    )

    ax1.set_title(
        "Original"
    )

    ax1.axis(
        "off"
    )


    ax2 = fig.add_subplot(
        1,
        5,
        2,
    )

    ax2.imshow(
        np.clip(
            recon_full,
            0.0,
            1.0,
        )
    )

    ax2.set_title(
        "Reconstruction"
    )

    ax2.axis(
        "off"
    )


    ax3 = fig.add_subplot(
        1,
        5,
        3,
    )

    im = ax3.imshow(
        display_score
    )

    ax3.set_title(
        "|x - x_hat|"
    )

    ax3.axis(
        "off"
    )

    fig.colorbar(
        im,
        ax=ax3,
        fraction=0.046,
        pad=0.04,
    )


    ax4 = fig.add_subplot(
        1,
        5,
        4,
    )

    ax4.imshow(
        gt,
        cmap="gray",
    )

    ax4.set_title(
        "GT"
    )

    ax4.axis(
        "off"
    )


    ax5 = fig.add_subplot(
        1,
        5,
        5,
    )

    ax5.imshow(
        overlay
    )

    ax5.imshow(
        display_score,
        alpha=0.45,
    )

    ax5.set_title(
        "GT + anomaly"
    )

    ax5.axis(
        "off"
    )


    fig.suptitle(
        image_name
    )

    fig.tight_layout()


    output_path = (
        OUT_DIR
        / (
            Path(
                image_name
            ).stem
            + "_failure_diagnosis.png"
        )
    )


    fig.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


    print(
        "Saved:",
        output_path,
    )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "Device:",
        DEVICE,
    )

    print(
        "Checkpoint:",
        CHECKPOINT_PATH,
    )

    print()


    model = load_model()


    rows = []


    for image_name in TARGET_IMAGES:

        image_path = (
            TEST_DIR
            / image_name
        )

        assert image_path.exists(), (
            f"Missing test image: "
            f"{image_path}"
        )


        (
            original,
            _,
            recon_full,
            score_map,
        ) = (
            get_reconstruction_and_score(
                model,
                image_path,
            )
        )


        gt = load_gt(
            image_path
        )


        assert (
            score_map.shape
            == gt.shape
        )


        stats = analyze_sample(
            score_map,
            gt,
        )


        row = {
            "image":
                image_name,
        }

        row.update(
            stats
        )

        rows.append(
            row
        )


        print(
            "=" * 80
        )

        print(
            image_name
        )

        print(
            "Pixel AUROC:",
            stats[
                "pixel_auroc"
            ],
        )

        print(
            "Pixel AP:",
            stats[
                "pixel_ap"
            ],
        )

        print(
            "GT+ mean:",
            stats[
                "gt_positive_mean"
            ],
        )

        print(
            "GT- mean:",
            stats[
                "gt_negative_mean"
            ],
        )

        print(
            "GT+ median:",
            stats[
                "gt_positive_median"
            ],
        )

        print(
            "Negative > positive median:",
            stats[
                "negative_above_positive_median"
            ],
        )

        print(
            "Precision@GT-size:",
            stats[
                "precision_at_gt_size"
            ],
        )

        print(
            "Max location:",
            (
                stats[
                    "max_row"
                ],
                stats[
                    "max_col"
                ],
            ),
        )

        print(
            "Max inside GT:",
            stats[
                "max_inside_gt"
            ],
        )

        print(
            "Top 0.1% negative:"
        )

        print(
            "  median distance:",
            stats[
                "top01_median_distance"
            ],
        )

        print(
            "  <=10 px:",
            stats[
                "top01_within_10px_ratio"
            ],
        )

        print(
            "  <=20 px:",
            stats[
                "top01_within_20px_ratio"
            ],
        )

        print(
            "  >100 px:",
            stats[
                "top01_over_100px_ratio"
            ],
        )

        print(
            "Top 1% negative:"
        )

        print(
            "  median distance:",
            stats[
                "top1_median_distance"
            ],
        )

        print(
            "  <=10 px:",
            stats[
                "top1_within_10px_ratio"
            ],
        )

        print(
            "  <=20 px:",
            stats[
                "top1_within_20px_ratio"
            ],
        )

        print(
            "  >100 px:",
            stats[
                "top1_over_100px_ratio"
            ],
        )


        save_visualization(
            image_name,
            original,
            recon_full,
            score_map,
            gt,
        )


    csv_path = (
        OUT_DIR
        / "diagnostics.csv"
    )


    with open(
        csv_path,
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


    print()

    print(
        "Saved:",
        csv_path,
    )


if __name__ == "__main__":

    main()