from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import distance_transform_edt


# ============================================================
# Config
# ============================================================

RESULT_ROOT = Path(
    "results/exp024/dinov3_resolution_ablation"
)

GT_PATH = Path(
    "data/raw/mvtec_anomaly_detection/"
    "bottle/ground_truth/broken_small/"
    "013_mask.png"
)

IMAGE_PATH = Path(
    "data/raw/mvtec_anomaly_detection/"
    "bottle/test/broken_small/"
    "013.png"
)

MAP_PATHS = {
    224: RESULT_ROOT / "anomaly_map_224.npy",
    448: RESULT_ROOT / "anomaly_map_448.npy",
}

OUTPUT_DIR = Path(
    "results/exp024/dinov3_fp_spatial_diagnosis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Helpers
# ============================================================

def load_gt():

    gt = Image.open(
        GT_PATH
    ).convert("L")

    gt = (
        np.asarray(gt) > 0
    )

    return gt


def upsample_map(
    anomaly_map,
    height,
    width,
):

    x = torch.from_numpy(
        anomaly_map
    ).float()

    x = (
        x
        .unsqueeze(0)
        .unsqueeze(0)
    )

    x = F.interpolate(
        x,
        size=(
            height,
            width,
        ),
        mode="bilinear",
        align_corners=False,
    )

    return (
        x.squeeze()
        .numpy()
    )


def summarize_distances(
    distances,
):

    if len(distances) == 0:
        return

    print(
        "Distance min:",
        distances.min(),
    )

    print(
        "Distance mean:",
        distances.mean(),
    )

    print(
        "Distance median:",
        np.median(
            distances
        ),
    )

    print(
        "Distance q90:",
        np.quantile(
            distances,
            0.90,
        ),
    )

    print(
        "Distance q95:",
        np.quantile(
            distances,
            0.95,
        ),
    )

    print(
        "Distance q99:",
        np.quantile(
            distances,
            0.99,
        ),
    )

    print(
        "Distance max:",
        distances.max(),
    )


def distance_bucket_report(
    distances,
):

    total = len(
        distances
    )

    if total == 0:
        return

    buckets = {
        "<= 5 px":
            distances <= 5,

        "<= 10 px":
            distances <= 10,

        "<= 20 px":
            distances <= 20,

        "<= 50 px":
            distances <= 50,

        "<= 100 px":
            distances <= 100,

        "> 100 px":
            distances > 100,
    }

    print()

    print(
        "Spatial concentration:"
    )

    for name, mask in (
        buckets.items()
    ):

        count = int(
            mask.sum()
        )

        ratio = (
            count / total
        )

        print(
            f"{name:10s}: "
            f"{count:5d} "
            f"({ratio:.4f})"
        )


# ============================================================
# Main
# ============================================================

def main():

    gt = load_gt()

    height, width = (
        gt.shape
    )

    image = Image.open(
        IMAGE_PATH
    ).convert("RGB")

    assert (
        image.size
        == (
            width,
            height,
        )
    )


    # --------------------------------------------------------
    # Distance from every non-GT pixel
    # to nearest GT-positive pixel.
    #
    # distance_transform_edt calculates
    # distance to nearest zero.
    #
    # So:
    # ~gt:
    # GT pixels     -> False / 0
    # negative area -> True / 1
    # --------------------------------------------------------

    distance_to_gt = (
        distance_transform_edt(
            ~gt
        )
    )


    print(
        "Image shape:",
        (
            height,
            width,
        )
    )

    print(
        "GT positive pixels:",
        int(
            gt.sum()
        )
    )


    figure_data = []


    for resolution, map_path in (
        MAP_PATHS.items()
    ):

        anomaly_map = np.load(
            map_path
        )

        score_map = upsample_map(
            anomaly_map,
            height,
            width,
        )

        negative_mask = (
            ~gt
        )

        negative_scores = (
            score_map[
                negative_mask
            ]
        )


        # ----------------------------------------------------
        # High-score negative thresholds
        # ----------------------------------------------------

        q99 = np.quantile(
            negative_scores,
            0.99,
        )

        q999 = np.quantile(
            negative_scores,
            0.999,
        )


        # Main diagnosis:
        # negative pixels in top 0.1%
        high_fp_mask = (
            negative_mask
            & (
                score_map
                >= q999
            )
        )

        fp_distances = (
            distance_to_gt[
                high_fp_mask
            ]
        )


        print()
        print(
            "=" * 70
        )

        print(
            "RESOLUTION:",
            resolution
        )

        print(
            "=" * 70
        )

        print(
            "Negative q99:",
            q99
        )

        print(
            "Negative q99.9:",
            q999
        )

        print(
            "High-score FP pixels:",
            int(
                high_fp_mask.sum()
            )
        )

        print()

        summarize_distances(
            fp_distances
        )

        distance_bucket_report(
            fp_distances
        )


        # ----------------------------------------------------
        # Also examine top 1% negatives
        # ----------------------------------------------------

        top1_mask = (
            negative_mask
            & (
                score_map
                >= q99
            )
        )

        top1_distances = (
            distance_to_gt[
                top1_mask
            ]
        )

        print()

        print(
            "TOP 1% NEGATIVE SUMMARY"
        )

        print(
            "Pixels:",
            int(
                top1_mask.sum()
            )
        )

        print(
            "Median distance:",
            float(
                np.median(
                    top1_distances
                )
            )
        )

        print(
            "Within 10 px:",
            float(
                (
                    top1_distances
                    <= 10
                ).mean()
            )
        )

        print(
            "Within 20 px:",
            float(
                (
                    top1_distances
                    <= 20
                ).mean()
            )
        )

        print(
            "Beyond 100 px:",
            float(
                (
                    top1_distances
                    > 100
                ).mean()
            )
        )


        figure_data.append(
            {
                "resolution":
                    resolution,

                "score_map":
                    score_map,

                "high_fp_mask":
                    high_fp_mask,

                "q999":
                    q999,
            }
        )


    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------

    fig = plt.figure(
        figsize=(16, 8)
    )


    ax1 = fig.add_subplot(
        2,
        3,
        1
    )

    ax1.imshow(
        image
    )

    ax1.set_title(
        "Input 013.png"
    )

    ax1.axis(
        "off"
    )


    ax2 = fig.add_subplot(
        2,
        3,
        2
    )

    ax2.imshow(
        gt
    )

    ax2.set_title(
        "Ground Truth"
    )

    ax2.axis(
        "off"
    )


    for i, data in enumerate(
        figure_data
    ):

        resolution = (
            data[
                "resolution"
            ]
        )

        score_map = (
            data[
                "score_map"
            ]
        )

        high_fp_mask = (
            data[
                "high_fp_mask"
            ]
        )


        # anomaly map
        ax_map = fig.add_subplot(
            2,
            3,
            3 + i
        )

        ax_map.imshow(
            score_map
        )

        ax_map.set_title(
            f"{resolution} anomaly map"
        )

        ax_map.axis(
            "off"
        )


        # high FP overlay
        ax_fp = fig.add_subplot(
            2,
            3,
            5 + i
        )

        ax_fp.imshow(
            image
        )

        overlay = np.zeros(
            (
                height,
                width,
                4,
            ),
            dtype=np.float32,
        )

        # Red = high-score negative pixels
        overlay[
            high_fp_mask,
            0
        ] = 1.0

        overlay[
            high_fp_mask,
            3
        ] = 0.8

        # Green = GT
        overlay[
            gt,
            1
        ] = 1.0

        overlay[
            gt,
            3
        ] = 0.5

        ax_fp.imshow(
            overlay
        )

        ax_fp.set_title(
            (
                f"{resolution}: "
                f"red=top0.1% FP, "
                f"green=GT"
            )
        )

        ax_fp.axis(
            "off"
        )


    fig.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "013_fp_spatial_diagnosis.png"
    )

    fig.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print()

    print(
        "Saved:",
        output_path
    )


if __name__ == "__main__":
    main()