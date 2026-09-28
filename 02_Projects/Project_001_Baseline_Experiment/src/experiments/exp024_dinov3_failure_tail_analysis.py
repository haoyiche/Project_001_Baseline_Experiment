from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)


ROOT = Path(
    "results/exp024/dinov3_resolution_ablation"
)

GT_PATH = Path(
    "data/raw/mvtec_anomaly_detection/"
    "bottle/ground_truth/broken_small/"
    "013_mask.png"
)

MAP_PATHS = {
    224: ROOT / "anomaly_map_224.npy",
    448: ROOT / "anomaly_map_448.npy",
}


def load_gt():

    gt = Image.open(
        GT_PATH
    ).convert("L")

    gt = (
        np.asarray(gt) > 0
    ).astype(np.uint8)

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
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )

    return (
        x.squeeze()
        .numpy()
    )


def describe(
    name,
    values,
):

    q = np.quantile(
        values,
        [
            0.50,
            0.90,
            0.95,
            0.99,
            0.999,
        ],
    )

    print(name)

    print(
        "  min:",
        values.min(),
    )

    print(
        "  mean:",
        values.mean(),
    )

    print(
        "  median:",
        q[0],
    )

    print(
        "  q90:",
        q[1],
    )

    print(
        "  q95:",
        q[2],
    )

    print(
        "  q99:",
        q[3],
    )

    print(
        "  q99.9:",
        q[4],
    )

    print(
        "  max:",
        values.max(),
    )


def main():

    gt = load_gt()

    height, width = (
        gt.shape
    )

    num_positive = int(
        gt.sum()
    )

    print(
        "GT positive pixels:",
        num_positive
    )

    for resolution, path in (
        MAP_PATHS.items()
    ):

        anomaly_map = np.load(
            path
        )

        score_map = upsample_map(
            anomaly_map,
            height,
            width,
        )

        labels = gt.reshape(-1)

        scores = score_map.reshape(
            -1
        )

        positive_scores = scores[
            labels == 1
        ]

        negative_scores = scores[
            labels == 0
        ]

        ap = average_precision_score(
            labels,
            scores,
        )

        auc = roc_auc_score(
            labels,
            scores,
        )

        # --------------------------------
        # Maximum score location
        # --------------------------------

        max_index = int(
            np.argmax(scores)
        )

        max_row = (
            max_index
            // width
        )

        max_col = (
            max_index
            % width
        )

        max_inside_gt = bool(
            gt[
                max_row,
                max_col,
            ]
        )

        # --------------------------------
        # Precision at |GT| pixels
        #
        # Take exactly as many top-ranked
        # pixels as there are GT-positive
        # pixels.
        # --------------------------------

        top_indices = np.argsort(
            -scores
        )[
            :num_positive
        ]

        precision_at_gt_size = (
            labels[
                top_indices
            ].mean()
        )

        # --------------------------------
        # Negative tail intrusion
        #
        # How many negative pixels score
        # above the median positive pixel?
        # --------------------------------

        positive_median = (
            np.median(
                positive_scores
            )
        )

        neg_above_pos_median = int(
            (
                negative_scores
                > positive_median
            ).sum()
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
            "Sklearn AUROC:",
            auc
        )

        print(
            "Sklearn AP:",
            ap
        )

        print(
            "Max location:",
            (
                max_row,
                max_col,
            )
        )

        print(
            "Max inside GT:",
            max_inside_gt
        )

        print(
            "Precision@GT-size:",
            precision_at_gt_size
        )

        print(
            "Negative pixels above "
            "positive median:",
            neg_above_pos_median
        )

        print()

        describe(
            "GT-positive scores",
            positive_scores,
        )

        print()

        describe(
            "GT-negative scores",
            negative_scores,
        )


if __name__ == "__main__":
    main()