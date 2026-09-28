from pathlib import Path
import csv
import gc
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from scipy.ndimage import distance_transform_edt
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from torch.utils.data import (
    DataLoader,
    Dataset,
)
from torchvision.transforms import functional as TF


# ============================================================
# Config
# ============================================================

SEED = 42

DATA_ROOT = Path(
    "data/raw/mvtec_anomaly_detection/bottle"
)

TRAIN_DIR = (
    DATA_ROOT
    / "train"
    / "good"
)

NORMAL_TEST_DIR = (
    DATA_ROOT
    / "test"
    / "good"
)

ANOMALY_TEST_DIR = (
    DATA_ROOT
    / "test"
    / "broken_small"
)

GT_DIR = (
    DATA_ROOT
    / "ground_truth"
    / "broken_small"
)

OUT_DIR = Path(
    "results/exp025/autoencoder_capacity_ablation"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


IMAGE_SIZE = 224

BATCH_SIZE = 8

EPOCHS = 20

LR = 1e-3


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ------------------------------------------------------------
# Only experimental variable:
# channel capacity
#
# Spatial bottleneck remains:
# 14 x 14
# ------------------------------------------------------------

CAPACITY_CONFIGS = {

    "small": (
        16,
        32,
        64,
        128,
    ),

    "base": (
        32,
        64,
        128,
        256,
    ),

    "large": (
        64,
        128,
        256,
        512,
    ),
}


FAILURE_IMAGES = {
    "000.png",
    "005.png",
    "008.png",
    "013.png",
    "015.png",
}


# ============================================================
# Reproducibility
# ============================================================

def set_seed(
    seed,
):

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


    # Make the experiment more reproducible.
    #
    # This does not magically make every CUDA operation
    # bit-identical on every machine/version, but it removes
    # several common nondeterministic sources.

    if torch.backends.cudnn.is_available():

        torch.backends.cudnn.deterministic = True

        torch.backends.cudnn.benchmark = False


    try:

        torch.use_deterministic_algorithms(
            True,
            warn_only=True,
        )

    except TypeError:

        pass


# ============================================================
# Dataset
# ============================================================

class ImageDataset(Dataset):

    def __init__(
        self,
        image_dir,
    ):

        self.paths = sorted(
            image_dir.glob(
                "*.png"
            )
        )

        assert len(self.paths) > 0, (
            f"No images found: "
            f"{image_dir}"
        )


    def __len__(
        self,
    ):

        return len(
            self.paths
        )


    def __getitem__(
        self,
        index,
    ):

        image_path = self.paths[
            index
        ]

        image = Image.open(
            image_path
        ).convert(
            "RGB"
        )

        image = TF.resize(
            image,
            [
                IMAGE_SIZE,
                IMAGE_SIZE,
            ],
        )

        x = TF.to_tensor(
            image
        )

        return x


# ============================================================
# Parameterized Autoencoder
# ============================================================

class ConvAutoencoder(nn.Module):

    def __init__(
        self,
        widths,
    ):

        super().__init__()


        c1, c2, c3, c4 = widths


        # ----------------------------------------------------
        # Encoder
        #
        # 224
        # -> 112
        # -> 56
        # -> 28
        # -> 14
        # ----------------------------------------------------

        self.encoder = nn.Sequential(

            nn.Conv2d(
                3,
                c1,
                kernel_size=3,
                stride=2,
                padding=1,
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                c1,
                c2,
                kernel_size=3,
                stride=2,
                padding=1,
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                c2,
                c3,
                kernel_size=3,
                stride=2,
                padding=1,
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                c3,
                c4,
                kernel_size=3,
                stride=2,
                padding=1,
            ),

            nn.ReLU(
                inplace=True
            ),
        )


        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(
                c4,
                c3,
                kernel_size=4,
                stride=2,
                padding=1,
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.ConvTranspose2d(
                c3,
                c2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.ConvTranspose2d(
                c2,
                c1,
                kernel_size=4,
                stride=2,
                padding=1,
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.ConvTranspose2d(
                c1,
                3,
                kernel_size=4,
                stride=2,
                padding=1,
            ),

            nn.Sigmoid(),
        )


    def forward(
        self,
        x,
    ):

        z = self.encoder(
            x
        )

        x_hat = self.decoder(
            z
        )

        return x_hat


# ============================================================
# Utilities
# ============================================================

def count_parameters(
    model,
):

    return sum(
        p.numel()
        for p in model.parameters()
    )


def checkpoint_size_mb(
    path,
):

    return (
        path.stat().st_size
        / 1024
        / 1024
    )


# ============================================================
# Training
# ============================================================

def train_model(
    model_name,
    widths,
):

    # --------------------------------------------------------
    # Reset seed for every capacity.
    #
    # Each architecture begins from the same nominal seed.
    # --------------------------------------------------------

    set_seed(
        SEED
    )


    dataset = ImageDataset(
        TRAIN_DIR
    )


    generator = torch.Generator()

    generator.manual_seed(
        SEED
    )


    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )


    model = ConvAutoencoder(
        widths
    ).to(
        DEVICE
    )


    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
    )


    print()

    print(
        "=" * 80
    )

    print(
        f"TRAIN: {model_name.upper()}"
    )

    print(
        "Widths:",
        widths,
    )

    print(
        "Parameters:",
        count_parameters(
            model
        ),
    )

    print(
        "=" * 80
    )


    start = time.perf_counter()


    final_loss = None


    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        model.train()

        total_loss = 0.0

        total_samples = 0


        for x in loader:

            x = x.to(
                DEVICE
            )


            x_hat = model(
                x
            )


            loss = F.l1_loss(
                x_hat,
                x,
            )


            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            optimizer.step()


            batch_size = (
                x.shape[0]
            )


            total_loss += (
                loss.item()
                * batch_size
            )

            total_samples += (
                batch_size
            )


        final_loss = (
            total_loss
            / total_samples
        )


        print(
            f"{model_name:>5s} | "
            f"Epoch "
            f"{epoch:02d}/{EPOCHS} | "
            f"L1={final_loss:.6f}"
        )


    train_seconds = (
        time.perf_counter()
        - start
    )


    checkpoint_path = (
        OUT_DIR
        / (
            f"autoencoder_"
            f"{model_name}_seed{SEED}.pt"
        )
    )


    torch.save(
        model.state_dict(),
        checkpoint_path,
    )


    print()

    print(
        "Saved:",
        checkpoint_path,
    )


    return (
        model,
        final_loss,
        train_seconds,
        checkpoint_path,
    )


# ============================================================
# Normal reconstruction evaluation
#
# Directly tests:
# "Can the AE reconstruct normal Bottle structure?"
# ============================================================

@torch.inference_mode()
def evaluate_normal_reconstruction(
    model,
):

    dataset = ImageDataset(
        NORMAL_TEST_DIR
    )


    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )


    model.eval()


    total_abs_error = 0.0

    total_pixels = 0


    for x in loader:

        x = x.to(
            DEVICE
        )


        x_hat = model(
            x
        )


        abs_error = torch.abs(
            x - x_hat
        )


        total_abs_error += (
            abs_error.sum().item()
        )


        total_pixels += (
            abs_error.numel()
        )


    mean_l1 = (
        total_abs_error
        / total_pixels
    )


    return float(
        mean_l1
    )


# ============================================================
# Anomaly sample loading
# ============================================================

def load_anomaly_image(
    image_path,
):

    image = Image.open(
        image_path
    ).convert(
        "RGB"
    )


    original_size = (
        image.height,
        image.width,
    )


    resized = TF.resize(
        image,
        [
            IMAGE_SIZE,
            IMAGE_SIZE,
        ],
    )


    x = TF.to_tensor(
        resized
    )


    return (
        x,
        original_size,
    )


def load_gt(
    image_path,
):

    gt_path = (
        GT_DIR
        / (
            image_path.stem
            + "_mask.png"
        )
    )


    assert gt_path.exists(), (
        f"GT missing: "
        f"{gt_path}"
    )


    gt = Image.open(
        gt_path
    ).convert(
        "L"
    )


    gt = (
        np.array(
            gt
        )
        > 0
    )


    return gt


# ============================================================
# Failure diagnostics
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


    indices = np.argpartition(
        scores,
        -k,
    )[
        -k:
    ]


    return float(
        labels[
            indices
        ].mean()
    )


def top_negative_distance_stats(
    score_map,
    gt,
    fraction=0.01,
):

    negative_mask = (
        ~gt
    )


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


    distance_map = (
        distance_transform_edt(
            ~gt
        )
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

        "median_distance":
            float(
                np.median(
                    distances
                )
            ),

        "over_100px_ratio":
            float(
                np.mean(
                    distances
                    > 100
                )
            ),

        "within_20px_ratio":
            float(
                np.mean(
                    distances
                    <= 20
                )
            ),
    }


# ============================================================
# Broken-small evaluation
# ============================================================

@torch.inference_mode()
def evaluate_anomaly(
    model_name,
    model,
):

    model.eval()


    image_paths = sorted(
        ANOMALY_TEST_DIR.glob(
            "*.png"
        )
    )


    assert len(image_paths) > 0


    rows = []

    failure_rows = []


    global_scores = []

    global_labels = []


    for image_path in image_paths:

        (
            x,
            original_size,
        ) = load_anomaly_image(
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


        # ----------------------------------------------------
        # Reconstruction anomaly score
        # ----------------------------------------------------

        score_map = torch.abs(
            x - x_hat
        ).mean(
            dim=1,
            keepdim=True,
        )


        # ----------------------------------------------------
        # Common protocol:
        # score -> original image geometry
        # ----------------------------------------------------

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


        gt = load_gt(
            image_path
        )


        assert (
            score_map.shape
            == gt.shape
        )


        y_score = (
            score_map
            .reshape(
                -1
            )
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


        gt_positive_mean = float(
            positive_scores.mean()
        )


        gt_negative_mean = float(
            negative_scores.mean()
        )


        gt_positive_median = float(
            np.median(
                positive_scores
            )
        )


        negative_above_positive_median = int(
            np.sum(
                negative_scores
                > gt_positive_median
            )
        )


        row = {

            "model":
                model_name,

            "image":
                image_path.name,

            "pixel_auroc":
                auroc,

            "pixel_ap":
                ap,

            "gt_positive_mean":
                gt_positive_mean,

            "gt_negative_mean":
                gt_negative_mean,

            "gt_positive_median":
                gt_positive_median,

            "negative_above_positive_median":
                negative_above_positive_median,
        }


        rows.append(
            row
        )


        global_scores.append(
            y_score
        )


        global_labels.append(
            y_true
        )


        # ----------------------------------------------------
        # Fixed failure subset diagnostics
        # ----------------------------------------------------

        if image_path.name in FAILURE_IMAGES:

            p_gt = precision_at_gt_size(
                score_map,
                gt,
            )


            top1 = (
                top_negative_distance_stats(
                    score_map,
                    gt,
                    fraction=0.01,
                )
            )


            max_flat_idx = int(
                np.argmax(
                    score_map
                )
            )


            max_row, max_col = (
                np.unravel_index(
                    max_flat_idx,
                    score_map.shape,
                )
            )


            failure_rows.append(
                {
                    "model":
                        model_name,

                    "image":
                        image_path.name,

                    "pixel_ap":
                        ap,

                    "precision_at_gt_size":
                        p_gt,

                    "negative_above_positive_median":
                        negative_above_positive_median,

                    "max_inside_gt":
                        bool(
                            gt[
                                max_row,
                                max_col,
                            ]
                        ),

                    "top1_median_distance":
                        top1[
                            "median_distance"
                        ],

                    "top1_within_20px_ratio":
                        top1[
                            "within_20px_ratio"
                        ],

                    "top1_over_100px_ratio":
                        top1[
                            "over_100px_ratio"
                        ],
                }
            )


    # ========================================================
    # Aggregate metrics
    # ========================================================

    mean_pixel_auroc = float(
        np.mean(
            [
                row[
                    "pixel_auroc"
                ]
                for row
                in rows
            ]
        )
    )


    mean_pixel_ap = float(
        np.mean(
            [
                row[
                    "pixel_ap"
                ]
                for row
                in rows
            ]
        )
    )


    mean_gt_positive = float(
        np.mean(
            [
                row[
                    "gt_positive_mean"
                ]
                for row
                in rows
            ]
        )
    )


    mean_gt_negative = float(
        np.mean(
            [
                row[
                    "gt_negative_mean"
                ]
                for row
                in rows
            ]
        )
    )


    global_scores = np.concatenate(
        global_scores
    )


    global_labels = np.concatenate(
        global_labels
    )


    global_pixel_auroc = float(
        roc_auc_score(
            global_labels,
            global_scores,
        )
    )


    global_pixel_ap = float(
        average_precision_score(
            global_labels,
            global_scores,
        )
    )


    failure_mean_pgt = float(
        np.mean(
            [
                row[
                    "precision_at_gt_size"
                ]
                for row
                in failure_rows
            ]
        )
    )


    failure_mean_top1_over100 = float(
        np.mean(
            [
                row[
                    "top1_over_100px_ratio"
                ]
                for row
                in failure_rows
            ]
        )
    )


    failure_mean_top1_within20 = float(
        np.mean(
            [
                row[
                    "top1_within_20px_ratio"
                ]
                for row
                in failure_rows
            ]
        )
    )


    failure_mean_negative_intrusion = float(
        np.mean(
            [
                row[
                    "negative_above_positive_median"
                ]
                for row
                in failure_rows
            ]
        )
    )


    result = {

        "mean_pixel_auroc":
            mean_pixel_auroc,

        "mean_pixel_ap":
            mean_pixel_ap,

        "global_pixel_auroc":
            global_pixel_auroc,

        "global_pixel_ap":
            global_pixel_ap,

        "mean_gt_positive":
            mean_gt_positive,

        "mean_gt_negative":
            mean_gt_negative,

        "mean_pos_neg_ratio":
            (
                mean_gt_positive
                / mean_gt_negative
            ),

        "failure_mean_precision_at_gt_size":
            failure_mean_pgt,

        "failure_mean_top1_over100_ratio":
            failure_mean_top1_over100,

        "failure_mean_top1_within20_ratio":
            failure_mean_top1_within20,

        "failure_mean_negative_above_positive_median":
            failure_mean_negative_intrusion,
    }


    return (
        result,
        rows,
        failure_rows,
    )


# ============================================================
# CSV helpers
# ============================================================

def save_csv(
    path,
    rows,
):

    assert len(rows) > 0


    with open(
        path,
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


# ============================================================
# Main
# ============================================================

def main():

    set_seed(
        SEED
    )


    print(
        "Device:",
        DEVICE
    )

    print(
        "Seed:",
        SEED
    )

    print(
        "Input:",
        IMAGE_SIZE
    )

    print(
        "Epochs:",
        EPOCHS
    )

    print(
        "Batch size:",
        BATCH_SIZE
    )

    print(
        "LR:",
        LR
    )


    summary_rows = []

    all_per_image_rows = []

    all_failure_rows = []


    for (
        model_name,
        widths,
    ) in CAPACITY_CONFIGS.items():

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        (
            model,
            final_train_loss,
            train_seconds,
            checkpoint_path,
        ) = train_model(
            model_name,
            widths,
        )


        # ----------------------------------------------------
        # Normal reconstruction
        # ----------------------------------------------------

        normal_test_l1 = (
            evaluate_normal_reconstruction(
                model
            )
        )


        # ----------------------------------------------------
        # Anomaly evaluation
        # ----------------------------------------------------

        (
            anomaly_result,
            per_image_rows,
            failure_rows,
        ) = evaluate_anomaly(
            model_name,
            model,
        )


        parameter_count = (
            count_parameters(
                model
            )
        )


        checkpoint_mb = (
            checkpoint_size_mb(
                checkpoint_path
            )
        )


        summary_row = {

            "model":
                model_name,

            "widths":
                "-".join(
                    str(x)
                    for x
                    in widths
                ),

            "parameters":
                parameter_count,

            "checkpoint_mb":
                checkpoint_mb,

            "train_seconds":
                train_seconds,

            "final_train_l1":
                final_train_loss,

            "normal_test_l1":
                normal_test_l1,

            "mean_pixel_auroc":
                anomaly_result[
                    "mean_pixel_auroc"
                ],

            "mean_pixel_ap":
                anomaly_result[
                    "mean_pixel_ap"
                ],

            "global_pixel_auroc":
                anomaly_result[
                    "global_pixel_auroc"
                ],

            "global_pixel_ap":
                anomaly_result[
                    "global_pixel_ap"
                ],

            "mean_gt_positive":
                anomaly_result[
                    "mean_gt_positive"
                ],

            "mean_gt_negative":
                anomaly_result[
                    "mean_gt_negative"
                ],

            "mean_pos_neg_ratio":
                anomaly_result[
                    "mean_pos_neg_ratio"
                ],

            "failure_mean_precision_at_gt_size":
                anomaly_result[
                    "failure_mean_precision_at_gt_size"
                ],

            "failure_mean_top1_over100_ratio":
                anomaly_result[
                    "failure_mean_top1_over100_ratio"
                ],

            "failure_mean_top1_within20_ratio":
                anomaly_result[
                    "failure_mean_top1_within20_ratio"
                ],

            "failure_mean_negative_above_positive_median":
                anomaly_result[
                    "failure_mean_negative_above_positive_median"
                ],
        }


        summary_rows.append(
            summary_row
        )


        all_per_image_rows.extend(
            per_image_rows
        )


        all_failure_rows.extend(
            failure_rows
        )


        print()

        print(
            "-" * 80
        )

        print(
            f"RESULT: {model_name.upper()}"
        )

        print(
            "-" * 80
        )

        print(
            "Normal test L1:",
            normal_test_l1,
        )

        print(
            "Mean Pixel AUROC:",
            anomaly_result[
                "mean_pixel_auroc"
            ],
        )

        print(
            "Mean Pixel AP:",
            anomaly_result[
                "mean_pixel_ap"
            ],
        )

        print(
            "Global Pixel AUROC:",
            anomaly_result[
                "global_pixel_auroc"
            ],
        )

        print(
            "Global Pixel AP:",
            anomaly_result[
                "global_pixel_ap"
            ],
        )

        print(
            "Mean GT+:",
            anomaly_result[
                "mean_gt_positive"
            ],
        )

        print(
            "Mean GT-:",
            anomaly_result[
                "mean_gt_negative"
            ],
        )

        print(
            "GT+/GT- ratio:",
            anomaly_result[
                "mean_pos_neg_ratio"
            ],
        )

        print(
            "Failure P@GT-size:",
            anomaly_result[
                "failure_mean_precision_at_gt_size"
            ],
        )

        print(
            "Failure top1 >100px:",
            anomaly_result[
                "failure_mean_top1_over100_ratio"
            ],
        )

        print(
            "Failure top1 <=20px:",
            anomaly_result[
                "failure_mean_top1_within20_ratio"
            ],
        )


        # ----------------------------------------------------
        # Release GPU memory before next architecture.
        # ----------------------------------------------------

        del model

        gc.collect()

        if torch.cuda.is_available():

            torch.cuda.empty_cache()


    # ========================================================
    # Save results
    # ========================================================

    summary_path = (
        OUT_DIR
        / "summary.csv"
    )


    per_image_path = (
        OUT_DIR
        / "per_image.csv"
    )


    failure_path = (
        OUT_DIR
        / "failure_diagnostics.csv"
    )


    save_csv(
        summary_path,
        summary_rows,
    )


    save_csv(
        per_image_path,
        all_per_image_rows,
    )


    save_csv(
        failure_path,
        all_failure_rows,
    )


    # ========================================================
    # Final comparison
    # ========================================================

    print()

    print(
        "=" * 100
    )

    print(
        "CAPACITY ABLATION SUMMARY"
    )

    print(
        "=" * 100
    )


    for row in summary_rows:

        print()

        print(
            row[
                "model"
            ].upper()
        )

        print(
            "  widths:",
            row[
                "widths"
            ],
        )

        print(
            "  parameters:",
            row[
                "parameters"
            ],
        )

        print(
            "  normal_test_l1:",
            row[
                "normal_test_l1"
            ],
        )

        print(
            "  mean_pixel_auroc:",
            row[
                "mean_pixel_auroc"
            ],
        )

        print(
            "  mean_pixel_ap:",
            row[
                "mean_pixel_ap"
            ],
        )

        print(
            "  mean_gt_positive:",
            row[
                "mean_gt_positive"
            ],
        )

        print(
            "  mean_gt_negative:",
            row[
                "mean_gt_negative"
            ],
        )

        print(
            "  pos_neg_ratio:",
            row[
                "mean_pos_neg_ratio"
            ],
        )

        print(
            "  failure_P@GT:",
            row[
                "failure_mean_precision_at_gt_size"
            ],
        )

        print(
            "  failure_top1_>100px:",
            row[
                "failure_mean_top1_over100_ratio"
            ],
        )


    print()

    print(
        "Saved:",
        summary_path,
    )

    print(
        "Saved:",
        per_image_path,
    )

    print(
        "Saved:",
        failure_path,
    )


if __name__ == "__main__":

    main()