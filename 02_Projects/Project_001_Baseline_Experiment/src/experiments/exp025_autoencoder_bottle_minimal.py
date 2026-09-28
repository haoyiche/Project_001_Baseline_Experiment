from pathlib import Path
import csv
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
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

TEST_DIR = (
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
    "results/exp025/autoencoder_bottle_minimal"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CHECKPOINT_PATH = (
    OUT_DIR
    / "autoencoder_seed42.pt"
)


# ------------------------------------------------------------
# Experiment parameters
# ------------------------------------------------------------

IMAGE_SIZE = 224

BATCH_SIZE = 8

EPOCHS = 20

LR = 1e-3


# ------------------------------------------------------------
# Mode
#
# False:
#   train -> save checkpoint -> evaluate
#
# True:
#   load checkpoint -> evaluate only
# ------------------------------------------------------------

EVAL_ONLY = True


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):

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


# ============================================================
# Dataset
# ============================================================

class NormalTrainDataset(Dataset):

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
            f"No training images found: "
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

        path = self.paths[
            index
        ]

        image = Image.open(
            path
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
# Model
# ============================================================

class ConvAutoencoder(nn.Module):

    def __init__(
        self,
    ):

        super().__init__()


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
                32,
                kernel_size=3,
                stride=2,
                padding=1,
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                stride=2,
                padding=1,
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                stride=2,
                padding=1,
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                128,
                256,
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
        #
        # 14
        # -> 28
        # -> 56
        # -> 112
        # -> 224
        # ----------------------------------------------------

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(
                256,
                128,
                kernel_size=4,
                stride=2,
                padding=1,
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.ConvTranspose2d(
                128,
                64,
                kernel_size=4,
                stride=2,
                padding=1,
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.ConvTranspose2d(
                64,
                32,
                kernel_size=4,
                stride=2,
                padding=1,
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.ConvTranspose2d(
                32,
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
# Checkpoint
# ============================================================

def save_checkpoint(
    model,
):

    torch.save(
        model.state_dict(),
        CHECKPOINT_PATH,
    )

    print()

    print(
        "Saved checkpoint:",
        CHECKPOINT_PATH,
    )


def load_checkpoint():

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
# Training
# ============================================================

def train_model():

    dataset = NormalTrainDataset(
        TRAIN_DIR
    )

    print(
        "Normal train images:",
        len(dataset),
    )


    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )


    model = ConvAutoencoder().to(
        DEVICE
    )


    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
    )


    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        model.train()

        epoch_loss = 0.0

        num_samples = 0


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


            current_batch_size = (
                x.shape[0]
            )

            epoch_loss += (
                loss.item()
                * current_batch_size
            )

            num_samples += (
                current_batch_size
            )


        epoch_loss /= (
            num_samples
        )


        print(
            f"Epoch "
            f"{epoch:02d}/{EPOCHS} "
            f"L1={epoch_loss:.6f}"
        )


    return model


# ============================================================
# Test image
# ============================================================

def load_test_image(
    image_path,
):

    image = Image.open(
        image_path
    ).convert(
        "RGB"
    )


    # F.interpolate expects:
    # size = (H, W)

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


# ============================================================
# Ground Truth
#
# Keep original GT geometry.
# Do NOT resize GT to 224.
# ============================================================

def load_gt(
    test_path,
):

    gt_path = (
        GT_DIR
        / (
            test_path.stem
            + "_mask.png"
        )
    )


    assert gt_path.exists(), (
        f"Missing GT: "
        f"{gt_path}"
    )


    mask = Image.open(
        gt_path
    ).convert(
        "L"
    )


    mask = (
        np.array(
            mask
        )
        > 0
    )


    return mask


# ============================================================
# Evaluation
# ============================================================

@torch.inference_mode()
def evaluate(
    model,
):

    model.eval()


    test_paths = sorted(
        TEST_DIR.glob(
            "*.png"
        )
    )


    assert len(test_paths) > 0, (
        f"No test images found: "
        f"{TEST_DIR}"
    )


    per_image = []


    all_scores = []

    all_labels = []


    print(
        "Test images:",
        len(test_paths),
    )


    for test_path in test_paths:

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        x, original_size = (
            load_test_image(
                test_path
            )
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


        # ----------------------------------------------------
        # Reconstruction
        # ----------------------------------------------------

        x_hat = model(
            x
        )


        # ----------------------------------------------------
        # Pixel reconstruction anomaly
        #
        # x:
        # [1, 3, 224, 224]
        #
        # score_map:
        # [1, 1, 224, 224]
        #
        # L1 RGB reconstruction error
        # ----------------------------------------------------

        score_map = torch.abs(
            x - x_hat
        ).mean(
            dim=1,
            keepdim=True,
        )


        # ----------------------------------------------------
        # Common evaluation protocol
        #
        # Resize continuous anomaly score map
        # back to original image geometry.
        #
        # Do NOT resize GT.
        # ----------------------------------------------------

        score_map = F.interpolate(
            score_map,
            size=original_size,
            mode="bilinear",
            align_corners=False,
        )


        # ----------------------------------------------------
        # [1, 1, H, W]
        # ->
        # [H, W]
        # ----------------------------------------------------

        score_map = (
            score_map[
                0,
                0,
            ]
            .detach()
            .cpu()
            .numpy()
        )


        # ----------------------------------------------------
        # Original-resolution GT
        # ----------------------------------------------------

        gt = load_gt(
            test_path
        )


        assert (
            score_map.shape
            == gt.shape
        ), (
            f"Geometry mismatch "
            f"for {test_path.name}: "
            f"score={score_map.shape}, "
            f"gt={gt.shape}"
        )


        # ----------------------------------------------------
        # Flatten
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # Pixel metrics
        # ----------------------------------------------------

        auroc = roc_auc_score(
            y_true,
            y_score,
        )


        ap = average_precision_score(
            y_true,
            y_score,
        )


        # ----------------------------------------------------
        # Distribution diagnostics
        # ----------------------------------------------------

        gt_positive_mean = float(
            y_score[
                y_true == 1
            ].mean()
        )


        gt_negative_mean = float(
            y_score[
                y_true == 0
            ].mean()
        )


        # ----------------------------------------------------
        # Print
        # ----------------------------------------------------

        print()

        print(
            test_path.name
        )


        print(
            "  Pixel AUROC:",
            auroc,
        )


        print(
            "  Pixel AP:",
            ap,
        )


        print(
            "  GT+ mean:",
            gt_positive_mean,
        )


        print(
            "  GT- mean:",
            gt_negative_mean,
        )


        # ----------------------------------------------------
        # Save per-image record
        # ----------------------------------------------------

        per_image.append(
            {
                "image":
                    test_path.name,

                "pixel_auroc":
                    auroc,

                "pixel_ap":
                    ap,

                "gt_positive_mean":
                    gt_positive_mean,

                "gt_negative_mean":
                    gt_negative_mean,
            }
        )


        all_scores.append(
            y_score
        )


        all_labels.append(
            y_true
        )


    # ========================================================
    # Dataset-level mean of per-image metrics
    # ========================================================

    mean_auroc = float(
        np.mean(
            [
                row[
                    "pixel_auroc"
                ]
                for row
                in per_image
            ]
        )
    )


    mean_ap = float(
        np.mean(
            [
                row[
                    "pixel_ap"
                ]
                for row
                in per_image
            ]
        )
    )


    # ========================================================
    # Global pixel metrics
    #
    # This is NOT the same as mean per-image metrics.
    #
    # Mean:
    # each image contributes equally.
    #
    # Global:
    # every pixel across all images is pooled.
    # ========================================================

    all_scores = np.concatenate(
        all_scores
    )


    all_labels = np.concatenate(
        all_labels
    )


    global_auroc = float(
        roc_auc_score(
            all_labels,
            all_scores,
        )
    )


    global_ap = float(
        average_precision_score(
            all_labels,
            all_scores,
        )
    )


    # ========================================================
    # Summary
    # ========================================================

    print()

    print(
        "=" * 80
    )

    print(
        "AUTOENCODER "
        "BROKEN_SMALL SUMMARY"
    )

    print(
        "=" * 80
    )


    print(
        "Images:",
        len(per_image),
    )


    print(
        "Mean Pixel AUROC:",
        mean_auroc,
    )


    print(
        "Mean Pixel AP:",
        mean_ap,
    )


    print(
        "Global Pixel AUROC:",
        global_auroc,
    )


    print(
        "Global Pixel AP:",
        global_ap,
    )


    # ========================================================
    # Save per-image CSV
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
            fieldnames=[
                "image",
                "pixel_auroc",
                "pixel_ap",
                "gt_positive_mean",
                "gt_negative_mean",
            ],
        )

        writer.writeheader()

        writer.writerows(
            per_image
        )


    # ========================================================
    # Save summary CSV
    # ========================================================

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

        writer = csv.writer(
            f
        )


        writer.writerow(
            [
                "seed",
                "image_size",
                "epochs",
                "batch_size",
                "lr",
                "num_images",
                "mean_pixel_auroc",
                "mean_pixel_ap",
                "global_pixel_auroc",
                "global_pixel_ap",
            ]
        )


        writer.writerow(
            [
                SEED,
                IMAGE_SIZE,
                EPOCHS,
                BATCH_SIZE,
                LR,
                len(per_image),
                mean_auroc,
                mean_ap,
                global_auroc,
                global_ap,
            ]
        )


    print()

    print(
        "Saved:",
        per_image_path,
    )

    print(
        "Saved:",
        summary_path,
    )


    return {
        "num_images":
            len(per_image),

        "mean_pixel_auroc":
            mean_auroc,

        "mean_pixel_ap":
            mean_ap,

        "global_pixel_auroc":
            global_auroc,

        "global_pixel_ap":
            global_ap,
    }


# ============================================================
# Main
# ============================================================

def main():

    set_seed(
        SEED
    )


    print(
        "Device:",
        DEVICE,
    )

    print(
        "Seed:",
        SEED,
    )

    print(
        "Image size:",
        IMAGE_SIZE,
    )

    print(
        "Epochs:",
        EPOCHS,
    )

    print(
        "Batch size:",
        BATCH_SIZE,
    )

    print(
        "Learning rate:",
        LR,
    )

    print(
        "Eval only:",
        EVAL_ONLY,
    )

    print(
        "Checkpoint:",
        CHECKPOINT_PATH,
    )

    print()


    # ========================================================
    # Train or load
    # ========================================================

    if EVAL_ONLY:

        model = load_checkpoint()


    else:

        start = time.perf_counter()


        model = train_model()


        elapsed = (
            time.perf_counter()
            - start
        )


        print()

        print(
            "Training elapsed:",
            elapsed,
            "seconds",
        )


        save_checkpoint(
            model
        )


    # ========================================================
    # Evaluate
    # ========================================================

    evaluate(
        model
    )


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":

    main()