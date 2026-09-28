import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModel,
)


# ============================================================
# Config
# ============================================================

MODEL_DIR = Path(
    r"D:\AI_Lab\models\dinov3"
    r"\dinov3-vits16-pretrain-lvd1689m"
)

DATA_ROOT = Path(
    "data/raw/mvtec_anomaly_detection"
)

CATEGORY = "bottle"
DEFECT_TYPE = "broken_small"
TEST_INDEX = 0

OUTPUT_DIR = Path(
    "results/exp024/dinov3_bottle_anomaly"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
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

BATCH_SIZE = 8


# ============================================================
# Helpers
# ============================================================

def list_images(directory):

    paths = sorted(
        list(directory.glob("*.png"))
        + list(directory.glob("*.jpg"))
        + list(directory.glob("*.jpeg"))
    )

    return paths


def extract_patch_tokens(
    image_paths,
    model,
    processor,
    num_register_tokens,
):

    all_tokens = []

    patch_height = None
    patch_width = None

    for start in range(
        0,
        len(image_paths),
        BATCH_SIZE,
    ):

        batch_paths = image_paths[
            start:
            start + BATCH_SIZE
        ]

        images = [
            Image.open(path).convert("RGB")
            for path in batch_paths
        ]

        inputs = processor(
            images=images,
            return_tensors="pt",
        )

        pixel_values = (
            inputs["pixel_values"]
        )

        # Determine patch grid from actual
        # processor output.
        image_height = (
            pixel_values.shape[2]
        )

        image_width = (
            pixel_values.shape[3]
        )

        patch_size = (
            model.config.patch_size
        )

        current_patch_height = (
            image_height
            // patch_size
        )

        current_patch_width = (
            image_width
            // patch_size
        )

        if patch_height is None:

            patch_height = (
                current_patch_height
            )

            patch_width = (
                current_patch_width
            )

        assert (
            patch_height
            == current_patch_height
        )

        assert (
            patch_width
            == current_patch_width
        )

        inputs = {
            key: value.to(DEVICE)
            for key, value
            in inputs.items()
        }

        with torch.inference_mode():

            outputs = model(
                **inputs
            )

        hidden = (
            outputs.last_hidden_state
        )

        # Layout:
        #
        # 0                    -> CLS
        # 1 ... R             -> register tokens
        # 1 + R ... end       -> patch tokens

        patch_tokens = hidden[
            :,
            1 + num_register_tokens:,
            :,
        ]

        expected_num_patches = (
            patch_height
            * patch_width
        )

        assert (
            patch_tokens.shape[1]
            == expected_num_patches
        )

        # Normalize now because this experiment
        # uses cosine nearest-neighbor similarity.

        patch_tokens = F.normalize(
            patch_tokens,
            p=2,
            dim=-1,
        )

        all_tokens.append(
            patch_tokens.cpu()
        )

        print(
            f"Extracted "
            f"{min(start + BATCH_SIZE, len(image_paths))}"
            f"/{len(image_paths)}"
        )

    all_tokens = torch.cat(
        all_tokens,
        dim=0,
    )

    return (
        all_tokens,
        patch_height,
        patch_width,
    )


def load_gt_mask(
    path,
    output_height,
    output_width,
):

    mask = Image.open(
        path
    ).convert("L")

    mask = mask.resize(
        (
            output_width,
            output_height,
        ),
        resample=Image.Resampling.NEAREST,
    )

    mask_np = np.asarray(
        mask,
        dtype=np.float32,
    )

    mask_np = (
        mask_np > 0
    ).astype(
        np.float32
    )

    return mask_np


def patch_gt_from_pixel_mask(
    pixel_mask,
    patch_size,
):

    mask_tensor = torch.from_numpy(
        pixel_mask
    ).float()

    mask_tensor = (
        mask_tensor
        .unsqueeze(0)
        .unsqueeze(0)
    )

    # If ANY defect pixel falls inside a patch,
    # that patch becomes positive.

    patch_mask = F.max_pool2d(
        mask_tensor,
        kernel_size=patch_size,
        stride=patch_size,
    )

    patch_mask = (
        patch_mask.squeeze()
        .numpy()
    )

    return patch_mask


def simple_roc_auc(
    labels,
    scores,
):

    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    scores = np.asarray(
        scores,
        dtype=np.float64,
    )

    positive = (
        labels == 1
    )

    negative = (
        labels == 0
    )

    n_pos = positive.sum()
    n_neg = negative.sum()

    if (
        n_pos == 0
        or n_neg == 0
    ):
        return float("nan")

    # Pairwise interpretation of AUROC:
    # probability that a positive score
    # is larger than a negative score.

    pos_scores = scores[
        positive
    ]

    neg_scores = scores[
        negative
    ]

    comparisons = (
        pos_scores[:, None]
        > neg_scores[None, :]
    ).mean()

    ties = (
        pos_scores[:, None]
        == neg_scores[None, :]
    ).mean()

    return float(
        comparisons
        + 0.5 * ties
    )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "Device:",
        DEVICE
    )

    # --------------------------------------------------------
    # 1. Dataset paths
    # --------------------------------------------------------

    category_root = (
        DATA_ROOT
        / CATEGORY
    )

    train_good_dir = (
        category_root
        / "train"
        / "good"
    )

    test_dir = (
        category_root
        / "test"
        / DEFECT_TYPE
    )

    gt_dir = (
        category_root
        / "ground_truth"
        / DEFECT_TYPE
    )

    train_paths = list_images(
        train_good_dir
    )

    test_paths = list_images(
        test_dir
    )

    assert train_paths, (
        f"No training images: "
        f"{train_good_dir}"
    )

    assert test_paths, (
        f"No test images: "
        f"{test_dir}"
    )

    assert (
        TEST_INDEX
        < len(test_paths)
    )

    test_path = (
        test_paths[
            TEST_INDEX
        ]
    )

    gt_path = (
        gt_dir
        / f"{test_path.stem}_mask.png"
    )

    assert gt_path.exists(), (
        f"GT mask not found: "
        f"{gt_path}"
    )

    print()
    print(
        "Train good images:",
        len(train_paths)
    )

    print(
        "Test image:",
        test_path
    )

    print(
        "GT mask:",
        gt_path
    )


    # --------------------------------------------------------
    # 2. Load DINOv3
    # --------------------------------------------------------

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

    patch_size = (
        model.config.patch_size
    )

    hidden_size = (
        model.config.hidden_size
    )

    num_register_tokens = (
        model.config.num_register_tokens
    )

    print()
    print(
        "Patch size:",
        patch_size
    )

    print(
        "Hidden size:",
        hidden_size
    )

    print(
        "Register tokens:",
        num_register_tokens
    )


    # --------------------------------------------------------
    # 3. Build NORMAL patch memory
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )

    print(
        "BUILDING NORMAL MEMORY"
    )

    print(
        "=" * 70
    )

    (
        normal_tokens,
        patch_height,
        patch_width,
    ) = extract_patch_tokens(
        train_paths,
        model,
        processor,
        num_register_tokens,
    )

    # normal_tokens:
    # [N_images, N_patches, C]

    print()
    print(
        "Normal token tensor:",
        normal_tokens.shape
    )

    memory = (
        normal_tokens.reshape(
            -1,
            hidden_size,
        )
    )

    print(
        "Memory bank:",
        memory.shape
    )

    print(
        "Patch grid:",
        (
            patch_height,
            patch_width,
        )
    )


    # --------------------------------------------------------
    # 4. Extract anomalous query patches
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )

    print(
        "QUERY EXTRACTION"
    )

    print(
        "=" * 70
    )

    (
        query_tokens,
        query_patch_height,
        query_patch_width,
    ) = extract_patch_tokens(
        [test_path],
        model,
        processor,
        num_register_tokens,
    )

    assert (
        query_patch_height
        == patch_height
    )

    assert (
        query_patch_width
        == patch_width
    )

    query = (
        query_tokens[
            0
        ]
    )

    print(
        "Query patches:",
        query.shape
    )


    # --------------------------------------------------------
    # 5. Patch NN cosine similarity
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )

    print(
        "PATCH NEAREST-NEIGHBOR"
    )

    print(
        "=" * 70
    )

    # Move only the NN computation to GPU.
    #
    # query:
    # [196, 384]
    #
    # memory:
    # [N_memory, 384]

    query_gpu = (
        query.to(
            DEVICE
        )
    )

    memory_gpu = (
        memory.to(
            DEVICE
        )
    )

    with torch.inference_mode():

        similarity = (
            query_gpu
            @ memory_gpu.T
        )

        max_similarity = (
            similarity.max(
                dim=1
            ).values
        )

        anomaly_scores = (
            1.0
            - max_similarity
        )

    anomaly_scores = (
        anomaly_scores
        .cpu()
    )

    anomaly_map = (
        anomaly_scores.reshape(
            patch_height,
            patch_width,
        )
    )

    print(
        "Similarity matrix:",
        similarity.shape
    )

    print(
        "Anomaly map:",
        anomaly_map.shape
    )

    print(
        "Anomaly score min:",
        anomaly_scores.min().item()
    )

    print(
        "Anomaly score mean:",
        anomaly_scores.mean().item()
    )

    print(
        "Anomaly score max:",
        anomaly_scores.max().item()
    )


    # --------------------------------------------------------
    # 6. Locate highest anomaly patch
    # --------------------------------------------------------

    max_index = int(
        torch.argmax(
            anomaly_scores
        ).item()
    )

    max_row = (
        max_index
        // patch_width
    )

    max_col = (
        max_index
        % patch_width
    )

    print()
    print(
        "Max anomaly patch index:",
        max_index
    )

    print(
        "Max anomaly patch row/col:",
        (
            max_row,
            max_col,
        )
    )


    # --------------------------------------------------------
    # 7. GT alignment
    # --------------------------------------------------------

    # Determine processor output size using
    # the same test image.

    test_image = Image.open(
        test_path
    ).convert("RGB")

    processed = processor(
        images=test_image,
        return_tensors="pt",
    )

    pixel_values = (
        processed[
            "pixel_values"
        ]
    )

    output_height = (
        pixel_values.shape[2]
    )

    output_width = (
        pixel_values.shape[3]
    )

    gt_pixel_mask = load_gt_mask(
        gt_path,
        output_height,
        output_width,
    )

    gt_patch_mask = (
        patch_gt_from_pixel_mask(
            gt_pixel_mask,
            patch_size,
        )
    )

    assert (
        gt_patch_mask.shape
        == (
            patch_height,
            patch_width,
        )
    )

    gt_patch_flat = (
        gt_patch_mask.reshape(
            -1
        )
    )

    anomaly_np = (
        anomaly_scores.numpy()
    )

    num_positive_patches = int(
        gt_patch_flat.sum()
    )

    max_patch_hits_gt = bool(
        gt_patch_flat[
            max_index
        ]
        > 0
    )

    positive_scores = (
        anomaly_np[
            gt_patch_flat > 0
        ]
    )

    negative_scores = (
        anomaly_np[
            gt_patch_flat == 0
        ]
    )

    patch_auroc = simple_roc_auc(
        gt_patch_flat,
        anomaly_np,
    )

    print()
    print(
        "=" * 70
    )

    print(
        "GT ALIGNMENT"
    )

    print(
        "=" * 70
    )

    print(
        "Positive GT patches:",
        num_positive_patches
    )

    print(
        "Max patch overlaps GT:",
        max_patch_hits_gt
    )

    if len(
        positive_scores
    ) > 0:

        print(
            "Mean anomaly on GT-positive:",
            float(
                positive_scores.mean()
            )
        )

    if len(
        negative_scores
    ) > 0:

        print(
            "Mean anomaly on GT-negative:",
            float(
                negative_scores.mean()
            )
        )

    print(
        "Patch AUROC:",
        patch_auroc
    )


    # --------------------------------------------------------
    # 8. Upsample anomaly map for visualization
    # --------------------------------------------------------

    anomaly_tensor = (
        anomaly_map
        .unsqueeze(0)
        .unsqueeze(0)
    )

    anomaly_up = F.interpolate(
        anomaly_tensor,
        size=(
            output_height,
            output_width,
        ),
        mode="bilinear",
        align_corners=False,
    )

    anomaly_up = (
        anomaly_up
        .squeeze()
        .numpy()
    )

    # Normalize only for DISPLAY.
    #
    # Do not use this normalized map for
    # quantitative metrics.

    display_min = (
        anomaly_up.min()
    )

    display_max = (
        anomaly_up.max()
    )

    display_map = (
        anomaly_up
        - display_min
    )

    if (
        display_max
        > display_min
    ):

        display_map = (
            display_map
            / (
                display_max
                - display_min
            )
        )


    # --------------------------------------------------------
    # 9. Visualization
    # --------------------------------------------------------

    image_display = (
        test_image.resize(
            (
                output_width,
                output_height,
            )
        )
    )

    figure_path = (
        OUTPUT_DIR
        / (
            f"{DEFECT_TYPE}_"
            f"{test_path.stem}_"
            f"anomaly_map.png"
        )
    )

    fig = plt.figure(
        figsize=(16, 4)
    )

    ax1 = fig.add_subplot(
        1,
        4,
        1
    )

    ax1.imshow(
        image_display
    )

    ax1.set_title(
        "Input"
    )

    ax1.axis(
        "off"
    )


    ax2 = fig.add_subplot(
        1,
        4,
        2
    )

    ax2.imshow(
        gt_pixel_mask
    )

    ax2.set_title(
        "Ground Truth"
    )

    ax2.axis(
        "off"
    )


    ax3 = fig.add_subplot(
        1,
        4,
        3
    )

    ax3.imshow(
        anomaly_map.numpy()
    )

    ax3.set_title(
        "14x14 Patch Anomaly"
    )

    ax3.axis(
        "off"
    )


    ax4 = fig.add_subplot(
        1,
        4,
        4
    )

    ax4.imshow(
        image_display
    )

    ax4.imshow(
        display_map,
        alpha=0.5,
    )

    ax4.set_title(
        "Overlay"
    )

    ax4.axis(
        "off"
    )


    fig.suptitle(
        (
            f"DINOv3 {CATEGORY}/{DEFECT_TYPE} | "
            f"max={anomaly_scores.max().item():.4f} | "
            f"patch AUROC={patch_auroc:.4f}"
        )
    )

    fig.tight_layout()

    fig.savefig(
        figure_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print()
    print(
        "Saved figure:",
        figure_path
    )


    # --------------------------------------------------------
    # 10. Save raw anomaly map
    # --------------------------------------------------------

    anomaly_npy_path = (
        OUTPUT_DIR
        / (
            f"{DEFECT_TYPE}_"
            f"{test_path.stem}_"
            f"anomaly_map.npy"
        )
    )

    np.save(
        anomaly_npy_path,
        anomaly_map.numpy(),
    )

    print(
        "Saved anomaly map:",
        anomaly_npy_path
    )


    # --------------------------------------------------------
    # 11. Save summary
    # --------------------------------------------------------

    summary = {
        "category":
            CATEGORY,

        "defect_type":
            DEFECT_TYPE,

        "test_image":
            test_path.name,

        "num_train_good":
            len(train_paths),

        "patch_grid":
            (
                f"{patch_height}"
                f"x{patch_width}"
            ),

        "memory_patches":
            memory.shape[0],

        "embedding_dim":
            hidden_size,

        "anomaly_min":
            anomaly_scores.min().item(),

        "anomaly_mean":
            anomaly_scores.mean().item(),

        "anomaly_max":
            anomaly_scores.max().item(),

        "max_patch_row":
            max_row,

        "max_patch_col":
            max_col,

        "gt_positive_patches":
            num_positive_patches,

        "max_patch_hits_gt":
            max_patch_hits_gt,

        "gt_positive_mean_score":
            (
                float(
                    positive_scores.mean()
                )
                if len(
                    positive_scores
                ) > 0
                else float("nan")
            ),

        "gt_negative_mean_score":
            (
                float(
                    negative_scores.mean()
                )
                if len(
                    negative_scores
                ) > 0
                else float("nan")
            ),

        "patch_auroc":
            patch_auroc,
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

    print(
        "Saved summary:",
        SUMMARY_PATH
    )


    # --------------------------------------------------------
    # 12. Final sanity checks
    # --------------------------------------------------------

    assert (
        patch_height
        == 14
    )

    assert (
        patch_width
        == 14
    )

    assert (
        query.shape
        == (
            196,
            hidden_size,
        )
    )

    assert (
        anomaly_map.shape
        == (
            14,
            14,
        )
    )

    print()
    print(
        "DINOv3 patch anomaly experiment PASS"
    )


if __name__ == "__main__":
    main()