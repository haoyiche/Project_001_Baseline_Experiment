import csv
import time
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
TEST_INDEX = 13

INPUT_SIZES = [
    224,
    448,
]

OUTPUT_DIR = Path(
    "results/exp024/dinov3_resolution_ablation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "resolution_ablation_summary.csv"
)

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# 448 is substantially heavier.
BATCH_SIZE_BY_SIZE = {
    224: 8,
    448: 2,
}

# Number of query patches processed
# against the full normal memory at once.
NN_QUERY_CHUNK = 32


# ============================================================
# Basic helpers
# ============================================================

def list_images(directory):

    return sorted(
        list(directory.glob("*.png"))
        + list(directory.glob("*.jpg"))
        + list(directory.glob("*.jpeg"))
    )


def resize_rgb(
    image,
    input_size,
):

    return image.resize(
        (
            input_size,
            input_size,
        ),
        resample=Image.Resampling.BILINEAR,
    )


def load_binary_gt(
    path,
):

    gt = Image.open(
        path
    ).convert("L")

    gt_np = np.asarray(
        gt,
        dtype=np.uint8,
    )

    return (
        gt_np > 0
    ).astype(
        np.uint8
    )


# ============================================================
# Metrics
# ============================================================

def roc_auc_binary(
    labels,
    scores,
):

    labels = np.asarray(
        labels,
        dtype=np.uint8,
    ).reshape(-1)

    scores = np.asarray(
        scores,
        dtype=np.float64,
    ).reshape(-1)

    positives = (
        labels == 1
    )

    negatives = (
        labels == 0
    )

    n_pos = int(
        positives.sum()
    )

    n_neg = int(
        negatives.sum()
    )

    if (
        n_pos == 0
        or n_neg == 0
    ):
        return float("nan")

    # Rank-based AUROC.
    # Handle tied scores using average ranks.

    order = np.argsort(
        scores,
        kind="mergesort",
    )

    sorted_scores = scores[
        order
    ]

    ranks = np.empty(
        len(scores),
        dtype=np.float64,
    )

    start = 0

    while start < len(scores):

        end = start + 1

        while (
            end < len(scores)
            and sorted_scores[end]
            == sorted_scores[start]
        ):
            end += 1

        # ranks are 1-based
        average_rank = (
            (start + 1)
            + end
        ) / 2.0

        ranks[
            order[start:end]
        ] = average_rank

        start = end

    positive_rank_sum = (
        ranks[
            positives
        ].sum()
    )

    auc = (
        positive_rank_sum
        - n_pos
        * (n_pos + 1)
        / 2.0
    ) / (
        n_pos
        * n_neg
    )

    return float(
        auc
    )


def average_precision_binary(
    labels,
    scores,
):

    labels = np.asarray(
        labels,
        dtype=np.uint8,
    ).reshape(-1)

    scores = np.asarray(
        scores,
        dtype=np.float64,
    ).reshape(-1)

    n_pos = int(
        labels.sum()
    )

    if n_pos == 0:
        return float("nan")

    order = np.argsort(
        -scores,
        kind="mergesort",
    )

    sorted_labels = labels[
        order
    ]

    tp = np.cumsum(
        sorted_labels
        == 1
    )

    fp = np.cumsum(
        sorted_labels
        == 0
    )

    precision = (
        tp
        / (
            tp
            + fp
        )
    )

    # AP = mean precision at each
    # positive retrieval position.
    ap = (
        precision[
            sorted_labels == 1
        ].sum()
        / n_pos
    )

    return float(
        ap
    )


# ============================================================
# DINOv3 feature extraction
# ============================================================

def extract_patch_tokens(
    image_paths,
    model,
    processor,
    input_size,
    batch_size,
):

    all_tokens = []

    patch_size = int(
        model.config.patch_size
    )

    hidden_size = int(
        model.config.hidden_size
    )

    num_register_tokens = int(
        model.config.num_register_tokens
    )

    assert (
        input_size
        % patch_size
        == 0
    )

    patch_height = (
        input_size
        // patch_size
    )

    patch_width = (
        input_size
        // patch_size
    )

    expected_patches = (
        patch_height
        * patch_width
    )

    for start in range(
        0,
        len(image_paths),
        batch_size,
    ):

        batch_paths = image_paths[
            start:
            start + batch_size
        ]

        images = []

        for path in batch_paths:

            image = Image.open(
                path
            ).convert("RGB")

            image = resize_rgb(
                image,
                input_size,
            )

            images.append(
                image
            )

        # We resize manually so the image and
        # experiment geometry are explicit.
        inputs = processor(
            images=images,
            do_resize=False,
            return_tensors="pt",
        )

        pixel_values = (
            inputs[
                "pixel_values"
            ]
        )

        assert (
            pixel_values.shape[-2:]
            == (
                input_size,
                input_size,
            )
        )

        inputs = {
            key: value.to(
                DEVICE
            )
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

        patch_tokens = hidden[
            :,
            1 + num_register_tokens:,
            :,
        ]

        assert (
            patch_tokens.shape[1]
            == expected_patches
        )

        assert (
            patch_tokens.shape[2]
            == hidden_size
        )

        patch_tokens = F.normalize(
            patch_tokens,
            p=2,
            dim=-1,
        )

        all_tokens.append(
            patch_tokens.cpu()
        )

        print(
            f"[{input_size}] "
            f"Extracted "
            f"{min(start + batch_size, len(image_paths))}"
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


# ============================================================
# Chunked nearest neighbor
# ============================================================

def cosine_nn_anomaly_scores(
    query,
    memory,
    query_chunk_size,
):

    # Both are already L2-normalized.
    #
    # query:
    # [N_query, C]
    #
    # memory:
    # [N_memory, C]

    memory_gpu = memory.to(
        DEVICE
    )

    results = []

    for start in range(
        0,
        query.shape[0],
        query_chunk_size,
    ):

        end = min(
            start + query_chunk_size,
            query.shape[0],
        )

        query_chunk = (
            query[
                start:end
            ].to(
                DEVICE
            )
        )

        with torch.inference_mode():

            similarity = (
                query_chunk
                @ memory_gpu.T
            )

            max_similarity = (
                similarity.max(
                    dim=1
                ).values
            )

            scores = (
                1.0
                - max_similarity
            )

        results.append(
            scores.cpu()
        )

        print(
            "NN patches:",
            f"{end}/{query.shape[0]}"
        )

        del (
            query_chunk,
            similarity,
            max_similarity,
            scores,
        )

    del memory_gpu

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return torch.cat(
        results,
        dim=0,
    )


# ============================================================
# Evaluation
# ============================================================

def evaluate_pixel_map(
    anomaly_map,
    original_height,
    original_width,
    gt_mask,
):

    anomaly_tensor = (
        anomaly_map
        .unsqueeze(0)
        .unsqueeze(0)
    )

    anomaly_up = F.interpolate(
        anomaly_tensor,
        size=(
            original_height,
            original_width,
        ),
        mode="bilinear",
        align_corners=False,
    )

    anomaly_up = (
        anomaly_up
        .squeeze()
        .numpy()
    )

    assert (
        anomaly_up.shape
        == gt_mask.shape
    )

    gt_flat = (
        gt_mask.reshape(-1)
    )

    score_flat = (
        anomaly_up.reshape(-1)
    )

    pixel_auc = roc_auc_binary(
        gt_flat,
        score_flat,
    )

    pixel_ap = average_precision_binary(
        gt_flat,
        score_flat,
    )

    positive_scores = (
        score_flat[
            gt_flat == 1
        ]
    )

    negative_scores = (
        score_flat[
            gt_flat == 0
        ]
    )

    return {
        "anomaly_up":
            anomaly_up,

        "pixel_auroc":
            pixel_auc,

        "pixel_ap":
            pixel_ap,

        "positive_mean":
            float(
                positive_scores.mean()
            ),

        "negative_mean":
            float(
                negative_scores.mean()
            ),
    }


# ============================================================
# One resolution
# ============================================================

def run_resolution(
    input_size,
    train_paths,
    test_path,
    gt_mask,
    model,
    processor,
):

    batch_size = (
        BATCH_SIZE_BY_SIZE[
            input_size
        ]
    )

    print()
    print(
        "=" * 80
    )

    print(
        f"RESOLUTION {input_size}x{input_size}"
    )

    print(
        "=" * 80
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start_time = time.perf_counter()


    # --------------------------------------------------------
    # Normal memory
    # --------------------------------------------------------

    print()
    print(
        "Building normal memory..."
    )

    (
        normal_tokens,
        patch_height,
        patch_width,
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

    print()
    print(
        "Normal tensor:",
        normal_tokens.shape
    )

    print(
        "Memory bank:",
        memory.shape
    )


    # --------------------------------------------------------
    # Query
    # --------------------------------------------------------

    print()
    print(
        "Extracting query..."
    )

    (
        query_tokens,
        query_patch_height,
        query_patch_width,
    ) = extract_patch_tokens(
        [test_path],
        model,
        processor,
        input_size,
        batch_size=1,
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
    # Chunked NN
    # --------------------------------------------------------

    print()
    print(
        "Running chunked cosine NN..."
    )

    anomaly_scores = (
        cosine_nn_anomaly_scores(
            query,
            memory,
            NN_QUERY_CHUNK,
        )
    )

    anomaly_map = (
        anomaly_scores.reshape(
            patch_height,
            patch_width,
        )
    )


    # --------------------------------------------------------
    # Original image geometry
    # --------------------------------------------------------

    original_image = Image.open(
        test_path
    ).convert("RGB")

    original_width = (
        original_image.width
    )

    original_height = (
        original_image.height
    )

    assert (
        gt_mask.shape
        == (
            original_height,
            original_width,
        )
    )


    # --------------------------------------------------------
    # Pixel evaluation
    # --------------------------------------------------------

    evaluation = (
        evaluate_pixel_map(
            anomaly_map,
            original_height,
            original_width,
            gt_mask,
        )
    )


    # --------------------------------------------------------
    # Timing / VRAM
    # --------------------------------------------------------

    elapsed = (
        time.perf_counter()
        - start_time
    )

    if torch.cuda.is_available():

        peak_memory_mib = (
            torch.cuda.max_memory_allocated()
            / 1024
            / 1024
        )

    else:

        peak_memory_mib = (
            float("nan")
        )


    # --------------------------------------------------------
    # Save anomaly map
    # --------------------------------------------------------

    npy_path = (
        OUTPUT_DIR
        / f"anomaly_map_{input_size}.npy"
    )

    np.save(
        npy_path,
        anomaly_map.numpy(),
    )


    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print()
    print(
        "-" * 80
    )

    print(
        f"RESULT {input_size}"
    )

    print(
        "-" * 80
    )

    print(
        "Patch grid:",
        (
            patch_height,
            patch_width,
        )
    )

    print(
        "Query patches:",
        query.shape[0]
    )

    print(
        "Memory patches:",
        memory.shape[0]
    )

    print(
        "Anomaly min:",
        anomaly_scores.min().item()
    )

    print(
        "Anomaly mean:",
        anomaly_scores.mean().item()
    )

    print(
        "Anomaly max:",
        anomaly_scores.max().item()
    )

    print(
        "Pixel AUROC:",
        evaluation[
            "pixel_auroc"
        ]
    )

    print(
        "Pixel AP:",
        evaluation[
            "pixel_ap"
        ]
    )

    print(
        "Pixel GT-positive mean:",
        evaluation[
            "positive_mean"
        ]
    )

    print(
        "Pixel GT-negative mean:",
        evaluation[
            "negative_mean"
        ]
    )

    print(
        "Elapsed seconds:",
        elapsed
    )

    print(
        "Peak allocated VRAM MiB:",
        peak_memory_mib
    )


    return {
        "input_size":
            input_size,

        "patch_grid":
            f"{patch_height}x{patch_width}",

        "query_patches":
            query.shape[0],

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

        "pixel_auroc":
            evaluation[
                "pixel_auroc"
            ],

        "pixel_ap":
            evaluation[
                "pixel_ap"
            ],

        "pixel_gt_positive_mean":
            evaluation[
                "positive_mean"
            ],

        "pixel_gt_negative_mean":
            evaluation[
                "negative_mean"
            ],

        "elapsed_seconds":
            elapsed,

        "peak_allocated_vram_mib":
            peak_memory_mib,

        "anomaly_up":
            evaluation[
                "anomaly_up"
            ],

        "anomaly_map":
            anomaly_map.numpy(),
    }


# ============================================================
# Visualization
# ============================================================

def save_comparison_figure(
    test_path,
    gt_mask,
    results,
):

    image = Image.open(
        test_path
    ).convert("RGB")

    fig = plt.figure(
        figsize=(15, 8)
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
        "Input"
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
        gt_mask
    )

    ax2.set_title(
        "Ground Truth"
    )

    ax2.axis(
        "off"
    )


    for index, result in enumerate(
        results
    ):

        ax_patch = fig.add_subplot(
            2,
            3,
            3 + index
        )

        ax_patch.imshow(
            result[
                "anomaly_map"
            ]
        )

        ax_patch.set_title(
            (
                f"{result['input_size']} "
                f"patch map "
                f"{result['patch_grid']}"
            )
        )

        ax_patch.axis(
            "off"
        )


    ax_overlay = fig.add_subplot(
        2,
        3,
        5
    )

    ax_overlay.imshow(
        image
    )

    ax_overlay.imshow(
        results[0][
            "anomaly_up"
        ],
        alpha=0.5,
    )

    ax_overlay.set_title(
        (
            f"224 overlay\n"
            f"AP="
            f"{results[0]['pixel_ap']:.4f}"
        )
    )

    ax_overlay.axis(
        "off"
    )


    ax_overlay2 = fig.add_subplot(
        2,
        3,
        6
    )

    ax_overlay2.imshow(
        image
    )

    ax_overlay2.imshow(
        results[1][
            "anomaly_up"
        ],
        alpha=0.5,
    )

    ax_overlay2.set_title(
        (
            f"448 overlay\n"
            f"AP="
            f"{results[1]['pixel_ap']:.4f}"
        )
    )

    ax_overlay2.axis(
        "off"
    )

    fig.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "resolution_224_vs_448.png"
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
        "Saved comparison:",
        output_path
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
    # Dataset
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

    assert train_paths
    assert test_paths

    test_path = (
        test_paths[
            TEST_INDEX
        ]
    )

    gt_path = (
        gt_dir
        / f"{test_path.stem}_mask.png"
    )

    assert gt_path.exists()

    gt_mask = load_binary_gt(
        gt_path
    )

    original_image = Image.open(
        test_path
    )

    print(
        "Train good:",
        len(train_paths)
    )

    print(
        "Test:",
        test_path
    )

    print(
        "GT:",
        gt_path
    )

    print(
        "Original image:",
        original_image.size
    )

    print(
        "GT shape:",
        gt_mask.shape
    )

    assert (
        gt_mask.shape
        == (
            original_image.height,
            original_image.width,
        )
    )


    # --------------------------------------------------------
    # Model
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

    print()
    print(
        "Patch size:",
        model.config.patch_size
    )

    print(
        "Hidden size:",
        model.config.hidden_size
    )

    print(
        "Register tokens:",
        model.config.num_register_tokens
    )


    # --------------------------------------------------------
    # Run both resolutions
    # --------------------------------------------------------

    results = []

    for input_size in (
        INPUT_SIZES
    ):

        result = run_resolution(
            input_size,
            train_paths,
            test_path,
            gt_mask,
            model,
            processor,
        )

        results.append(
            result
        )


    # --------------------------------------------------------
    # Save figure
    # --------------------------------------------------------

    save_comparison_figure(
        test_path,
        gt_mask,
        results,
    )


    # --------------------------------------------------------
    # Save CSV without arrays
    # --------------------------------------------------------

    csv_rows = []

    for result in results:

        csv_rows.append(
            {
                key: value
                for key, value
                in result.items()
                if key not in {
                    "anomaly_up",
                    "anomaly_map",
                }
            }
        )

    with open(
        SUMMARY_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=csv_rows[
                0
            ].keys(),
        )

        writer.writeheader()
        writer.writerows(
            csv_rows
        )


    # --------------------------------------------------------
    # Final comparison
    # --------------------------------------------------------

    r224 = results[0]
    r448 = results[1]

    print()
    print(
        "=" * 80
    )

    print(
        "224 VS 448 COMPARISON"
    )

    print(
        "=" * 80
    )

    print(
        "Pixel AUROC:"
    )

    print(
        "224:",
        r224[
            "pixel_auroc"
        ]
    )

    print(
        "448:",
        r448[
            "pixel_auroc"
        ]
    )

    print(
        "Delta:",
        (
            r448[
                "pixel_auroc"
            ]
            - r224[
                "pixel_auroc"
            ]
        )
    )

    print()

    print(
        "Pixel AP:"
    )

    print(
        "224:",
        r224[
            "pixel_ap"
        ]
    )

    print(
        "448:",
        r448[
            "pixel_ap"
        ]
    )

    print(
        "Delta:",
        (
            r448[
                "pixel_ap"
            ]
            - r224[
                "pixel_ap"
            ]
        )
    )

    print()

    print(
        "Runtime ratio "
        "448 / 224:",
        (
            r448[
                "elapsed_seconds"
            ]
            / r224[
                "elapsed_seconds"
            ]
        )
    )

    print()

    print(
        "Memory patch ratio "
        "448 / 224:",
        (
            r448[
                "memory_patches"
            ]
            / r224[
                "memory_patches"
            ]
        )
    )

    print()
    print(
        "Saved:",
        SUMMARY_PATH
    )


if __name__ == "__main__":
    main()