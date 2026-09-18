from pathlib import Path

import torch
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights

from src.experiments.build_patch_memory_bank import (
    extract_layer2_feature,
    feature_map_to_patches,
)

from src.experiments.single_patch_nearest_neighbor import (
    build_memory_bank,
)


def extract_image_patches(
    image_path: Path,
    model,
    preprocess,
    device: torch.device,
):
    """
    Extract all layer2 patch features from one image.

    Returns:
        query_patches:
            [784, 128]

        height:
            28

        width:
            28
    """

    image = Image.open(
        image_path
    ).convert("RGB")

    x = preprocess(image)

    # [3,224,224]
    # ↓
    # [1,3,224,224]
    x = x.unsqueeze(0).to(device)

    with torch.no_grad():

        # [1,128,28,28]
        feature_map = extract_layer2_feature(
            model,
            x,
        )

        # [1,784,128]
        patches = feature_map_to_patches(
            feature_map
        )

    _, _, height, width = feature_map.shape

    # Remove batch dimension:
    #
    # [1,784,128]
    # ↓
    # [784,128]

    query_patches = patches[0].cpu()

    return (
        query_patches,
        height,
        width,
    )


def nearest_neighbor_scores_chunked(
    query_patches: torch.Tensor,
    memory_bank: torch.Tensor,
    memory_chunk_size: int = 4096,
):
    """
    For every query patch, find its nearest normal patch.

    query_patches:
        [Q, C]

    memory_bank:
        [M, C]

    Returns:
        best_distances:
            [Q]

        best_indices:
            [Q]
    """

    num_queries = query_patches.shape[0]

    # Initially:
    # every query's best distance = infinity
    best_distances = torch.full(
        (num_queries,),
        float("inf"),
        dtype=query_patches.dtype,
    )

    # Nearest memory index for each query
    best_indices = torch.full(
        (num_queries,),
        -1,
        dtype=torch.long,
    )

    num_memory_patches = memory_bank.shape[0]

    print("\nNearest-neighbor search:")

    # --------------------------------------------------------
    # Scan memory bank chunk by chunk
    # --------------------------------------------------------

    for start in range(
        0,
        num_memory_patches,
        memory_chunk_size,
    ):

        end = min(
            start + memory_chunk_size,
            num_memory_patches,
        )

        memory_chunk = memory_bank[
            start:end
        ]

        # query:
        # [784,128]
        #
        # memory chunk:
        # [chunk_size,128]
        #
        # distances:
        # [784,chunk_size]

        distances = torch.cdist(
            query_patches,
            memory_chunk,
            p=2,
        )

        # For every query patch:
        # nearest patch INSIDE this chunk
        chunk_min_distances, chunk_min_indices = (
            torch.min(
                distances,
                dim=1,
            )
        )

        # Determine which queries found a better
        # nearest neighbor in this chunk.
        better_mask = (
            chunk_min_distances
            < best_distances
        )

        # Update minimum distance
        best_distances[better_mask] = (
            chunk_min_distances[better_mask]
        )

        # Local chunk index
        # ↓
        # Global memory-bank index
        global_indices = (
            start
            + chunk_min_indices
        )

        best_indices[better_mask] = (
            global_indices[better_mask]
        )

        print(
            f"Memory patches "
            f"{start:6d} -> {end:6d} / "
            f"{num_memory_patches}"
        )

    return (
        best_distances,
        best_indices,
    )


def decode_patch_index(
    patch_index: int,
    width: int,
):
    """
    Flat patch index -> (row, col)
    """

    row = patch_index // width
    col = patch_index % width

    return row, col


def main():

    print("=" * 70)
    print("EXP011 - Full Image Patch Scoring")
    print("=" * 70)

    # ========================================================
    # 1. Device
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"\nDevice: {device}")

    # ========================================================
    # 2. Model
    # ========================================================

    weights = ResNet18_Weights.DEFAULT

    preprocess = weights.transforms()

    model = resnet18(
        weights=weights
    )

    model.eval()
    model.to(device)

    # ========================================================
    # 3. Build Normal Memory Bank
    # ========================================================

    train_good_dir = Path(
        "data/raw/mvtec_anomaly_detection/"
        "bottle/train/good"
    )

    print("\nBuilding normal memory bank...")

    memory_bank = build_memory_bank(
        model=model,
        preprocess=preprocess,
        train_good_dir=train_good_dir,
        device=device,
    )

    print("\nMemory bank shape:")
    print(memory_bank.shape)

    # ========================================================
    # 4. Test Image
    # ========================================================

    test_image_path = Path(
        "data/raw/mvtec_anomaly_detection/"
        "bottle/test/broken_small/000.png"
    )

    print("\nTest image:")
    print(test_image_path)

    # ========================================================
    # 5. Extract ALL Query Patches
    # ========================================================

    (
        query_patches,
        height,
        width,
    ) = extract_image_patches(
        image_path=test_image_path,
        model=model,
        preprocess=preprocess,
        device=device,
    )

    print("\nQuery patches shape:")
    print(query_patches.shape)

    print(
        f"\nFeature map size: "
        f"{height} x {width}"
    )

    # ========================================================
    # 6. Nearest Neighbor Search
    # ========================================================

    patch_scores, nearest_indices = (
        nearest_neighbor_scores_chunked(
            query_patches=query_patches,
            memory_bank=memory_bank,
            memory_chunk_size=4096,
        )
    )

    # ========================================================
    # 7. Verify Score Shape
    # ========================================================

    print("\n" + "=" * 70)
    print("Patch Score Result")
    print("=" * 70)

    print("\nPatch scores shape:")
    print(patch_scores.shape)

    print("\nNearest indices shape:")
    print(nearest_indices.shape)

    # ========================================================
    # 8. Score Statistics
    # ========================================================

    min_score = patch_scores.min()
    mean_score = patch_scores.mean()
    max_score = patch_scores.max()

    print("\nScore statistics:")

    print(
        f"Minimum score: "
        f"{min_score.item():.6f}"
    )

    print(
        f"Mean score: "
        f"{mean_score.item():.6f}"
    )

    print(
        f"Maximum score: "
        f"{max_score.item():.6f}"
    )

    # ========================================================
    # 9. Convert [784] -> [28,28]
    # ========================================================

    anomaly_map = patch_scores.reshape(
        height,
        width,
    )

    print("\nAnomaly map shape:")
    print(anomaly_map.shape)

    # ========================================================
    # 10. Find highest-scoring patch
    # ========================================================

    max_patch_index = torch.argmax(
        patch_scores
    ).item()

    max_row, max_col = decode_patch_index(
        max_patch_index,
        width,
    )

    max_nearest_memory_index = (
        nearest_indices[
            max_patch_index
        ].item()
    )

    print("\n" + "=" * 70)
    print("Highest-Scoring Patch")
    print("=" * 70)

    print(
        f"\nPatch index: "
        f"{max_patch_index}"
    )

    print(
        f"Spatial position: "
        f"({max_row}, {max_col})"
    )

    print(
        f"Anomaly score: "
        f"{patch_scores[max_patch_index].item():.6f}"
    )

    print(
        f"Nearest memory index: "
        f"{max_nearest_memory_index}"
    )

    # ========================================================
    # 11. Top-5 Highest Scores
    # ========================================================

    top_k = 5

    top_scores, top_indices = torch.topk(
        patch_scores,
        k=top_k,
    )

    print("\n" + "=" * 70)
    print("Top-5 Highest-Scoring Patches")
    print("=" * 70)

    for rank in range(top_k):

        patch_index = (
            top_indices[rank].item()
        )

        row, col = decode_patch_index(
            patch_index,
            width,
        )

        score = top_scores[
            rank
        ].item()

        print(
            f"\nRank {rank + 1}:"
            f"\n  patch index = {patch_index}"
            f"\n  position    = ({row}, {col})"
            f"\n  score       = {score:.6f}"
        )

    # ========================================================
    # 12. Verification
    # ========================================================

    expected_patch_shape = (
        height * width,
    )

    expected_map_shape = (
        height,
        width,
    )

    patch_shape_correct = (
        tuple(patch_scores.shape)
        == expected_patch_shape
    )

    map_shape_correct = (
        tuple(anomaly_map.shape)
        == expected_map_shape
    )

    print("\n" + "=" * 70)
    print("Verification")
    print("=" * 70)

    print(
        f"\nExpected patch score shape: "
        f"{expected_patch_shape}"
    )

    print(
        f"Actual patch score shape:   "
        f"{tuple(patch_scores.shape)}"
    )

    print(
        f"Patch score shape correct:  "
        f"{patch_shape_correct}"
    )

    print(
        f"\nExpected anomaly map shape: "
        f"{expected_map_shape}"
    )

    print(
        f"Actual anomaly map shape:   "
        f"{tuple(anomaly_map.shape)}"
    )

    print(
        f"Anomaly map shape correct:  "
        f"{map_shape_correct}"
    )

    if not patch_shape_correct:
        raise RuntimeError(
            "Patch score shape is incorrect."
        )

    if not map_shape_correct:
        raise RuntimeError(
            "Anomaly map shape is incorrect."
        )

    print("\n" + "=" * 70)
    print("Experiment completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()