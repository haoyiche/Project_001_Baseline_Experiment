from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights

from src.experiments.build_patch_memory_bank import (
    NormalBottleDataset,
    extract_layer2_feature,
    feature_map_to_patches,
)


def build_memory_bank(
    model,
    preprocess,
    train_good_dir: Path,
    device: torch.device,
) -> torch.Tensor:
    """
    Build normal patch memory bank.

    Returns:
        [N_patches, 128]
    """

    dataset = NormalBottleDataset(
        root_dir=train_good_dir,
        transform=preprocess,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
    )

    memory_bank_parts = []

    with torch.no_grad():

        for images, _ in dataloader:

            images = images.to(device)

            # [B, 128, 28, 28]
            feature_map = extract_layer2_feature(
                model,
                images,
            )

            # [B, 784, 128]
            patches = feature_map_to_patches(
                feature_map
            )

            # [B*784, 128]
            patches = patches.reshape(
                -1,
                patches.shape[-1],
            )

            memory_bank_parts.append(
                patches.cpu()
            )

    memory_bank = torch.cat(
        memory_bank_parts,
        dim=0,
    )

    return memory_bank


def extract_query_patch(
    image_path: Path,
    model,
    preprocess,
    device: torch.device,
    row: int,
    col: int,
):
    """
    Extract one layer2 patch from one image.
    """

    image = Image.open(image_path).convert("RGB")

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

    patch_index = row * width + col

    # [128]
    query_patch = patches[
        0,
        patch_index,
        :
    ]

    return (
        query_patch.cpu(),
        patch_index,
        height,
        width,
    )


def main():

    print("=" * 70)
    print("EXP010 - Single Patch Nearest Neighbor")
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
    # 3. Build Memory Bank
    # ========================================================

    train_good_dir = Path(
        "data/raw/mvtec_anomaly_detection/"
        "bottle/train/good"
    )

    print("\nBuilding memory bank...")

    memory_bank = build_memory_bank(
        model=model,
        preprocess=preprocess,
        train_good_dir=train_good_dir,
        device=device,
    )

    print("\nMemory bank shape:")
    print(memory_bank.shape)

    # ========================================================
    # 4. Query image
    #
    # IMPORTANT:
    # 先用训练集第一张图做 sanity check
    # ========================================================

    query_image_path = Path(
        "data/raw/mvtec_anomaly_detection/"
        "bottle/test/good/000.png"
    )

    row = 10
    col = 5

    query_patch, patch_index, height, width = (
        extract_query_patch(
            image_path=query_image_path,
            model=model,
            preprocess=preprocess,
            device=device,
            row=row,
            col=col,
        )
    )

    print("\n" + "=" * 70)
    print("Query Patch")
    print("=" * 70)

    print("\nQuery image:")
    print(query_image_path)

    print(
        f"\nFeature map size: "
        f"{height} x {width}"
    )

    print(
        f"Spatial position: "
        f"({row}, {col})"
    )

    print(
        f"Patch index: "
        f"{patch_index}"
    )

    print("\nQuery patch shape:")
    print(query_patch.shape)

    print("\nFirst 10 query values:")
    print(query_patch[:10])

    # ========================================================
    # 5. Compute distance to EVERY normal patch
    # ========================================================

    # memory_bank:
    # [163856,128]
    #
    # query_patch:
    # [128]
    #
    # Broadcasting:
    #
    # [163856,128]
    # -
    # [128]
    #
    # →
    # [163856,128]

    difference = (
        memory_bank
        - query_patch.unsqueeze(0)
    )

    print("\nDifference shape:")
    print(difference.shape)

    # Euclidean distance:
    #
    # sqrt(
    #     sum((m - q)^2)
    # )
    #
    # dim=1 means:
    # calculate one distance per memory patch

    distances = torch.norm(
        difference,
        p=2,
        dim=1,
    )

    print("\nDistance vector shape:")
    print(distances.shape)

    # ========================================================
    # 6. Nearest Neighbor
    # ========================================================

    min_distance, nearest_index = (
        torch.min(
            distances,
            dim=0,
        )
    )

    print("\n" + "=" * 70)
    print("Nearest Neighbor Result")
    print("=" * 70)

    print(
        f"\nNearest memory index: "
        f"{nearest_index.item()}"
    )

    print(
        f"Minimum distance: "
        f"{min_distance.item():.10f}"
    )

    # ========================================================
    # 7. Verify nearest feature
    # ========================================================

    nearest_patch = memory_bank[
        nearest_index
    ]

    max_abs_error = torch.max(
        torch.abs(
            query_patch
            - nearest_patch
        )
    )

    print("\nMaximum absolute feature error:")
    print(max_abs_error.item())

    # ========================================================
    # 8. Prediction verification
    # ========================================================

    print("\n" + "=" * 70)
    print("Prediction Verification")
    print("=" * 70)

    print("\nPrediction:")
    print("Nearest memory index = 285")
    print("Minimum distance = 0.0")

    print("\nActual:")
    print(
        f"Nearest memory index = "
        f"{nearest_index.item()}"
    )

    print(
        f"Minimum distance = "
        f"{min_distance.item():.10f}"
    )

    index_correct = (
        nearest_index.item()
        == patch_index
    )

    distance_zero = torch.isclose(
        min_distance,
        torch.tensor(0.0),
        atol=1e-6,
    ).item()

    print(
        f"\nIndex prediction correct: "
        f"{index_correct}"
    )

    print(
        f"Distance prediction correct: "
        f"{distance_zero}"
    )

    print("\n" + "=" * 70)
    print("Experiment completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()