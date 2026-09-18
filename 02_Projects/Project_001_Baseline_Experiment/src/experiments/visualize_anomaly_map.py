from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image

from torchvision.models import (
    resnet18,
    ResNet18_Weights,
)

from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

from src.experiments.single_patch_nearest_neighbor import (
    build_memory_bank,
)

from src.experiments.full_image_patch_scoring import (
    extract_image_patches,
    nearest_neighbor_scores_chunked,
    decode_patch_index,
)


def prepare_display_image(
    image_path: Path,
    preprocess,
):
    """
    Apply EXACTLY the same model preprocessing,
    then undo normalization only.

    Result:
        display image aligned with model input
        shape = [224,224,3]
    """

    image = Image.open(
        image_path
    ).convert("RGB")

    # This includes:
    # resize
    # center crop
    # tensor conversion
    # normalization
    x = preprocess(image)

    # --------------------------------------------------------
    # Undo ImageNet normalization
    # so we can visualize the actual 224x224 model input.
    # --------------------------------------------------------

    mean = torch.tensor(
        preprocess.mean,
        dtype=x.dtype,
    ).view(3, 1, 1)

    std = torch.tensor(
        preprocess.std,
        dtype=x.dtype,
    ).view(3, 1, 1)

    display = (
        x * std + mean
    ).clamp(0, 1)

    # [3,H,W]
    # ↓
    # [H,W,3]
    display = display.permute(
        1,
        2,
        0,
    )

    return display


def prepare_ground_truth(
    mask_path: Path,
    preprocess,
):
    """
    Apply the SAME geometric transforms as the
    ResNet preprocessing, but do not normalize.

    Ground-truth mask uses nearest-neighbor resize.
    """

    mask = Image.open(
        mask_path
    ).convert("L")

    # PIL -> Tensor
    #
    # [1,H,W]
    mask = TF.pil_to_tensor(
        mask
    ).float() / 255.0

    # --------------------------------------------------------
    # Same resize geometry as model preprocessing
    # --------------------------------------------------------

    mask = TF.resize(
        mask,
        size=preprocess.resize_size,
        interpolation=InterpolationMode.NEAREST,
        antialias=False,
    )

    # --------------------------------------------------------
    # Same center crop
    # --------------------------------------------------------

    mask = TF.center_crop(
        mask,
        output_size=preprocess.crop_size,
    )

    # [1,224,224]
    # ↓
    # [224,224]

    mask = mask.squeeze(0)

    # Convert to binary mask
    mask = (
        mask > 0.5
    ).float()

    return mask


def upsample_anomaly_map(
    anomaly_map: torch.Tensor,
    output_height: int,
    output_width: int,
):
    """
    [28,28]
        ↓
    [224,224]
    """

    x = anomaly_map.unsqueeze(0).unsqueeze(0)

    upsampled = F.interpolate(
        x,
        size=(
            output_height,
            output_width,
        ),
        mode="bilinear",
        align_corners=False,
    )

    return upsampled[0, 0]


def normalize_for_visualization(
    score_map: torch.Tensor,
):
    """
    Min-max normalization ONLY for visualization.

    Important:
    This does NOT change the raw anomaly scores
    used for evaluation.
    """

    minimum = score_map.min()
    maximum = score_map.max()

    normalized = (
        score_map - minimum
    ) / (
        maximum - minimum + 1e-8
    )

    return normalized


def get_gt_bbox(
    gt_mask: torch.Tensor,
):
    """
    Return GT bounding box:

        (x_min, y_min, x_max, y_max)

    in 224x224 model-input coordinates.
    """

    positions = torch.nonzero(
        gt_mask > 0.5,
        as_tuple=False,
    )

    if len(positions) == 0:
        return None

    y_min = positions[:, 0].min().item()
    y_max = positions[:, 0].max().item()

    x_min = positions[:, 1].min().item()
    x_max = positions[:, 1].max().item()

    return (
        x_min,
        y_min,
        x_max,
        y_max,
    )


def main():

    print("=" * 70)
    print("EXP012 - Anomaly Map Visualization")
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

    print("\nModel preprocessing:")

    print(
        f"Resize size: "
        f"{preprocess.resize_size}"
    )

    print(
        f"Crop size: "
        f"{preprocess.crop_size}"
    )

    # ========================================================
    # 3. Paths
    # ========================================================

    train_good_dir = Path(
        "data/raw/mvtec_anomaly_detection/"
        "bottle/train/good"
    )

    test_image_path = Path(
        "data/raw/mvtec_anomaly_detection/"
        "bottle/test/broken_small/000.png"
    )

    ground_truth_path = Path(
        "data/raw/mvtec_anomaly_detection/"
        "bottle/ground_truth/broken_small/000_mask.png"
    )

    if not test_image_path.exists():
        raise FileNotFoundError(
            test_image_path
        )

    if not ground_truth_path.exists():
        raise FileNotFoundError(
            ground_truth_path
        )

    print("\nTest image:")
    print(test_image_path)

    print("\nGround truth:")
    print(ground_truth_path)

    # ========================================================
    # 4. Build Normal Memory Bank
    # ========================================================

    print(
        "\nBuilding normal memory bank..."
    )

    memory_bank = build_memory_bank(
        model=model,
        preprocess=preprocess,
        train_good_dir=train_good_dir,
        device=device,
    )

    print("\nMemory bank shape:")
    print(memory_bank.shape)

    # ========================================================
    # 5. Extract Query Patches
    # ========================================================

    (
        query_patches,
        feature_height,
        feature_width,
    ) = extract_image_patches(
        image_path=test_image_path,
        model=model,
        preprocess=preprocess,
        device=device,
    )

    print("\nQuery patches shape:")
    print(query_patches.shape)

    # ========================================================
    # 6. Calculate Patch Anomaly Scores
    # ========================================================

    patch_scores, nearest_indices = (
        nearest_neighbor_scores_chunked(
            query_patches=query_patches,
            memory_bank=memory_bank,
            memory_chunk_size=4096,
        )
    )

    print("\nPatch scores shape:")
    print(patch_scores.shape)

    # ========================================================
    # 7. [784] -> [28,28]
    # ========================================================

    anomaly_map = patch_scores.reshape(
        feature_height,
        feature_width,
    )

    print("\nLow-resolution anomaly map:")
    print(anomaly_map.shape)

    # ========================================================
    # 8. Prepare model-input visualization
    # ========================================================

    display_image = prepare_display_image(
        image_path=test_image_path,
        preprocess=preprocess,
    )

    display_height = (
        display_image.shape[0]
    )

    display_width = (
        display_image.shape[1]
    )

    print("\nDisplay image shape:")
    print(display_image.shape)

    # ========================================================
    # 9. Upsample anomaly map
    # ========================================================

    upsampled_map = upsample_anomaly_map(
        anomaly_map,
        output_height=display_height,
        output_width=display_width,
    )

    print("\nUpsampled anomaly map:")
    print(upsampled_map.shape)

    # Visualization-only normalization
    heatmap = normalize_for_visualization(
        upsampled_map
    )

    # ========================================================
    # 10. Prepare GT with identical geometry
    # ========================================================

    gt_mask = prepare_ground_truth(
        mask_path=ground_truth_path,
        preprocess=preprocess,
    )

    print("\nGround-truth mask shape:")
    print(gt_mask.shape)

    print(
        f"Ground-truth positive pixels: "
        f"{int(gt_mask.sum().item())}"
    )

    gt_bbox = get_gt_bbox(
        gt_mask
    )

    print("\nGround-truth bbox:")
    print(gt_bbox)

    # ========================================================
    # 11. Highest-scoring Patch
    # ========================================================

    max_patch_index = torch.argmax(
        patch_scores
    ).item()

    max_row, max_col = decode_patch_index(
        max_patch_index,
        feature_width,
    )

    max_score = patch_scores[
        max_patch_index
    ].item()

    print("\n" + "=" * 70)
    print("Highest-Scoring Patch")
    print("=" * 70)

    print(
        f"\nPatch index: "
        f"{max_patch_index}"
    )

    print(
        f"Feature-map position: "
        f"({max_row}, {max_col})"
    )

    print(
        f"Score: "
        f"{max_score:.6f}"
    )

    # --------------------------------------------------------
    # Approximate center location in 224x224 input space
    # --------------------------------------------------------

    scale_y = (
        display_height
        / feature_height
    )

    scale_x = (
        display_width
        / feature_width
    )

    center_y = int(
        (max_row + 0.5)
        * scale_y
    )

    center_x = int(
        (max_col + 0.5)
        * scale_x
    )

    center_y = min(
        center_y,
        display_height - 1,
    )

    center_x = min(
        center_x,
        display_width - 1,
    )

    print(
        f"Approx input-space center: "
        f"(x={center_x}, y={center_y})"
    )

    # This is only a diagnostic.
    # CNN feature receptive field is larger than one
    # simple 8x8 cell.

    center_inside_gt = bool(
        gt_mask[
            center_y,
            center_x,
        ].item()
        > 0.5
    )

    print(
        f"Patch center inside GT mask: "
        f"{center_inside_gt}"
    )

    # ========================================================
    # 12. Top-5 Patch Centers
    # ========================================================

    top_scores, top_indices = (
        torch.topk(
            patch_scores,
            k=5,
        )
    )

    print("\n" + "=" * 70)
    print("Top-5 Patch / Ground Truth Check")
    print("=" * 70)

    for rank in range(5):

        patch_index = (
            top_indices[rank].item()
        )

        score = (
            top_scores[rank].item()
        )

        row, col = decode_patch_index(
            patch_index,
            feature_width,
        )

        y = int(
            (row + 0.5)
            * scale_y
        )

        x = int(
            (col + 0.5)
            * scale_x
        )

        y = min(
            y,
            display_height - 1,
        )

        x = min(
            x,
            display_width - 1,
        )

        inside_gt = bool(
            gt_mask[
                y,
                x,
            ].item()
            > 0.5
        )

        print(
            f"\nRank {rank + 1}:"
            f"\n  patch = {patch_index}"
            f"\n  feature position = ({row},{col})"
            f"\n  input center = (x={x}, y={y})"
            f"\n  score = {score:.6f}"
            f"\n  center inside GT = {inside_gt}"
        )

    # ========================================================
    # 13. Save Visualization
    # ========================================================

    output_dir = Path(
        "results/exp012"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "broken_small_000_anomaly_map.png"
    )

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(16, 4),
    )

    # --------------------------------------------------------
    # Original model input
    # --------------------------------------------------------

    axes[0].imshow(
        display_image.numpy()
    )

    axes[0].set_title(
        "Model Input"
    )

    axes[0].axis("off")

    # --------------------------------------------------------
    # Anomaly heatmap
    # --------------------------------------------------------

    im = axes[1].imshow(
        heatmap.numpy(),
        cmap="jet",
    )

    axes[1].set_title(
        "Anomaly Map"
    )

    axes[1].axis("off")

    fig.colorbar(
        im,
        ax=axes[1],
        fraction=0.046,
        pad=0.04,
    )

    # --------------------------------------------------------
    # Overlay
    # --------------------------------------------------------

    axes[2].imshow(
        display_image.numpy()
    )

    axes[2].imshow(
        heatmap.numpy(),
        cmap="jet",
        alpha=0.5,
    )

    axes[2].set_title(
        "Image + Anomaly"
    )

    axes[2].axis("off")

    # --------------------------------------------------------
    # Ground Truth
    # --------------------------------------------------------

    axes[3].imshow(
        display_image.numpy()
    )

    axes[3].imshow(
        gt_mask.numpy(),
        cmap="Reds",
        alpha=0.5,
    )

    axes[3].set_title(
        "Ground Truth"
    )

    axes[3].axis("off")

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("\n" + "=" * 70)
    print("Saved Result")
    print("=" * 70)

    print("\nVisualization saved to:")
    print(output_path.resolve())

    # ========================================================
    # 14. Save raw score data
    # ========================================================

    raw_output_path = (
        output_dir
        / "broken_small_000_scores.pt"
    )

    torch.save(
        {
            "patch_scores": patch_scores,
            "anomaly_map": anomaly_map,
            "upsampled_map": upsampled_map,
            "gt_mask": gt_mask,
            "nearest_indices": nearest_indices,
        },
        raw_output_path,
    )

    print("\nRaw scores saved to:")
    print(raw_output_path.resolve())

    print("\n" + "=" * 70)
    print("Experiment completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()