from pathlib import Path

import torch
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights


def feature_map_to_patches(feature_map: torch.Tensor) -> torch.Tensor:
    """
    Convert feature map:

        [B, C, H, W]

    to patch features:

        [B, H*W, C]

    Each spatial position (h, w) becomes one local feature vector.
    """

    if feature_map.ndim != 4:
        raise ValueError(
            f"Expected feature map with 4 dimensions [B,C,H,W], "
            f"but got shape {feature_map.shape}"
        )

    batch_size, channels, height, width = feature_map.shape

    # [B, C, H, W]
    #       ↓
    # [B, H, W, C]
    patches = feature_map.permute(0, 2, 3, 1)

    # [B, H, W, C]
    #       ↓
    # [B, H*W, C]
    patches = patches.reshape(
        batch_size,
        height * width,
        channels,
    )

    return patches


def main():

    # =========================================================
    # 1. Experiment information
    # =========================================================

    print("=" * 60)
    print("EXP008 - ResNet18 Patch Feature Inspection")
    print("=" * 60)

    # 修改成你本地真实存在的一张 bottle 图片
    image_path = Path(
        "data/raw/mvtec_anomaly_detection/"
        "bottle/train/good/000.png"
    )

    if not image_path.exists():
        raise FileNotFoundError(
            f"\nImage not found:\n{image_path.resolve()}\n"
            f"Please modify image_path in inspect_patch_features.py"
        )

    print(f"\nImage:")
    print(image_path.resolve())

    # =========================================================
    # 2. Load pretrained ResNet18
    # =========================================================

    weights = ResNet18_Weights.DEFAULT

    model = resnet18(weights=weights)

    # Inference mode:
    # BatchNorm / Dropout use evaluation behavior
    model.eval()

    print("\nModel:")
    print("ResNet18 pretrained on ImageNet")

    # =========================================================
    # 3. Image preprocessing
    # =========================================================

    # Official preprocessing corresponding to pretrained weights
    preprocess = weights.transforms()

    image = Image.open(image_path).convert("RGB")

    # PIL
    # ↓
    # Tensor
    #
    # [3, 224, 224]
    x = preprocess(image)

    print("\nBefore adding batch dimension:")
    print(x.shape)

    # [C, H, W]
    # ↓
    # [B, C, H, W]
    #
    # [3, 224, 224]
    # ↓
    # [1, 3, 224, 224]
    x = x.unsqueeze(0)

    print("\nInput shape:")
    print(x.shape)

    # =========================================================
    # 4. Extract intermediate ResNet features
    # =========================================================

    with torch.no_grad():

        # -----------------------------------------------------
        # ResNet stem
        # -----------------------------------------------------

        x = model.conv1(x)

        print("\nAfter conv1:")
        print(x.shape)

        x = model.bn1(x)
        x = model.relu(x)

        x = model.maxpool(x)

        print("\nAfter maxpool:")
        print(x.shape)

        # -----------------------------------------------------
        # layer1
        # -----------------------------------------------------

        layer1_feature = model.layer1(x)

        print("\nLayer1 feature map:")
        print(layer1_feature.shape)

        # -----------------------------------------------------
        # layer2
        # -----------------------------------------------------

        layer2_feature = model.layer2(layer1_feature)

        print("\nLayer2 feature map:")
        print(layer2_feature.shape)

        # -----------------------------------------------------
        # layer3
        # -----------------------------------------------------

        layer3_feature = model.layer3(layer2_feature)

        print("\nLayer3 feature map:")
        print(layer3_feature.shape)

        # -----------------------------------------------------
        # layer4
        #
        # 这里只为了观察，不用于本次 patch baseline
        # -----------------------------------------------------

        layer4_feature = model.layer4(layer3_feature)

        print("\nLayer4 feature map:")
        print(layer4_feature.shape)

    # =========================================================
    # 5. Convert feature maps to patch features
    # =========================================================

    layer2_patches = feature_map_to_patches(layer2_feature)

    layer3_patches = feature_map_to_patches(layer3_feature)

    print("\n" + "=" * 60)
    print("Patch Features")
    print("=" * 60)

    print("\nLayer2 patches:")
    print(layer2_patches.shape)

    print("\nLayer3 patches:")
    print(layer3_patches.shape)

    # =========================================================
    # 6. Inspect one patch feature
    # =========================================================

    print("\n" + "=" * 60)
    print("Single Patch Inspection")
    print("=" * 60)

    # First image, first spatial patch
    first_layer2_patch = layer2_patches[0, 0]


# =========================================================
# Verify spatial mapping
# =========================================================

    row = 10
    col = 5

    _, _, _, width = layer2_feature.shape

    patch_index = row * width + col

    # Original feature vector at spatial location (row, col)
    feature_from_map = layer2_feature[0, :, row, col]

    # Corresponding flattened patch feature
    feature_from_patches = layer2_patches[0, patch_index, :]

    max_abs_error = torch.max(
        torch.abs(feature_from_map - feature_from_patches)
    )

    print("\n" + "=" * 60)
    print("Patch Spatial Mapping Verification")
    print("=" * 60)

    print(f"\nSpatial position: ({row}, {col})")
    print(f"Patch index: {patch_index}")

    print("\nFeature from feature map:")
    print(feature_from_map[:10])

    print("\nFeature from patch tensor:")
    print(feature_from_patches[:10])

    print("\nMaximum absolute error:")
    print(max_abs_error.item())

    print("\nFirst layer2 patch shape:")
    print(first_layer2_patch.shape)

    print("\nFirst 10 values of first layer2 patch:")
    print(first_layer2_patch[:10])

    # =========================================================
    # 7. Summary
    # =========================================================

    _, c2, h2, w2 = layer2_feature.shape
    _, c3, h3, w3 = layer3_feature.shape

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    print(
        f"\nLayer2:"
        f"\nFeature map = [{c2}, {h2}, {w2}]"
        f"\nPatch count = {h2} x {w2} = {h2 * w2}"
        f"\nPatch dimension = {c2}"
    )

    print(
        f"\nLayer3:"
        f"\nFeature map = [{c3}, {h3}, {w3}]"
        f"\nPatch count = {h3} x {w3} = {h3 * w3}"
        f"\nPatch dimension = {c3}"
    )

    print("\nExperiment completed.")


if __name__ == "__main__":
    main()