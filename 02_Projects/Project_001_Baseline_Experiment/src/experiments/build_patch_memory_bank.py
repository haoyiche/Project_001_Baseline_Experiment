from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights


# ============================================================
# Dataset
# ============================================================

class NormalBottleDataset(Dataset):
    """
    Load only MVTec bottle/train/good images.
    """

    def __init__(self, root_dir: Path, transform):
        self.root_dir = Path(root_dir)
        self.transform = transform

        valid_extensions = {
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp",
            ".tif",
            ".tiff",
        }

        self.image_paths = sorted(
            [
                p
                for p in self.root_dir.iterdir()
                if p.is_file() and p.suffix.lower() in valid_extensions
            ]
        )

        if len(self.image_paths) == 0:
            raise RuntimeError(
                f"No images found in:\n{self.root_dir.resolve()}"
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        return image, str(image_path)


# ============================================================
# Feature extraction
# ============================================================

def extract_layer2_feature(
    model,
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Input:
        [B, 3, 224, 224]

    Output:
        layer2 feature map
        [B, 128, 28, 28]
    """

    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)
    x = model.layer2(x)

    return x


def feature_map_to_patches(
    feature_map: torch.Tensor,
) -> torch.Tensor:
    """
    [B, C, H, W]
        ↓
    [B, H*W, C]
    """

    if feature_map.ndim != 4:
        raise ValueError(
            f"Expected [B,C,H,W], got {feature_map.shape}"
        )

    batch_size, channels, height, width = feature_map.shape

    patches = feature_map.permute(0, 2, 3, 1)

    patches = patches.reshape(
        batch_size,
        height * width,
        channels,
    )

    return patches


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("EXP009 - Build Normal Patch Memory Bank")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"\nDevice: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # --------------------------------------------------------
    # 2. Dataset path
    # --------------------------------------------------------

    train_good_dir = Path(
        "data/raw/mvtec_anomaly_detection/"
        "bottle/train/good"
    )

    if not train_good_dir.exists():
        raise FileNotFoundError(
            f"\nDataset path does not exist:\n"
            f"{train_good_dir.resolve()}"
        )

    # --------------------------------------------------------
    # 3. Model + preprocessing
    # --------------------------------------------------------

    weights = ResNet18_Weights.DEFAULT

    preprocess = weights.transforms()

    model = resnet18(weights=weights)

    model.eval()
    model.to(device)

    # --------------------------------------------------------
    # 4. Dataset + DataLoader
    # --------------------------------------------------------

    dataset = NormalBottleDataset(
        root_dir=train_good_dir,
        transform=preprocess,
    )

    batch_size = 16

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    print(f"\nNormal training images: {len(dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Number of batches: {len(dataloader)}")

    # --------------------------------------------------------
    # 5. Prediction
    # --------------------------------------------------------

    expected_patches_per_image = 28 * 28
    expected_total_patches = (
        len(dataset) * expected_patches_per_image
    )

    print("\n" + "=" * 70)
    print("Prediction")
    print("=" * 70)

    print(
        f"\nExpected patches per image: "
        f"{expected_patches_per_image}"
    )

    print(
        f"Expected total patches: "
        f"{len(dataset)} × {expected_patches_per_image} "
        f"= {expected_total_patches}"
    )

    print(
        f"Expected memory bank shape: "
        f"[{expected_total_patches}, 128]"
    )

    # --------------------------------------------------------
    # 6. Build memory bank
    # --------------------------------------------------------

    memory_bank_parts = []

    total_images_processed = 0

    with torch.no_grad():

        for batch_index, (images, paths) in enumerate(dataloader):

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

            batch_size_actual = patches.shape[0]
            num_patches = patches.shape[1]
            feature_dim = patches.shape[2]

            # ------------------------------------------------
            # Flatten batch dimension:
            #
            # [B, 784, 128]
            #       ↓
            # [B*784, 128]
            # ------------------------------------------------

            patches = patches.reshape(
                batch_size_actual * num_patches,
                feature_dim,
            )

            # Move to CPU before storing.
            #
            # Otherwise all patches remain on GPU and memory
            # usage keeps increasing.
            patches = patches.cpu()

            memory_bank_parts.append(patches)

            total_images_processed += batch_size_actual

            print(
                f"Batch {batch_index + 1:02d}/{len(dataloader):02d} | "
                f"images processed: {total_images_processed:3d}/{len(dataset)} | "
                f"feature map: {tuple(feature_map.shape)} | "
                f"patches: {tuple(patches.shape)}"
            )

    # --------------------------------------------------------
    # 7. Concatenate all normal patches
    # --------------------------------------------------------

    memory_bank = torch.cat(
        memory_bank_parts,
        dim=0,
    )

    # --------------------------------------------------------
    # 8. Results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("Memory Bank Result")
    print("=" * 70)

    print("\nMemory bank shape:")
    print(memory_bank.shape)

    print("\nMemory bank dtype:")
    print(memory_bank.dtype)

    print("\nMemory bank device:")
    print(memory_bank.device)

    # --------------------------------------------------------
    # 9. Memory usage
    # --------------------------------------------------------

    num_elements = memory_bank.numel()
    bytes_per_element = memory_bank.element_size()

    total_bytes = (
        num_elements * bytes_per_element
    )

    total_mb = (
        total_bytes / 1024 / 1024
    )

    print("\nMemory bank size:")
    print(f"{total_mb:.2f} MiB")

    # --------------------------------------------------------
    # 10. Verification
    # --------------------------------------------------------

    expected_shape = (
        expected_total_patches,
        128,
    )

    print("\n" + "=" * 70)
    print("Verification")
    print("=" * 70)

    print(f"\nExpected shape: {expected_shape}")
    print(f"Actual shape:   {tuple(memory_bank.shape)}")

    shape_correct = (
        tuple(memory_bank.shape)
        == expected_shape
    )

    print(f"\nShape correct: {shape_correct}")

    if not shape_correct:
        raise RuntimeError(
            "Memory bank shape does not match prediction."
        )

    # --------------------------------------------------------
    # 11. Inspect first normal patch
    # --------------------------------------------------------

    print("\nFirst memory-bank patch:")
    print(memory_bank[0, :10])

    # --------------------------------------------------------
    # 12. Optional save
    #
    # 暂时不保存。
    # 本实验重点是理解 memory bank 如何构建。
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("Experiment completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()