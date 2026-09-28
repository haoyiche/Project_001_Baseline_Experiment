import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import (
    ViT_B_16_Weights,
    vit_b_16,
)


# ============================================================
# Config
# ============================================================

IMAGE_PATH = Path(
    "data/raw/coco2017/val2017_subset500/000000097988.jpg"
)

OUTPUT_DIR = Path(
    "results/exp024"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "vit_cls_vs_mean_patch.csv"
)

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Device:",
        DEVICE,
    )

    print(
        "Image:",
        IMAGE_PATH,
    )


    # --------------------------------------------------------
    # 1. Load pretrained ViT
    # --------------------------------------------------------

    weights = (
        ViT_B_16_Weights.DEFAULT
    )

    transform = (
        weights.transforms()
    )

    model = vit_b_16(
        weights=weights
    )

    model.eval()
    model.to(
        DEVICE
    )


    # --------------------------------------------------------
    # 2. Load image
    # --------------------------------------------------------

    image = Image.open(
        IMAGE_PATH
    ).convert("RGB")

    x = transform(
        image
    ).unsqueeze(0)

    x = x.to(
        DEVICE
    )

    print()
    print(
        "Input shape:",
        x.shape,
    )


    # --------------------------------------------------------
    # 3. Image -> patch embeddings
    #
    # ViT-B/16:
    #
    # 224x224 image
    # patch size = 16
    #
    # 14x14 = 196 patches
    #
    # Output:
    # [B, 196, 768]
    # --------------------------------------------------------

    with torch.inference_mode():

        patch_tokens_before_cls = (
            model._process_input(
                x
            )
        )

    print(
        "Patch tokens before CLS:",
        patch_tokens_before_cls.shape,
    )


    # --------------------------------------------------------
    # 4. Add CLS token
    #
    # [B, 196, 768]
    # ->
    # [B, 197, 768]
    # --------------------------------------------------------

    batch_size = (
        patch_tokens_before_cls.shape[0]
    )

    cls_token = (
        model.class_token.expand(
            batch_size,
            -1,
            -1,
        )
    )

    tokens_with_cls = torch.cat(
        [
            cls_token,
            patch_tokens_before_cls,
        ],
        dim=1,
    )

    print(
        "Tokens with CLS:",
        tokens_with_cls.shape,
    )


    # --------------------------------------------------------
    # 5. Transformer encoder
    #
    # Encoder includes:
    # - positional embedding
    # - dropout
    # - Transformer blocks
    # - final LayerNorm
    #
    # Output:
    # [B, 197, 768]
    # --------------------------------------------------------

    with torch.inference_mode():

        encoded_tokens = (
            model.encoder(
                tokens_with_cls
            )
        )

    print(
        "Encoded tokens:",
        encoded_tokens.shape,
    )


    # --------------------------------------------------------
    # 6. Split global CLS and patch representations
    # --------------------------------------------------------

    cls_embedding = (
        encoded_tokens[:, 0, :]
    )

    patch_embeddings = (
        encoded_tokens[:, 1:, :]
    )

    print()
    print(
        "CLS embedding:",
        cls_embedding.shape,
    )

    print(
        "Patch embeddings:",
        patch_embeddings.shape,
    )


    # --------------------------------------------------------
    # 7. Mean-pool all encoded patch tokens
    #
    # [B, 196, 768]
    # mean over patch dimension
    # ->
    # [B, 768]
    # --------------------------------------------------------

    mean_patch_embedding = (
        patch_embeddings.mean(
            dim=1
        )
    )

    print(
        "Mean-patch embedding:",
        mean_patch_embedding.shape,
    )


    # --------------------------------------------------------
    # 8. Norm comparison
    # --------------------------------------------------------

    cls_norm = torch.norm(
        cls_embedding,
        p=2,
        dim=1,
    )

    mean_patch_norm = torch.norm(
        mean_patch_embedding,
        p=2,
        dim=1,
    )

    patch_norms = torch.norm(
        patch_embeddings,
        p=2,
        dim=2,
    )

    print()
    print(
        "=" * 70
    )

    print(
        "NORM COMPARISON"
    )

    print(
        "=" * 70
    )

    print(
        "CLS norm:",
        cls_norm.item(),
    )

    print(
        "Mean-patch norm:",
        mean_patch_norm.item(),
    )

    print(
        "Patch norm min:",
        patch_norms.min().item(),
    )

    print(
        "Patch norm mean:",
        patch_norms.mean().item(),
    )

    print(
        "Patch norm max:",
        patch_norms.max().item(),
    )


    # --------------------------------------------------------
    # 9. CLS vs mean-patch cosine similarity
    # --------------------------------------------------------

    cosine_similarity = (
        F.cosine_similarity(
            cls_embedding,
            mean_patch_embedding,
            dim=1,
        )
    )

    print()
    print(
        "=" * 70
    )

    print(
        "CLS VS MEAN-PATCH"
    )

    print(
        "=" * 70
    )

    print(
        "Cosine similarity:",
        cosine_similarity.item(),
    )


    # --------------------------------------------------------
    # 10. Raw L2 distance
    # --------------------------------------------------------

    raw_l2 = torch.norm(
        cls_embedding
        - mean_patch_embedding,
        p=2,
        dim=1,
    )

    print(
        "Raw L2 distance:",
        raw_l2.item(),
    )


    # --------------------------------------------------------
    # 11. Normalize both embeddings
    # --------------------------------------------------------

    cls_normalized = F.normalize(
        cls_embedding,
        p=2,
        dim=1,
    )

    mean_patch_normalized = (
        F.normalize(
            mean_patch_embedding,
            p=2,
            dim=1,
        )
    )


    # --------------------------------------------------------
    # 12. Normalized L2
    # --------------------------------------------------------

    normalized_l2 = torch.norm(
        cls_normalized
        - mean_patch_normalized,
        p=2,
        dim=1,
    )

    print(
        "Normalized L2 distance:",
        normalized_l2.item(),
    )


    # --------------------------------------------------------
    # 13. Verify geometry:
    #
    # ||x_n - y_n||^2
    # =
    # 2 - 2 cos(theta)
    # --------------------------------------------------------

    normalized_l2_squared = (
        normalized_l2 ** 2
    )

    theoretical_l2_squared = (
        2.0
        - 2.0
        * cosine_similarity
    )

    geometry_error = torch.abs(
        normalized_l2_squared
        - theoretical_l2_squared
    )

    print()
    print(
        "=" * 70
    )

    print(
        "GEOMETRY CHECK"
    )

    print(
        "=" * 70
    )

    print(
        "Normalized L2^2:",
        normalized_l2_squared.item(),
    )

    print(
        "2 - 2*cosine:",
        theoretical_l2_squared.item(),
    )

    print(
        "Absolute error:",
        geometry_error.item(),
    )


    # --------------------------------------------------------
    # 14. Sanity checks
    # --------------------------------------------------------

    assert (
        x.shape
        == (1, 3, 224, 224)
    )

    assert (
        patch_tokens_before_cls.shape
        == (1, 196, 768)
    )

    assert (
        tokens_with_cls.shape
        == (1, 197, 768)
    )

    assert (
        encoded_tokens.shape
        == (1, 197, 768)
    )

    assert (
        cls_embedding.shape
        == (1, 768)
    )

    assert (
        patch_embeddings.shape
        == (1, 196, 768)
    )

    assert (
        mean_patch_embedding.shape
        == (1, 768)
    )

    assert (
        geometry_error.item()
        < 1e-5
    )

    print()
    print(
        "All sanity checks PASS"
    )


    # --------------------------------------------------------
    # 15. Save experiment result
    # --------------------------------------------------------

    result = {
        "image_id":
            IMAGE_PATH.stem,

        "cls_norm":
            cls_norm.item(),

        "mean_patch_norm":
            mean_patch_norm.item(),

        "patch_norm_min":
            patch_norms.min().item(),

        "patch_norm_mean":
            patch_norms.mean().item(),

        "patch_norm_max":
            patch_norms.max().item(),

        "cls_mean_cosine":
            cosine_similarity.item(),

        "cls_mean_raw_l2":
            raw_l2.item(),

        "cls_mean_normalized_l2":
            normalized_l2.item(),

        "geometry_error":
            geometry_error.item(),
    }


    with open(
        OUTPUT_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=result.keys(),
        )

        writer.writeheader()
        writer.writerow(
            result
        )

    print()
    print(
        "Saved:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()