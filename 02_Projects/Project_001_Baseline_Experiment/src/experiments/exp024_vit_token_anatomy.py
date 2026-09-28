from pathlib import Path

import torch
from PIL import Image
from torchvision.models import (
    ViT_B_16_Weights,
    vit_b_16,
)


IMAGE_PATH = Path(
    "data/raw/coco2017/val2017_subset500/000000097988.jpg"
)

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


def main():

    print("Device:", DEVICE)

    # -----------------------------
    # Model + preprocessing
    # -----------------------------

    weights = ViT_B_16_Weights.DEFAULT
    transform = weights.transforms()

    model = vit_b_16(
        weights=weights
    )

    model.eval()
    model.to(DEVICE)

    # -----------------------------
    # Image
    # -----------------------------

    image = Image.open(
        IMAGE_PATH
    ).convert("RGB")

    x = transform(
        image
    ).unsqueeze(0).to(
        DEVICE
    )

    print(
        "Input shape:",
        x.shape
    )

    # -----------------------------
    # Patch embedding
    # -----------------------------

    with torch.no_grad():

        patch_tokens = (
            model._process_input(x)
        )

    print(
        "Patch tokens before CLS:",
        patch_tokens.shape
    )

    # -----------------------------
    # Add CLS token
    # -----------------------------

    batch_size = (
        patch_tokens.shape[0]
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
            patch_tokens,
        ],
        dim=1,
    )

    print(
        "Tokens after adding CLS:",
        tokens_with_cls.shape
    )

    # -----------------------------
    # Transformer encoder
    # -----------------------------

    with torch.no_grad():

        encoded_tokens = (
            model.encoder(
                tokens_with_cls
            )

        )

    print(
        "Encoded tokens:",
        encoded_tokens.shape
    )

    # -----------------------------
    # Split CLS / patch tokens
    # -----------------------------

    cls_embedding = (
        encoded_tokens[:, 0]
    )

    patch_embeddings = (
        encoded_tokens[:, 1:]
    )

    print(
        "CLS embedding:",
        cls_embedding.shape
    )

    print(
        "Patch embeddings:",
        patch_embeddings.shape
    )

    # -----------------------------
    # Basic statistics
    # -----------------------------

    cls_norm = torch.norm(
        cls_embedding,
        dim=1,
    )

    patch_norms = torch.norm(
        patch_embeddings,
        dim=2,
    )

    print()
    print(
        "CLS norm:",
        cls_norm.item()
    )

    print(
        "Patch norm min:",
        patch_norms.min().item()
    )

    print(
        "Patch norm mean:",
        patch_norms.mean().item()
    )

    print(
        "Patch norm max:",
        patch_norms.max().item()
    )

    # -----------------------------
    # Sanity checks
    # -----------------------------

    assert (
        patch_tokens.shape[1]
        == 196
    )

    assert (
        tokens_with_cls.shape[1]
        == 197
    )

    assert (
        cls_embedding.shape[-1]
        == 768
    )

    assert (
        patch_embeddings.shape
        == (1, 196, 768)
    )

    print()
    print(
        "All shape checks PASS"
    )


if __name__ == "__main__":
    main()