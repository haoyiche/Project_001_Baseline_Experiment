from pathlib import Path

import torch
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

IMAGE_PATH = Path(
    "data/raw/coco2017/val2017_subset500"
    "/000000097988.jpg"
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

    print(
        "Device:",
        DEVICE,
    )

    print(
        "Model dir:",
        MODEL_DIR,
    )

    print(
        "Image:",
        IMAGE_PATH,
    )


    # --------------------------------------------------------
    # 1. Load processor + model
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


    # --------------------------------------------------------
    # 2. Config
    # --------------------------------------------------------

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
        "=" * 70
    )

    print(
        "MODEL CONFIG"
    )

    print(
        "=" * 70
    )

    print(
        "Patch size:",
        patch_size,
    )

    print(
        "Hidden size:",
        hidden_size,
    )

    print(
        "Register tokens:",
        num_register_tokens,
    )


    # --------------------------------------------------------
    # 3. Load image
    # --------------------------------------------------------

    image = Image.open(
        IMAGE_PATH
    ).convert("RGB")

    print()
    print(
        "Original image size:",
        image.size,
    )


    # --------------------------------------------------------
    # 4. Preprocess
    # --------------------------------------------------------

    inputs = processor(
        images=image,
        return_tensors="pt",
    )

    pixel_values = (
        inputs["pixel_values"]
    )

    print(
        "Preprocessed pixel_values:",
        pixel_values.shape,
    )


    # --------------------------------------------------------
    # 5. Calculate EXPECTED patch grid
    # --------------------------------------------------------

    batch_size = (
        pixel_values.shape[0]
    )

    image_height = (
        pixel_values.shape[2]
    )

    image_width = (
        pixel_values.shape[3]
    )

    patch_height = (
        image_height
        // patch_size
    )

    patch_width = (
        image_width
        // patch_size
    )

    num_patches = (
        patch_height
        * patch_width
    )

    expected_sequence_length = (
        1
        + num_register_tokens
        + num_patches
    )

    print()
    print(
        "=" * 70
    )

    print(
        "EXPECTED TOKEN STRUCTURE"
    )

    print(
        "=" * 70
    )

    print(
        "Patch grid:",
        f"{patch_height} x {patch_width}",
    )

    print(
        "Num patches:",
        num_patches,
    )

    print(
        "Expected sequence length:",
        expected_sequence_length,
    )

    print(
        "Expected layout:",
        (
            f"1 CLS + "
            f"{num_register_tokens} register + "
            f"{num_patches} patch"
        ),
    )


    # --------------------------------------------------------
    # 6. Move inputs to GPU
    # --------------------------------------------------------

    inputs = {
        key: value.to(
            DEVICE
        )
        for key, value
        in inputs.items()
    }


    # --------------------------------------------------------
    # 7. Forward
    # --------------------------------------------------------

    with torch.inference_mode():

        outputs = model(
            **inputs
        )

    last_hidden_state = (
        outputs.last_hidden_state
    )

    print()
    print(
        "=" * 70
    )

    print(
        "MODEL OUTPUT"
    )

    print(
        "=" * 70
    )

    print(
        "Last hidden state:",
        last_hidden_state.shape,
    )


    # --------------------------------------------------------
    # 8. Split token types
    #
    # Sequence:
    #
    # [CLS]
    # [REGISTER x 4]
    # [PATCH x N]
    #
    # --------------------------------------------------------

    cls_token = (
        last_hidden_state[
            :,
            0,
            :,
        ]
    )

    register_tokens = (
        last_hidden_state[
            :,
            1:
            1 + num_register_tokens,
            :,
        ]
    )

    patch_tokens = (
        last_hidden_state[
            :,
            1 + num_register_tokens:,
            :,
        ]
    )

    print()
    print(
        "CLS token:",
        cls_token.shape,
    )

    print(
        "Register tokens:",
        register_tokens.shape,
    )

    print(
        "Patch tokens:",
        patch_tokens.shape,
    )


    # --------------------------------------------------------
    # 9. Restore spatial grid
    #
    # [B, N, C]
    # ->
    # [B, H_patch, W_patch, C]
    # --------------------------------------------------------

    patch_grid = (
        patch_tokens.reshape(
            batch_size,
            patch_height,
            patch_width,
            hidden_size,
        )
    )

    print(
        "Patch grid:",
        patch_grid.shape,
    )


    # --------------------------------------------------------
    # 10. Norm analysis
    # --------------------------------------------------------

    cls_norm = torch.norm(
        cls_token,
        p=2,
        dim=-1,
    )

    register_norms = torch.norm(
        register_tokens,
        p=2,
        dim=-1,
    )

    patch_norms = torch.norm(
        patch_tokens,
        p=2,
        dim=-1,
    )

    print()
    print(
        "=" * 70
    )

    print(
        "TOKEN NORMS"
    )

    print(
        "=" * 70
    )

    print(
        "CLS norm:",
        cls_norm.item(),
    )

    print(
        "Register norm min:",
        register_norms.min().item(),
    )

    print(
        "Register norm mean:",
        register_norms.mean().item(),
    )

    print(
        "Register norm max:",
        register_norms.max().item(),
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
    # 11. Mean-patch representation
    # --------------------------------------------------------

    mean_patch = (
        patch_tokens.mean(
            dim=1
        )
    )

    cls_mean_cosine = (
        torch.nn.functional.cosine_similarity(
            cls_token,
            mean_patch,
            dim=-1,
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
        "Mean-patch shape:",
        mean_patch.shape,
    )

    print(
        "CLS vs mean-patch cosine:",
        cls_mean_cosine.item(),
    )


    # --------------------------------------------------------
    # 12. Sanity checks
    # --------------------------------------------------------

    assert (
        last_hidden_state.shape
        == (
            batch_size,
            expected_sequence_length,
            hidden_size,
        )
    )

    assert (
        cls_token.shape
        == (
            batch_size,
            hidden_size,
        )
    )

    assert (
        register_tokens.shape
        == (
            batch_size,
            num_register_tokens,
            hidden_size,
        )
    )

    assert (
        patch_tokens.shape
        == (
            batch_size,
            num_patches,
            hidden_size,
        )
    )

    assert (
        patch_grid.shape
        == (
            batch_size,
            patch_height,
            patch_width,
            hidden_size,
        )
    )

    print()
    print(
        "All DINOv3 token checks PASS"
    )


if __name__ == "__main__":
    main()