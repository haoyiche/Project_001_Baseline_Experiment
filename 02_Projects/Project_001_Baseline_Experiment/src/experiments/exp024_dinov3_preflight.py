from pathlib import Path

import torch
import transformers
from transformers import (
    AutoImageProcessor,
    AutoModel,
)


MODEL_DIR = Path(
    r"D:\AI_Lab\models\dinov3"
    r"\dinov3-vits16-pretrain-lvd1689m"
)


def main():

    print("Torch:", torch.__version__)
    print(
        "Transformers:",
        transformers.__version__,
    )

    print(
        "CUDA available:",
        torch.cuda.is_available(),
    )

    print(
        "Model dir:",
        MODEL_DIR,
    )

    print(
        "Model dir exists:",
        MODEL_DIR.exists(),
    )

    assert MODEL_DIR.exists()

    # --------------------------------
    # Local processor
    # --------------------------------

    processor = (
        AutoImageProcessor.from_pretrained(
            MODEL_DIR,
            local_files_only=True,
        )
    )

    print()
    print(
        "Processor:",
        type(processor).__name__,
    )

    # --------------------------------
    # Local model
    # --------------------------------

    model = AutoModel.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
    )

    print(
        "Model:",
        type(model).__name__,
    )

    # --------------------------------
    # Architecture config
    # --------------------------------

    print()
    print(
        "Patch size:",
        model.config.patch_size,
    )

    print(
        "Hidden size:",
        model.config.hidden_size,
    )

    print(
        "Register tokens:",
        model.config.num_register_tokens,
    )

    print(
        "Image size:",
        model.config.image_size,
    )

    # --------------------------------
    # Expected DINOv3 ViT-S/16
    # --------------------------------

    assert model.config.patch_size == 16
    assert model.config.hidden_size == 384
    assert (
        model.config.num_register_tokens
        == 4
    )

    print()
    print(
        "DINOv3 local model preflight PASS"
    )


if __name__ == "__main__":
    main()