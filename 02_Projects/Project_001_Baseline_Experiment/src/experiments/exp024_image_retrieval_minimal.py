import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import (
    ResNet18_Weights,
    resnet18,
)


MANIFEST_PATH = Path(
    "configs/exp023_coco_subset500_ids.json"
)

IMAGE_DIR = Path(
    "data/raw/coco2017/val2017_subset500"
)

OUTPUT_DIR = Path(
    "results/exp024"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

NUM_GALLERY = 80
NUM_QUERY = 5


def image_path_from_id(image_id):
    return (
        IMAGE_DIR
        / f"{image_id:012d}.jpg"
    )


def extract_embedding(
    image_path,
    model,
    transform,
):
    image = Image.open(
        image_path
    ).convert("RGB")

    x = transform(
        image
    ).unsqueeze(0).to(
        DEVICE
    )

    with torch.no_grad():
        embedding = model(
            x
        )

    return embedding.squeeze(0)


def topk_indices(
    values,
    k,
    largest,
):
    return torch.topk(
        values,
        k=k,
        largest=largest,
    ).indices


def main():

    with open(
        MANIFEST_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        manifest = json.load(f)

    image_ids = manifest[
        "image_ids"
    ]

    gallery_ids = image_ids[
        :NUM_GALLERY
    ]

    query_ids = image_ids[
        NUM_GALLERY:
        NUM_GALLERY + NUM_QUERY
    ]


    weights = (
        ResNet18_Weights.DEFAULT
    )

    transform = (
        weights.transforms()
    )

    model = resnet18(
        weights=weights
    )

    model.fc = (
        torch.nn.Identity()
    )

    model.eval()
    model.to(
        DEVICE
    )


    print(
        "Device:",
        DEVICE,
    )

    print(
        "Extracting gallery embeddings..."
    )


    gallery_embeddings = []

    for image_id in gallery_ids:

        embedding = (
            extract_embedding(
                image_path_from_id(
                    image_id
                ),
                model,
                transform,
            )
        )

        gallery_embeddings.append(
            embedding
        )


    gallery_embeddings = torch.stack(
        gallery_embeddings,
        dim=0,
    )


    print(
        "Gallery shape:",
        gallery_embeddings.shape,
    )


    gallery_norms = (
        torch.norm(
            gallery_embeddings,
            dim=1,
        )
    )


    print()
    print(
        "Embedding norm:"
    )

    print(
        "min :",
        gallery_norms.min().item(),
    )

    print(
        "mean:",
        gallery_norms.mean().item(),
    )

    print(
        "max :",
        gallery_norms.max().item(),
    )


    gallery_normalized = (
        F.normalize(
            gallery_embeddings,
            dim=1,
        )
    )


    rows = []


    for query_id in query_ids:

        query = extract_embedding(
            image_path_from_id(
                query_id
            ),
            model,
            transform,
        )


        query_norm = torch.norm(
            query
        ).item()


        raw_l2 = torch.norm(
            gallery_embeddings
            - query.unsqueeze(0),
            dim=1,
        )


        cosine = (
            F.cosine_similarity(
                gallery_embeddings,
                query.unsqueeze(0),
                dim=1,
            )
        )


        query_normalized = (
            F.normalize(
                query,
                dim=0,
            )
        )


        normalized_l2 = torch.norm(
            gallery_normalized
            - query_normalized.unsqueeze(0),
            dim=1,
        )


        raw_top5 = topk_indices(
            raw_l2,
            k=5,
            largest=False,
        )


        cosine_top5 = topk_indices(
            cosine,
            k=5,
            largest=True,
        )


        normalized_l2_top5 = (
            topk_indices(
                normalized_l2,
                k=5,
                largest=False,
            )
        )


        raw_ids = [
            gallery_ids[i]
            for i in raw_top5.tolist()
        ]

        cosine_ids = [
            gallery_ids[i]
            for i in cosine_top5.tolist()
        ]

        normalized_ids = [
            gallery_ids[i]
            for i
            in normalized_l2_top5.tolist()
        ]


        print()
        print(
            "=" * 70
        )

        print(
            "Query:",
            query_id,
        )

        print(
            "Query norm:",
            query_norm,
        )

        print(
            "Raw L2 top5:",
            raw_ids,
        )

        print(
            "Cosine top5:",
            cosine_ids,
        )

        print(
            "Normalized L2 top5:",
            normalized_ids,
        )

        print(
            "Cosine == normalized L2:",
            cosine_ids
            == normalized_ids,
        )


        rows.append(
            {
                "query_id": query_id,
                "query_norm": query_norm,
                "raw_l2_top5": str(
                    raw_ids
                ),
                "cosine_top5": str(
                    cosine_ids
                ),
                "normalized_l2_top5": str(
                    normalized_ids
                ),
                "cosine_equals_normalized_l2": (
                    cosine_ids
                    == normalized_ids
                ),
            }
        )


    output_path = (
        OUTPUT_DIR
        / "retrieval_minimal.csv"
    )


    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=rows[
                0
            ].keys(),
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


    print()
    print(
        "Saved:",
        output_path,
    )


if __name__ == "__main__":
    main()