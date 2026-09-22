import json
import random
import time
import urllib.request

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


SEED = 42
NUM_IMAGES = 500
NUM_WORKERS = 16
MAX_RETRIES = 3


ROOT = Path("data/raw/coco2017")

ANN_PATH = (
    ROOT
    / "annotations"
    / "instances_val2017.json"
)

IMAGE_DIR = (
    ROOT
    / "val2017_subset500"
)

MANIFEST_PATH = Path(
    "configs/exp023_coco_subset500_ids.json"
)


IMAGE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ==============================
# Load COCO metadata
# ==============================

with open(
    ANN_PATH,
    "r",
    encoding="utf-8",
) as f:

    coco = json.load(f)


images = coco["images"]

print(
    "COCO val images:",
    len(images),
)


# ==============================
# Deterministic subset
# ==============================

rng = random.Random(SEED)

selected = rng.sample(
    images,
    NUM_IMAGES,
)

selected.sort(
    key=lambda x: x["id"]
)


subset_ids = [
    img["id"]
    for img in selected
]


with open(
    MANIFEST_PATH,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        {
            "dataset": "COCO 2017 val",
            "seed": SEED,
            "num_images": NUM_IMAGES,
            "image_ids": subset_ids,
        },
        f,
        indent=2,
    )


print(
    "Subset manifest:",
    MANIFEST_PATH,
)


# ==============================
# Download
# ==============================

def download_one(img):

    file_name = img["file_name"]

    output_path = (
        IMAGE_DIR
        / file_name
    )

    if (
        output_path.exists()
        and output_path.stat().st_size > 0
    ):

        return (
            file_name,
            "exists",
        )


    url = img.get(
        "coco_url",
        (
            "http://images.cocodataset.org/"
            f"val2017/{file_name}"
        ),
    )


    last_error = None


    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):

        try:

            urllib.request.urlretrieve(
                url,
                output_path,
            )

            return (
                file_name,
                "downloaded",
            )


        except Exception as exc:

            last_error = exc

            if output_path.exists():
                output_path.unlink()

            time.sleep(attempt)


    return (
        file_name,
        f"FAILED: {last_error}",
    )


# ==============================
# Parallel download
# ==============================

success = 0
failed = []


with ThreadPoolExecutor(
    max_workers=NUM_WORKERS
) as executor:

    futures = {
        executor.submit(
            download_one,
            img,
        ): img
        for img in selected
    }


    for index, future in enumerate(
        as_completed(futures),
        start=1,
    ):

        file_name, status = (
            future.result()
        )


        if status in (
            "downloaded",
            "exists",
        ):

            success += 1

        else:

            failed.append(
                (
                    file_name,
                    status,
                )
            )


        print(
            f"[{index:03d}/{NUM_IMAGES}] "
            f"{status}: "
            f"{file_name}"
        )


print()
print("==========================")
print("EXP023 COCO subset")
print("==========================")

print(
    "Expected:",
    NUM_IMAGES,
)

print(
    "Success :",
    success,
)

print(
    "Failed  :",
    len(failed),
)


if failed:

    print()
    print("Failed files:")

    for item in failed:
        print(item)

else:

    print(
        "COCO subset preparation: PASS"
    )