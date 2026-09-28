import ast
import csv
from pathlib import Path

from PIL import Image, ImageDraw


IMAGE_DIR = Path(
    "data/raw/coco2017/val2017_subset500"
)

CSV_PATH = Path(
    "results/exp024/retrieval_minimal.csv"
)

OUT_DIR = Path(
    "results/exp024/retrieval_visualizations"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


THUMB_SIZE = (240, 180)
LABEL_HEIGHT = 30


def image_path(image_id):
    return (
        IMAGE_DIR
        / f"{image_id:012d}.jpg"
    )


def load_thumb(image_id):

    image = Image.open(
        image_path(image_id)
    ).convert("RGB")

    image.thumbnail(
        THUMB_SIZE
    )

    canvas = Image.new(
        "RGB",
        (
            THUMB_SIZE[0],
            THUMB_SIZE[1]
            + LABEL_HEIGHT,
        ),
    )

    x = (
        THUMB_SIZE[0]
        - image.width
    ) // 2

    y = (
        THUMB_SIZE[1]
        - image.height
    ) // 2

    canvas.paste(
        image,
        (x, y),
    )

    draw = ImageDraw.Draw(
        canvas
    )

    draw.text(
        (5, THUMB_SIZE[1] + 5),
        str(image_id),
    )

    return canvas


def make_row(
    title,
    ids,
):

    width = (
        THUMB_SIZE[0]
        * len(ids)
    )

    height = (
        THUMB_SIZE[1]
        + LABEL_HEIGHT
        + 35
    )

    canvas = Image.new(
        "RGB",
        (width, height),
    )

    draw = ImageDraw.Draw(
        canvas
    )

    draw.text(
        (5, 5),
        title,
    )

    for i, image_id in enumerate(
        ids
    ):

        thumb = load_thumb(
            image_id
        )

        canvas.paste(
            thumb,
            (
                i * THUMB_SIZE[0],
                30,
            ),
        )

    return canvas


with open(
    CSV_PATH,
    "r",
    encoding="utf-8",
) as f:

    rows = list(
        csv.DictReader(f)
    )


for row in rows:

    query_id = int(
        row["query_id"]
    )

    raw_ids = ast.literal_eval(
        row["raw_l2_top5"]
    )

    cosine_ids = ast.literal_eval(
        row["cosine_top5"]
    )


    query_row = make_row(
        "QUERY",
        [query_id],
    )

    raw_row = make_row(
        "RAW L2 TOP-5",
        raw_ids,
    )

    cosine_row = make_row(
        "COSINE TOP-5",
        cosine_ids,
    )


    width = max(
        query_row.width,
        raw_row.width,
        cosine_row.width,
    )

    height = (
        query_row.height
        + raw_row.height
        + cosine_row.height
    )


    final = Image.new(
        "RGB",
        (width, height),
    )


    y = 0

    for block in [
        query_row,
        raw_row,
        cosine_row,
    ]:

        final.paste(
            block,
            (0, y),
        )

        y += block.height


    output_path = (
        OUT_DIR
        / f"query_{query_id}.jpg"
    )


    final.save(
        output_path
    )


    print(
        "Saved:",
        output_path,
    )