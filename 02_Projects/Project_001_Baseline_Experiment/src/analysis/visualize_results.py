import os

import matplotlib.pyplot as plt

from PIL import Image



def visualize_samples(
        samples,
        save_path,
        title
):


    num = len(samples)


    plt.figure(
        figsize=(15,3)
    )


    for i, item in enumerate(samples):

        score, label, path = item


        image = Image.open(
            path
        )


        plt.subplot(
            1,
            num,
            i+1
        )


        plt.imshow(
            image
        )


        plt.axis(
            "off"
        )


        plt.title(
            f"score={score:.2f}\nlabel={label}"
        )


    plt.suptitle(
        title
    )


    plt.tight_layout()


    os.makedirs(
        os.path.dirname(save_path),
        exist_ok=True
    )


    plt.savefig(
        save_path,
        dpi=200
    )


    plt.close()