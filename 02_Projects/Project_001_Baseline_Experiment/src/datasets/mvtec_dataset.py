import os
from PIL import Image
from torch.utils.data import Dataset


class MVTecDataset(Dataset):

    def __init__(
        self,
        root,
        category,
        split="train",
        transform=None
    ):

        self.images = []
        self.labels = []

        self.transform = transform


        folder = os.path.join(
            root,
            category,
            split
        )


        if split == "train":

            folder = os.path.join(
                folder,
                "good"
            )


        for dirpath, _, files in os.walk(folder):

            for file in files:

                if file.endswith(".png"):

                    path = os.path.join(
                        dirpath,
                        file
                    )

                    self.images.append(
                        path
                    )


                    # good = 0
                    # defect = 1

                    if "good" in path:

                        self.labels.append(0)

                    else:

                        self.labels.append(1)



    def __len__(self):

        return len(self.images)



    def __getitem__(self,index):

        path = self.images[index]


        image = Image.open(
            path
        ).convert("RGB")


        if self.transform:

            image = self.transform(
                image
            )


        label = self.labels[index]


        return image, label, path