import torch


class AnomalyBaseline:


    def __init__(
        self,
        extractor
    ):

        self.extractor = extractor

        self.center = None



    def fit(
        self,
        dataloader
    ):

        features = []


        self.extractor.eval()


        with torch.no_grad():

            for images, labels, paths in dataloader:


                feat = self.extractor(
                    images
                )


                features.append(
                    feat
                )


        features = torch.cat(
            features,
            dim=0
        )


        self.center = features.mean(
            dim=0
        )


        return self.center