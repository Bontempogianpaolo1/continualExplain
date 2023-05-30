import torch

import VAE


class FeatureExtractor:
    def __init__(self, type):
        if (type == "VAE"):
            self.featureExtractor = VAE()
        elif (type == "something else"):
            self.featureExtractor = "somethingelse"

    def train(self, trainLoader):
        pass

    @torch.no_grad
    def getFeature(self, imgs):
        return self.featureExtractor(imgs)
