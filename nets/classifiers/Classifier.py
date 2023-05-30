class Classifier:
    def __init__(self, type):
        if (type == "Linear"):
            self.classifier = "Linear"
        elif (type == "something else"):
            self.classifier = "somethingelse"

    def train(self, trainLoader,featureExtractor):
        for e in range(1000):
            for data in trainLoader:
                img,target =data
                z= featureExtractor.getFeature(img)
                self.classifier(z)

        pass

    @torch.no_grad
    def predict(self, z):
        return self.classifier(z)