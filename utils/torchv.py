'''torchvision models.'''

import torch
from torchvision import transforms, models


RESIZE_SHAPE = (256, 256)
SHAPE = (224, 224)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


transform = {
    # resize and crop
    'resize_and_crop': transforms.Compose([
        transforms.Resize(RESIZE_SHAPE),
        transforms.CenterCrop(SHAPE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ]),

    # resize
    'resize': transforms.Compose([
        transforms.Resize(SHAPE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ]),

    # cropped images
    'crop': transforms.Compose([
        transforms.CenterCrop(SHAPE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ]),

    # full-sized
    'full': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])
}


# TODO: enable GPU processing
class TVResNet18():
    '''ResNet-18 torchvision model wrapper.'''

    def __init__(self):

        self.weights = models.ResNet18_Weights.DEFAULT

        self.model = models.resnet18(weights=self.weights)
        self.model = self.model.eval()

        self.preprocessor = self.weights.transforms()
        # self.preprocessor = transform['resize_and_crop']

    @property
    def class_names(self):
        return self.weights.meta['categories']

    def __call__(self, images):
        '''Predict.'''

        # preprocess images
        if not isinstance(images, (list, tuple)):
            images = [images]

        x = []
        for img in images:
            tensor = self.preprocessor(img) # (3, h, w)
            x.append(tensor)

        x = torch.stack(x, dim=0) # (b, 3, h, w)

        # run model
        with torch.no_grad():
            logits = self.model(x) # (b, 1000)

        # postprocess prediction
        probs = logits.softmax(dim=1) # (b, 1000)

        max_probs, max_ids = probs.max(dim=1) # (b,)

        scores = max_probs.tolist()
        labels = [self.class_names[class_idx.item()] for class_idx in max_ids]

        # format output (analogous to HF pipeline)
        out = [{'label': label, 'score': score} for label, score in zip(labels, scores)]

        return out

