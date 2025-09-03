'''torchvision models.'''

from collections.abc import Sequence

import torch
from torchvision import transforms, models
from PIL import Image


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
class TVResNet18:
    '''
    ResNet-18 torchvision model wrapper.

    Summary
    -------
    This class implements a standardized interface to load
    and use a pretrained ResNet-18 model from torchvision.

    '''

    def __init__(self) -> None:

        self.weights = models.ResNet18_Weights.DEFAULT

        self.model = models.resnet18(weights=self.weights)
        self.model = self.model.eval()

        self.preprocessor = self.weights.transforms()
        # self.preprocessor = transform['resize_and_crop']

    @property
    def class_names(self) -> list[str]:
        return self.weights.meta['categories']

    def __call__(self, images: Image.Image | Sequence[Image.Image]) -> list[dict[str, str | float]]:
        '''Predict.'''

        # preprocess images
        if not isinstance(images, Sequence):
            images = [images]

        x_list = []

        for img in images:
            tensor = self.preprocessor(img)  # (3, h, w)
            x_list.append(tensor)

        x = torch.stack(x_list, dim=0)  # (b, 3, h, w)

        # run model
        with torch.no_grad():
            logits = self.model(x)  # (b, 1000)

        # postprocess prediction
        probs = logits.softmax(dim=1)  # (b, 1000)

        max_probs, max_ids = probs.max(dim=1)  # (b,)

        scores = max_probs.tolist()
        labels = [self.class_names[class_idx.item()] for class_idx in max_ids]

        # format output (analogous to HF pipeline)
        out = [{'label': label, 'score': score} for label, score in zip(labels, scores)]

        return out
