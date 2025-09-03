'''Hugging Face models.'''

from collections.abc import Sequence

from transformers import pipeline
from PIL import Image


# TODO: enable GPU processing
class HFResNet18:
    '''
    ResNet-18 Hugging Face pipeline wrapper.

    Summary
    -------
    This is a simple wrapper for a pretrained ResNet-18 model.
    It just facilitates to load the model and to predict the top class.

    '''

    def __init__(self) -> None:

        # load pipeline (preprocessor, model and postprocessor)
        self.pipe = pipeline(
            task='image-classification',
            model='microsoft/resnet-18'
        )

    def __call__(self, images: Image.Image | Sequence[Image.Image]) -> list[list[dict[str, str | float]]]:
        '''Predict.'''
        return self.pipe(images, top_k=1)
