'''Hugging Face models.'''

from transformers import pipeline


# TODO: enable GPU processing
class HFResNet18():
    '''ResNet-18 Hugging Face pipeline wrapper.'''

    def __init__(self, model_name='microsoft/resnet-18'):

        # load pipeline (preprocessor, model and postprocessor)
        self.pipe = pipeline('image-classification', model=model_name)

    def __call__(self, images):
        '''Predict.'''
        return self.pipe(images, top_k=1)

