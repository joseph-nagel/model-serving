'''Hugging Face models.'''

from transformers import pipeline


# TODO: enable GPU processing
class HFResNet18():
    '''
    ResNet-18 Hugging Face pipeline wrapper.

    Summary
    -------
    This is a simple wrapper for a pretrained ResNet-18 model.
    It just facilitates to load the model and to predict the top class.

    '''

    def __init__(self):

        # load pipeline (preprocessor, model and postprocessor)
        self.pipe = pipeline(
            task='image-classification',
            model='microsoft/resnet-18'
        )

    def __call__(self, images):
        '''Predict.'''
        return self.pipe(images, top_k=1)

