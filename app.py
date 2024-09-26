'''
Simple inference server (Flask).

Summary
-------
The inference can be launched through:
python app.py

A request can then send by:
curl -X POST http://localhost:5000/predict -F image=@test.jpg

'''

# from io import BytesIO

from PIL import Image
from flask import Flask, jsonify, request

from utils import TVResNet18


# initialize pretrained model
model = TVResNet18()


# create Flask application
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():

    if request.method == 'POST':

        # get image
        image = request.files['image']

        # image = Image.open(BytesIO(image.read()))
        image = Image.open(image)

        # predict
        result = model(image)[0]

        return jsonify(result)


if __name__ == '__main__':

    app.run()

