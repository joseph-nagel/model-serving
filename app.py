'''
Simple Flask inference server.

Summary
-------
A very simple inference server is implemented with Flask.
It allows for deploying ML models as a web application.

The inference server is launched through:
python app.py

A request can then be sent by:
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

        # run model
        result = model(image)[0]

        return jsonify(result)


if __name__ == '__main__':

    app.run()

