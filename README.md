# ML model serving

A small demonstration of deploying ML models as an online API endpoint is provided.
We rely on [Flask](https://flask.palletsprojects.com/), a lightweight framework for building web apps,
for implementing an inference server.
It provides access to pretrained models through HTTP requests.

## Instructions

The simplest way to get started is running an inference server locally.
It can then be easily requested to make model predictions for some given input data.
The whole procedure just involves the following two steps.

1.  Start a local inference server:
    ```
    python app.py
    ```

2.  Send a POST request to the inference server:
    ```
    curl -X POST http://localhost:5000/predict -F image=@test.jpg
    ```

Another possibility is to package and run the application inside a Docker container.
Such a container basically provides a VM-like isolated environment.
After installing the [Docker Engine](https://docs.docker.com/engine/),
and starting the service through `sudo systemctl start docker` (on many Linux systems at least),
one may simply proceed as follows.

1.  Build an image from the dockerfile:
    ```
    sudo docker build -t flask_image_classif .
    ```

2.  Create and run a container from the image:
    ```
    sudo docker run --name imgclassif -d -p 5000:5000 flask_image_classif
    ```

The running container can be stopped through `sudo docker stop imgclassif`
and restarted at any time through `sudo docker start imgclassif`.
A request can be send as before by running an appropriate `curl` command.

