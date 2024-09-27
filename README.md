# ML model serving

## Instructions

1.  Build an image from the shipped dockerfile:
    ```
    sudo docker build -t flask_image_classif .
    ```

2.  Create and run a container from the image:
    ```
    sudo docker run -d -p 5000:5000 flask_image_classif
    ```

3.  Send a request to the inference server:
    ```
    curl -X POST http://localhost:5000/predict -F image=@test.jpg
    ```

