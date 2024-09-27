# specify parent image
FROM python:latest

# copy to image
WORKDIR /usr/src/app
COPY . .

# install updates and requirements
RUN apt update -y && \
  pip install --no-cache-dir -r requirements.txt

# expose port
EXPOSE 5000

# run at container start
# CMD ["python", "app.py"]
CMD ["flask", "run", "--host=0.0.0.0"]

