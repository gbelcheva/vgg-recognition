# vgg-recognition

VGG model for fruit recognition trained on https://www.kaggle.com/datasets/sshikamaru/fruit-recognition.

Requires Docker.

```
docker build -f Dockerfile -t vgg .
```
```
docker run -p 8000:8000 -it vgg
```

Open http://0.0.0.0:8000/vgg and submit an image for recognition.
