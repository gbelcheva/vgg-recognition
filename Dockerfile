FROM python:3.9

WORKDIR /app

COPY ./Pipfile.lock /app/Pipfile.lock

RUN pip install pipenv uvicorn jinja2 python-multipart Pillow torch torchvision
RUN pipenv sync

COPY ./main.py /app/
COPY ./templates /app/templates

ADD https://drive.google.com/u/1/uc?id=1IdqzmyuijIi-jth2uLrihntk7qfyfG_a&export=download&confirm=t&uuid=df70eb51-5638-470c-bc5c-df4ac6c5166 /app/vgg16.ml

CMD ["pipenv",  "run",  "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]