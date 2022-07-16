import pickle

from fastapi import FastAPI, Request, File
from fastapi.templating import Jinja2Templates
from PIL import Image
from torchvision import transforms
import torch
import io

app = FastAPI()
templates = Jinja2Templates(directory="templates/")

CLASS_NAMES = ['Apple Braeburn', 'Apple Granny Smith', 'Apricot', 'Avocado', 'Banana', 'Blueberry', 'Cactus fruit', 'Cantaloupe', 'Cherry', 'Clementine', 'Corn', 'Cucumber Ripe', 'Grape Blue', 'Kiwi', 'Lemon', 'Limes', 'Mango', 'Onion White', 'Orange', 'Papaya', 'Passion Fruit', 'Peach', 'Pear', 'Pepper Green', 'Pepper Red', 'Pineapple', 'Plum', 'Pomegranate', 'Potato Red', 'Raspberry', 'Strawberry', 'Tomato', 'Watermelon']



class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

with open('vgg16.ml', 'rb') as pickle_file:
    VGG16 = CPU_Unpickler(pickle_file).load()


def recognise_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path)
    image = preprocess(image).unsqueeze(0)

    output = VGG16(image)
    output
    _, pred = torch.max(output.data, 1)
    prediction = pred.item()

    return CLASS_NAMES[prediction]

@app.get("/")
async def hello():
    return {"Hello"}

@app.get("/vgg")
def form_post(request: Request):
    result = ""
    return templates.TemplateResponse('form.html', context={'request': request, 'result': result})

@app.post("/vgg")
async def form_post(request: Request, file: bytes = File(...)):
    try:
        data = io.BytesIO(file)
        predicted_class = recognise_image(data)
    except Exception as ex:
        return {"message": "Something went wrong. " + ex}
    result = predicted_class
    return templates.TemplateResponse('form.html', context={'request': request, 'result': result})