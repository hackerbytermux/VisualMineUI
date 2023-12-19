#torch
import torch
import torch.nn as nn
from torchvision.utils import save_image
import numpy as np
from PIL import Image

#other
from flask import Flask, url_for,render_template,request
import os
import importlib

#config
base_model = "VisualMineColor"

def load_model(name):
    model = importlib.import_module(f"models.{name}")
    return model


model = load_model(base_model)
ai = model.Model()

app = Flask(__name__,
            static_url_path='', 
            static_folder='static')



@app.route("/")
def index():
    context = {}
    context['model'] = base_model
    return render_template("index.html", context=context)

@app.route("/models")
def models():
    _models = os.listdir("models")
    context = {}
    context['models'] = _models
    context['model'] = base_model
    return render_template("models.html", context=context)

@app.route("/api/v1/generate",methods=['POST'])
def generate():
    json_data = request.get_json()
    image = np.array(json_data['img'])
    steps = json_data['steps']
    noise = json_data['noise']
    contrast = json_data['contrast']
    rgb_array = np.array(image, dtype=np.uint8)
    img = Image.fromarray(rgb_array)
    result = ai.generate(img, steps,noise,contrast)
    return {"success": True, "img": result}

@app.route("/api/v1/change_model",methods=['POST'])
def change_model():
    global ai, base_model
    json_data = request.get_json()
    name = json_data['name']
    base_model = name
    success = True
    try:
        model = load_model(name)
        ai = model.Model()
    except Exception as e:
        print(f"Failed to load model {name}, {e}")
        success = False
        return {"success": success, "error": str(e)}
    return {"success": success}


if __name__ == '__main__':
   app.run()