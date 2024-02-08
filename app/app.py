from flask import Flask, request, render_template
import torch
from model import *
from utils import *

app = Flask(__name__)

# Initialize and load models dynamically
def load_model(model_name, model_path):
    model = initialize_model(model_name)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model

models = {
    'general': load_model('general_attention', './models/general_attention.pt'),
    'multiplicative': load_model('multiplicative_attention', './models/multiplicative_attention.pt'),
    'additive': load_model('additive_attention', './models/additive_attention.pt'),
}

# Define the index route
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html', query = '')

    if request.method == 'POST':
        src_sentence   = request.form['src_sentence']
        model_type     = request.form['model_type']
        model          = models[model_type]
        translation, _ = translate_sentence(model, src_sentence)
        return render_template('index.html', translation=translation, src_sentence=src_sentence, model_type=model_type)
    else:
        return render_template('index.html', translation=None)

if __name__ == '__main__':
    app.run(debug=True)
