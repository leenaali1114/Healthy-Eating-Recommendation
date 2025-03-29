import os
import uuid
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf

from utils.freshness import load_freshness_model, predict_freshness
from utils.recipes import recommend_recipes

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the model at startup
model = load_freshness_model()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        # Generate unique filename
        filename = str(uuid.uuid4()) + secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Get other ingredients from form
        other_ingredients = request.form.get('ingredients', '').split(',')
        other_ingredients = [ing.strip().lower() for ing in other_ingredients if ing.strip()]
        
        # Predict freshness
        result = predict_freshness(filepath, model)
        
        # Get recipe recommendations
        recipes = recommend_recipes(
            result['fruit_type'], 
            other_ingredients, 
            result['is_fresh']
        )
        
        # Prepare response
        response = {
            'image_path': filepath,
            'freshness': result,
            'recipes': recipes
        }
        
        return render_template('result.html', result=response)
    
    return jsonify({'error': 'Invalid file type'})

if __name__ == '__main__':
    app.run(debug=True)
