from flask import Flask, render_template, request
import os
from model_predict import predict_waste

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

reuse_ideas = {
    'Plastic Bottle': {
        'idea': 'Convert plastic bottles into plant pots or storage containers.'
    },
    'Glass Bottle': {
        'idea': 'Reuse glass bottles as decorative lamps or flower vases.'
    },
    'Cardboard': {
        'idea': 'Use cardboard for DIY organizers or packaging reuse.'
    },
    'Aluminum Can': {
        'idea': 'Transform cans into pen holders or mini planters.'
    },
    'Cloth Waste': {
        'idea': 'Reuse cloth waste for cleaning wipes or handmade bags.'
    }
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return 'No image uploaded'

    file = request.files['image']

    if file.filename == '':
        return 'No selected image'

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    prediction = predict_waste(filepath)

    recommendation = reuse_ideas.get(prediction, {
        'idea': 'No recommendation available.'
    })

    return render_template(
        'result.html',
        prediction=prediction,
        image_path=filepath,
        recommendation=recommendation['idea']
    )

if __name__ == '__main__':
    app.run(debug=True)
