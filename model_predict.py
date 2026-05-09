import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model = load_model('models/waste_model.h5')

classes = [
    'Plastic Bottle',
    'Glass Bottle',
    'Cardboard',
    'Aluminum Can',
    'Cloth Waste'
]

def predict_waste(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)

    return classes[class_index]
