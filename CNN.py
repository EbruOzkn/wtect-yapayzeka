# CNN model deployment
# pip install python-multipart

from fastapi import File, UploadFile, FastAPI
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io 


app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Welcome to the CNN API."}

@app.post("/predict_image/")
async def predict_image(file: UploadFile = File(...)): 
    load_ann = load_model('models/m.h5')
    contents = await file.read() 
    img = image.load_img(io.BytesIO(contents), target_size=(210, 140)) 
    img_array = image.img_to_array(img) 
    img_array = np.expand_dims(img_array, axis=0) 
    prediction = load_ann.predict(img_array) 
    predicted_class_index = np.argmax(prediction[0]) 

    class_labels = ["airplane","car","ship"] 

    prediction = class_labels[predicted_class_index] 
    return {"prediction": prediction}


# python -m uvicorn CNN:app --reload