import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from io import BytesIO
from cv2 import cv2
import tensorflow as tf

#instance
app = FastAPI()

MODEL = tf.keras.models.load_model("cotton_model")


CLASS_NAMES = ['diseased cotton leaf', 'diseased cotton plant', 'fresh cotton leaf', 'fresh cotton plant']

#ping routine
#to make sure the server is running
#/ping endpoint
@app.get("/ping")
async def ping():
    return "Test successfull"

#bytes converted to array
def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

#model prediction

@app.post("/predict")
async def predict(
        file: UploadFile = File(...) 
):  
    #file read
    image = read_file_as_image(await file.read())

    image = cv2.resize(image, dsize = (256, 256))
    #adding one more dimension
    img_batch = np.expand_dims(image, 0)
    #returns 4 values in 2D
    prediction = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
    #returns highest value
    confidence = np.max(prediction[0])

    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }


if __name__ == "__main__":
    uvicorn.run(app, debug=True)
