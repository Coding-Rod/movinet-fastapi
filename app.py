from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import RedirectResponse
import uvicorn

import hashlib as hl
import numpy as np
import pathlib
import os

import tensorflow as tf
import tensorflow_hub as hub

description = """
This is an API implementation of the [MoViNet](https://www.tensorflow.org/hub/tutorials/movinet) model.

The classes of the model are the 600 classes of the Kinetics-600 dataset. 
The full list of labels can be found at [this link](https://raw.githubusercontent.com/tensorflow/models/f8af2291cced43fc9f1d9b41ddbf772ae7b0d7d2/official/projects/movinet/files/kinetics_600_labels.txt).

Author: Rodrigo Fernandez  
GitHub: [@Coding-Rod](https://github.com/Coding-Rod)
"""

app = FastAPI(description=description)

# Tensorflow configuration

labels_path = tf.keras.utils.get_file(
    fname='labels.txt',
    origin='https://raw.githubusercontent.com/tensorflow/models/f8af2291cced43fc9f1d9b41ddbf772ae7b0d7d2/official/projects/movinet/files/kinetics_600_labels.txt'
)
labels_path = pathlib.Path(labels_path)

lines = labels_path.read_text().splitlines()
LABEL_MAP = np.array([line.strip() for line in lines])

id = 'a2'
mode = 'base'
version = '3'
hub_url = f'https://tfhub.dev/tensorflow/movinet/{id}/{mode}/kinetics-600/classification/{version}'
model = hub.load(hub_url)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_gif(file_path, image_size=(224, 224)):
    """Loads a gif file into a TF tensor.

    Use images resized to match what's expected by your model.
    The model pages say the "A2" models expect 224 x 224 images at 5 fps

    Args:
        file_path: path to the location of a gif file.
        image_size: a tuple of target size.

    Returns:
        a video of the gif file
    """
    
    # Load a gif file, convert it to a TF tensor
    raw = tf.io.read_file(file_path)
    video = tf.io.decode_gif(raw)
    # Resize the video
    video = tf.image.resize(video, image_size)
    # change dtype to a float32
    # Hub models always want images normalized to [0,1]
    # ref: https://www.tensorflow.org/hub/common_signatures/images#input
    video = tf.cast(video, tf.float32) / 255.
    return video

def inference(video):
    # Add model signature
    sig = model.signatures['serving_default']
    
    # Outer batch dimension
    sig(image = video[tf.newaxis, :1])
    
    # Create logits
    logits = sig(image = video[tf.newaxis, ...])
    logits = logits['classifier_head'][0]

    # Gettings probs
    probs = tf.nn.softmax(logits, axis=-1)
    
    # Sort predictions to find top_k
    top_predictions = tf.argsort(probs, axis=-1, direction='DESCENDING')
    
    # collect the labels of top_k predictions
    top_labels = tf.gather(LABEL_MAP, top_predictions, axis=-1)
    
    # decode lablels
    top_labels = [label.decode('utf8') for label in top_labels.numpy()]
    
    # top_k probabilities of the predictions
    top_probs = tf.gather(probs, top_predictions, axis=-1).numpy()
    
    return f"The result is {top_labels[0]} with a probability of {top_probs[0]:.2%}"

@app.post("/get_inference")
async def get_inference(file: UploadFile = File(...)):
    """ This function receives a gif file and returns the prediction of the model
    
    Args:
        file (UploadFile, optional): The gif file to be processed.

    Returns:
        dict: The prediction of the model
    """
    # Create gifs folder if it doesn't exist
    os.makedirs("uploads", exist_ok=True)

    # Save the received gif file
    filename = "uploads/" + hl.md5(file.filename.encode()).hexdigest() + ".gif"
    with open(filename, "wb") as f:
        f.write(await file.read())
    
    # Make predictions
    result = inference(load_gif(filename))
    
    return {"message": result}

@app.get("/", include_in_schema=False)
async def root():
    # Redirect to the documentation
    return RedirectResponse(url="/docs")

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)