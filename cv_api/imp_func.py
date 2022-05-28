import os
import sys
import urllib.request as urlibi
from keras.models import model_from_json
from keras.preprocessing import image as imager
from PIL import Image
import io
import ssl
import cv2
import numpy as np
import certifi
from pathlib import Path
import tensorflow as tf
def _grab_model(app_name):

    with open("{base_path}/{app}/model.json".format(
            base_path=os.path.abspath(os.getcwd()),
            app = app_name), "r") as json_file:
        loaded_model_json = json_file.read()
        loaded_model = model_from_json(loaded_model_json)

    loaded_model.load_weights(os.path.join("{base_path}/{app}/model_weights.h5".format(
            base_path=os.path.abspath(os.getcwd()),
            app = app_name) ))
    loaded_model.make_predict_function()
    return loaded_model


def _grab_image(stream=None, url=None):
    # if the URL is not None, then download the image
    if url is not None:
        resp = urlibi.urlopen(url, context=ssl.create_default_context(cafile=certifi.where()))
        data = resp.read()
    # if the stream is not None, then the image has been uploaded
    elif stream is not None:
        data = stream.read()
    # convert the image to a NumPy array and then read it into
    # OpenCV format
    image_1 = tf.keras.preprocessing.image.img_to_array(Image.open(io.BytesIO(data)))


    image_2 = np.asarray(bytearray(data), dtype="uint8")
    image_2 = cv2.imdecode(image_2, cv2.IMREAD_COLOR)
    # return the image
    return image_1, image_2


def _grab_faces(image):
    face_detect_path = "{base_path}/haarcascade_frontalface_default.xml".format(
        base_path=os.path.abspath(os.path.dirname(__file__))
    )
    detector = cv2.CascadeClassifier(face_detect_path)
    rects = detector.detectMultiScale(image, scaleFactor=1.3, minNeighbors=2, )
    rects = [(int(x), int(y), int(x + w), int(y + h)) for (x, y, w, h) in rects]
    return rects