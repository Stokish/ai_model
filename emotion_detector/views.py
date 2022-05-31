from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from skimage import transform
import numpy as np
import cv2

from cv_api.imp_func import _grab_image, _grab_model, _grab_faces

EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral","Sad","Surprised"]


@csrf_exempt
def detect(request):
    # initialize the data dictionary to be returned by the request
    data = {"success": False}
    # check to see if this is a post request
    if request.method == "POST":
        # check to see if an image was uploaded
        if request.FILES.get("image", None) is not None:
            # grab the uploaded image
            image_1, image_2 = _grab_image(stream=request.FILES["image"])
        # otherwise, assume that a URL was passed in
        else:
            # grab the URL from the request
            url = request.POST.get("url", None)
            # if the URL is None, then return an error
            if url is None:
                data["error"] = "No URL provided."
                return JsonResponse(data)
            # load the image and convert
            image_1, image_2 = _grab_image(url=url )
        # convert the image to grayscale, load the detected faces

        preds = []
        image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
        rects = _grab_faces(image_2)
        res = []
        emotions = []
        if len(rects):
            for (x, y, w, h) in rects:
                face = image_2[y:y + h, x:x + w]

                model = _grab_model('emotion_detector')

                resized = cv2.resize(face, (48, 48))

                prediction = model.predict(resized[np.newaxis, :, :, np.newaxis])

                emotion = EMOTIONS_LIST[np.argmax(prediction)]

                emotions.append(emotion)

            data.update({
                 "success": True,
                 'faces': rects,
                 'emotions': emotions,
            })
        else:
            data.update({
                'success': False,
                'error': 'No face detected. Upload another image, with face closer or further from camera'
            })


    # return a JSON response
    return JsonResponse(data)