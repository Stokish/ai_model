from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from skimage import transform
import numpy as np
import cv2


from cv_api.imp_func import _grab_image, _grab_model


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
            image_1, image_2 = _grab_image(url=url)
        # convert the image to grayscale, load the face cascade detector,
        # and detect faces in the image
        image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
        preds = []

        resized_1 = transform.resize(image_1, (224, 224, 1))
        resized_2 = cv2.resize(image_2, (224, 224))

        model = _grab_model('gan_detector')

        pred_1 = model.predict(resized_1[np.newaxis, :, :, np.newaxis])
        pred_2 = model.predict(resized_2[np.newaxis, :, :, np.newaxis])

        preds.append(float(pred_1[0]) * 100)
        preds.append(float(pred_2[0]) * 100)

        dif = abs(preds[0] - preds[1])
        is_real = False
        if dif < 1 and preds[0] + preds[1] > 180:
            is_real = True

        data.update({
             "success": True,
             'pred_transform': preds[0],
             'pred_cv2': preds[1],
             'dif': dif,
             "is_real": is_real
        })

    # return a JSON response
    return JsonResponse(data)






