from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from skimage import transform
import numpy as np
import cv2


from cv_api.imp_func import _grab_image, _grab_model, _grab_faces


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
            images = _grab_image(url=url)
        pred = []
        is_grab_face = request.POST.get('is_grab_face', False)
        print(str(is_grab_face).lower())
        rects = None
        if str(is_grab_face).lower() in ['true', '1']:
            rects = _grab_faces(image_2)
            print(rects)
            res = []
            for (x, y, w, h) in rects:
                faces = image_2[y:y + h, x:x + w]
                print('a')
                resized = cv2.resize(faces, (100, 100))
                print('b')
                model = _grab_model('liveness_detector')
                pred = model.predict(resized[np.newaxis, :, :])

                res.append(pred[0][0])
            is_live = []
            for i in range(len(res)):
                print(res[i])
                res[i] = (1 - res[i]) * 100
                if res[i] > 60:
                    is_live.append(True)
                else:
                    is_live.append(False)

            print(res)
        else:

            resized = transform.resize(image_1, (100, 100, 3))

            model = _grab_model('liveness_detector')
            pred = model.predict(resized[np.newaxis, :, :])
            res = pred[0][0]
            is_live = False
            res = (1 - res) * 100
            if res > 60:
                is_live = True
            print(res)



        data.update({
             "success": True,
             'pred': res,
              "is_live": is_live,
              "is_grab_face":str(is_grab_face).lower() in ['true', '1']
        })
        if rects is not None:
            data.update({
                "faces": rects
            })

    # return a JSON response
    return JsonResponse(data)






