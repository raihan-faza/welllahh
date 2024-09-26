import random
import string
from uuid import uuid4

import numpy as np
import tensorflow as tf
from django.conf import settings
from django.contrib.auth import authenticate, login
from django.contrib.auth.models import User
from django.core.files.storage import default_storage
from django.db.models import Q
from django.http.response import JsonResponse
from django.shortcuts import redirect, render
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing.image import img_to_array, load_img
from PIL import Image
from tensorflow.python.keras.backend import set_session

from .models import CustomUser


def register_user(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")
        email = request.POST.get("email")
        age = request.POST.get("age")

        user_exists = CustomUser.objects.filter(
            Q(username=username) | Q(email=email)
        ).exists()
        if user_exists:
            return JsonResponse(
                {"message": "Username tersebut sudah ada, ganti username lain yaa."}
            )
        user = User(username=username, password=password)
        user.save()
        custom_user = CustomUser(
            user=user,
            email=email,
            secret_phrase=generate_random_string,
            status="known",
            age=age,
        )
        custom_user.save()
        # login(request, user)
        return JsonResponse(
            {
                "User Berhasil Dibuat": f"Nama:{custom_user.user.username}\nSecretPhrase={custom_user.secret_phrase}"
            }
        )
    return JsonResponse({"message": "html soon"})


def register_anonymous_user(request):
    if request.method == "POST":
        username = request.POST.get("username")
        user = User.objects.create(username=username)
        custom_user = CustomUser(
            user=user, passphrase=generate_random_string, status="anonymous"
        )
        custom_user.save()
        return JsonResponse(
            {"User Berhasil Dibuat": f"fSecretPhrase:{custom_user.secret_phrase}"}
        )
    return JsonResponse({"message": "html soon"})


def normal_login(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")
        user = authenticate(username=username, password=password)
        if user is not None:
            login(request, user=user)
        else:
            return JsonResponse({"message": "invalid creds"})
    return JsonResponse({"message": "html soon"})


def login_using_passphrase(request):
    if request.method == "POST":
        secret_phrase = request.POST.get("secret_phrase")
        user = User.objects.filter(secret_phrase=secret_phrase).get()
        user = authenticate(username=user.username, password=user.password)
        if user is not None:
            login(request, user=user)
        else:
            return JsonResponse({"message": "invalid phrase"})
    return JsonResponse({"message": "html soon"})


def generate_login_token(request):
    pass



def GetFoodNutrition(foodName: str):
    foodName = foodName.lower()
    myFoodNutrition = {}
    for idx, food in settings.INDOFOOD_NUTRITIONS_DF.iterrows():
        if foodName == food["food_name"]:
            myFoodNutrition["calories"] = food["calories"]
            myFoodNutrition["fat"] = food["fat"]
            myFoodNutrition["carbohydrate"] = food["carbohydrate"]
            myFoodNutrition["proteins"] = food["proteins"]
    return myFoodNutrition


def inference_indofood_image(request):
    if request.method == "POST":
        file = request.FILES["imageFile"]
        file_name = default_storage.save(file.name, file)
        file_url = default_storage.path(file_name)

        image = load_img(file_url, target_size=(320, 320))
        numpy_array = img_to_array(image)

        image_batch = np.expand_dims(numpy_array, axis=0)
        prediction = settings.INDOFOOD_IMAGE_MODEL.predict(image_batch)

        labels = settings.INDOFOOD_IMAGE_LABELS
        res = labels[tf.argmax(prediction[0])]
       
        nutritions = GetFoodNutrition(res[1])
        return render(
            request, "index.html", {"predictions": res, "nutritions": nutritions}
        )
    else:
        return render(request, "index.html")


def generate_random_string(length=12):
    characters = string.ascii_letters + string.digits
    return "".join(random.choice(characters) for _ in range(length))
