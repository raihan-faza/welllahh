import random
import string
from uuid import uuid4

from django.contrib.auth import authenticate, login
from django.contrib.auth.models import User
from django.db.models import Q
from django.http.response import JsonResponse
from django.shortcuts import redirect, render
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing.image import img_to_array, load_img
from tensorflow.python.keras.backend import set_session
from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import default_storage
from PIL import Image
import numpy as np
import tensorflow as tf 

from .models import CustomUser

# Create your views here.


def register_user(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")
        email = request.POST.get("email")

        user_exists = CustomUser.objects.filter(
            Q(username=username) | Q(email=email)
        ).exists()
        if user_exists:
            return JsonResponse(
                {"message": "Username tersebut sudah ada, ganti username lain yaa."}
            )
        user = User(username=username, password=password)
        user.save()
        custom_user = CustomUser(user=user, email=email)
        custom_user.save()
        login(request, user)
        return redirect(request, "index.html")
    return render(request, "register.html")


def register_anonymous_user(request):
    if request.method == "POST":
        passphrase = request.POST.get("passphrase")
        custom_user = CustomUser.objects.filter(passphrase=passphrase)
    return render(request, "anonymous_login.html")


def normal_login(request):
    username = request.POST.get("login")
    password = request.POST.get("password")
    pass


def login_using_passphrase(request):
    pass


def generate_login_token(request):
    pass

def GetFoodNutrition(foodName: str):
    foodName = foodName.lower()
    myFoodNutrition = {}
    for  idx, food  in settings.INDOFOOD_NUTRITIONS_DF.iterrows():
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

    
        image = load_img(file_url, target_size=(224, 224))
        numpy_array = img_to_array(image)
        

        image_batch = np.expand_dims(numpy_array, axis=0)
        prediction  = settings.INDOFOOD_IMAGE_MODEL.predict(image_batch)
            
        sorted_categories = np.argsort(prediction[0])[:-65:-1]
        labels = settings.INDOFOOD_IMAGE_LABELS
        prob = np.sort(prediction)
        cats = []
        i = 0 
        prob = prob[0][:-65:-1]

        res = []
        for predCat in sorted_categories:
            if prob[i] > 0.15:
                cats.append(labels[predCat])
            i +=1
        if len(cats) == 0:
            res = [labels[tf.argmax(prediction[0])]]
        else:
            res = cats
        # label = decode_predictions(predictions, top=1)
        nutritions =  GetFoodNutrition(res[0][1])
        return render(request, "index.html", {"predictions": res, "nutritions":  nutritions})
    else:
        return render(request, "index.html")

def generate_random_string(length=12):
    characters = string.ascii_letters + string.digits
    return "".join(random.choice(characters) for _ in range(length))
