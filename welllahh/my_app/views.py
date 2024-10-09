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
from .models import *
from django.http.response import JsonResponse

import datetime

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


def get_bmi_condition(request):
    if (request.method == "GET"):
        weight = request.POST.get("weight")
        height = request.POST.get("height")
        bmi = round((weight/(height * height)), 2)
        # https://en.wikipedia.org/wiki/Body_mass_index
        if bmi >= 30.0:
            return JsonResponse(data={"condition": "Obesitas"}) 
        elif bmi >= 25.0 and bmi < 30:
            return JsonResponse(data={"condition": "Overweight (Pre-obesitas)"}) 
        elif bmi >= 18.5 and bmi < 25.0:
            return JsonResponse(data={"condition": "Normal"}) 
        elif bmi >= 17.0 and bmi < 18.5:
            return JsonResponse(data={"condition": "Underweight (Mild thinness)"})
        elif bmi >= 16.0 and bmi < 17.0:
            return JsonResponse(data={"condition": "Underweight (Moderate thinness)"})
        elif bmi >= 16.0 and bmi < 17.0:
            return JsonResponse(data={"condition": "Underweight (Severe thinness)	"})
        


def get_health_condition_based_on_nutrition_val(request):
    if (request.method == "GET"):
        nutrition_type = request.POST.get("nutrition_type")
        nutrition_value = request.POST.get("nutrition_value")
        user = request.user
        user_age = user.age
        if nutrition_type == "BLOOD_SUGAR":
            # https://www.halodoc.com/kesehatan/diabetes?srsltid=AfmBOorAHmPVZ_SFLW0q8RSkk8xjPeMjfor1bEsOx3HHs1kgrNbvApR9
           
            if nutrition_value >= 126:
                return JsonResponse(data={"disease_prediction": "diabetes"})
            elif nutrition_value >= 100 and nutrition_value <= 125:
                return JsonResponse(data={"disease_prediction": "prediabetes"})
            else:
                return JsonResponse(data={"disease_prediction": "normal"})
           
        if nutrition_type == "CHOLESTEROL":
            if user_age >= 20:
                if nutrition_value > 200:
                    # https://www.siloamhospitals.com/en/informasi-siloam/artikel/kadar-kolesterol-normal
                    return JsonResponse(data={"disease_prediction": "Penyakit Jantung dan stroke"})
                else:
                    return JsonResponse(data={"disease_prediction": "normal"})
            else:
                if nutrition_value > 170:
                    # https://www.siloamhospitals.com/en/informasi-siloam/artikel/kadar-kolesterol-normal
                    return JsonResponse(data={"disease_prediction": "Penyakit Jantung dan stroke"})
                else:
                    return JsonResponse(data={"disease_prediction": "normal"})
        if nutrition_type == "URIC_ACID":
            # asam urat
            # https://www.siloamhospitals.com/en/informasi-siloam/artikel/prosedur-cek-asam-urat
            if nutrition_value >= 7.0:
                return JsonResponse(data={"disease_prediction": "Serangan Asam Urat Dan Batu Ginjal"})
            else:
                return JsonResponse(data={"disease_prediction": "normal"})
        if nutrition_type == "BLOOD_PRESSURE":
            # https://www.siloamhospitals.com/informasi-siloam/artikel/mengenal-tekanan-darah-normal
            if user_age >= 18:
                if nutrition_value > 120:
                    return JsonResponse(data={"disease_prediction": "Hipertensi"})
                else:
                    return JsonResponse(data={"disease_prediction": "normal"})
            elif user_age < 18 and user_age >= 13:
                if nutrition_value > 128:
                    return JsonResponse(data={"disease_prediction": "Hipertensi"})
                else:
                    return JsonResponse(data={"disease_prediction": "normal"})
            elif user_age >= 6 and user_age <= 12:
                if nutrition_value > 131:
                    return JsonResponse(data={"disease_prediction": "Hipertensi"})
                else:
                    return JsonResponse(data={"disease_prediction": "normal"})
            elif user_age >= 3 and user_age < 6:
                if nutrition_value > 120:
                    return JsonResponse(data={"disease_prediction": "Hipertensi"})
                else:
                    return JsonResponse(data={"disease_prediction": "normal"})
            else:
                if nutrition_value > 100:
                    return JsonResponse(data={"disease_prediction": "Hipertensi"})
                else:
                    return JsonResponse(data={"disease_prediction": "normal"})





def catat_nutrisi(request):
    if (request.method == "POST"):
        nutrition_type = request.POST.get("nutrition_type")
        nutrition_value = request.POST.get("nutrition_value")
        check_time = request.POST.get("check_time")
        user = request.user
        nutrition_progress = NutritionProgress(
            nutrition_type=nutrition_type, nutrition_value=nutrition_value,
            check_time=check_time, user=user
        )

        nutrition_progress.save()




def catat_tinggi_berat(request):
    if (request.method =="POST"):
        height = request.POST.get('height')
        weight = request.POST.get('weight')
        muscle_mass = request.POST.get("muscle_mass")
        fat_mass = request.POST.get("fat_mass")
        check_time = request.POST.get("check_time")
        
        body_info = UserBodyInfo(
            weight=weight, muscle_mass=muscle_mass, height=height,
            fat_mass=fat_mass, check_time=check_time, custom_user=request.user
        )
        body_info.save()

    
