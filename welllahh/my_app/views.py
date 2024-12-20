import datetime
import json
import random
import string
from datetime import date
from uuid import uuid4

import numpy as np
import tensorflow as tf
from constraint import *
from django.conf import settings
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.core.files.storage import default_storage
from django.db.models import Q, Sum, Avg

# from .medical_ai_chatbot import answer_pipeline
from django.db.models.functions import TruncDate
from django.http.response import HttpResponse, JsonResponse
from django.shortcuts import redirect, render
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing.image import img_to_array, load_img
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.python.keras.backend import set_session

from .medical_ai_chatbot import answer_pipeline
from .models import *


def landing_page(request):
    return render(request, "welllahh_landing_page.html")


@csrf_exempt
def register_user(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")
        email = request.POST.get("email")
        age = request.POST.get("age")

        user_exists = CustomUser.objects.filter(
            Q(user__username=username) | Q(user__email=email)
        ).exists()
        if user_exists:
            return JsonResponse(
                {"message": "Username tersebut sudah ada, ganti username lain yaa."}
            )
        user = User.objects.create_user(
            username=username, password=password, email=email
        )
        custom_user = CustomUser(
            user=user,
            secret_phrase=generate_random_string,
            status="known",
            age=age,
        )
        custom_user.save()
        login(request, user)
        return redirect("my_app:catat_tinggi_berat")
    return render(request, "register.html")


@csrf_exempt
def register_anonymous_user(request):
    if request.method == "POST":
        username = request.POST.get("username")
        user = User.objects.create(username=username)
        custom_user = CustomUser(
            user=user, passphrase=generate_random_string, status="anonymous"
        )
        custom_user.save()
        return redirect("my_app:dashboard")
    return render(request, "anonymous_register.html")


@csrf_exempt
def normal_login(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user=user)
            return redirect("my_app:dashboard")
        else:
            return JsonResponse(
                {"message": f"invalid creds {user}-{username}-{password}"}
            )
    return render(request, "login.html")


@csrf_exempt
def login_using_passphrase(request):
    if request.method == "POST":
        secret_phrase = request.POST.get("secret_phrase")
        user = User.objects.filter(secret_phrase=secret_phrase).get()
        user = authenticate(username=user.username, password=user.password)
        if user is not None:
            login(request, user=user)
            return redirect("my_app:dashboard")
        else:
            return JsonResponse({"message": "invalid phrase"})
    return render(request, "anonymous_login.html")


def generate_login_token(request):
    pass


def GetFoodNutrition(foodName: str):
    foodName = foodName.lower()
    myFoodNutrition = {}
    for idx, food in settings.INDOFOOD_NUTRITIONS_DF.iterrows():
        if foodName == food["food_name"]:
            myFoodNutrition["calories"] = food["calories"].split("kcal")[0]
            myFoodNutrition["fat"] = food["fat"]
            myFoodNutrition["carbohydrate"] = food["carbohydrate"]
            myFoodNutrition["proteins"] = food["proteins"]
    return myFoodNutrition


def inference_indofood_image(request):
    if request.method == "POST":
        if request.POST.get("submit-tracker"):
            nutrition_name = request.POST.get("nutrition-name")
            calorie = request.POST.get("calorie").replace("kcal", "")
            carbs = request.POST.get("carbs").replace("g", "")
            protein = request.POST.get("protein").replace("g", "")
            fat = request.POST.get("fat").replace("g", "")
            user = CustomUser.objects.get(user=request.user)
            nutrition = NutritionProgress(
                user=user,
                nutrition_name=nutrition_name,
                calorie=calorie,
                carbs=carbs,
                protein=protein,
                fat=fat,
            )
            nutrition.save()
            return redirect("my_app:dashboard")
        file = request.FILES.get("imageFile")
        if file is not None:
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
                request,
                "food_nutrition.html",
                {"predictions": res, "nutritions": nutritions},
            )
        else:
            return
    else:
        feature_tour = get_feature_tour(request)
        feature_tour_item = feature_tour["food_nutrition_page"]

        return render(
            request,
            "food_nutrition.html",
            {
                "name": "Unknown",
                "nutritions": {
                    "calories": 0,
                    "fat": 0,
                    "carbohydrate": 0,
                    "proteins": 0,
                },
                "feature_tour": feature_tour_item,
            },
        )


def generate_random_string(length=12):
    characters = string.ascii_letters + string.digits
    return "".join(random.choice(characters) for _ in range(length))


def get_bmi_condition(request):
    if request.method == "GET":
        weight = request.POST.get("weight")
        height = request.POST.get("height")
        bmi = round((weight / (height * height)), 2)
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
    if request.method == "GET":
        nutrition_type = request.POST.get("nutrition_type")
        nutrition_value = request.POST.get("nutrition_value")
        user = request.user
        custom_user = CustomUser.objects.get(user=user)
        user_age = custom_user.age
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
                    return JsonResponse(
                        data={"disease_prediction": "Penyakit Jantung dan stroke"}
                    )
                else:
                    return JsonResponse(data={"disease_prediction": "normal"})
            else:
                if nutrition_value > 170:
                    # https://www.siloamhospitals.com/en/informasi-siloam/artikel/kadar-kolesterol-normal
                    return JsonResponse(
                        data={"disease_prediction": "Penyakit Jantung dan stroke"}
                    )
                else:
                    return JsonResponse(data={"disease_prediction": "normal"})
        if nutrition_type == "URIC_ACID":
            # asam urat
            # https://www.siloamhospitals.com/en/informasi-siloam/artikel/prosedur-cek-asam-urat
            if nutrition_value >= 7.0:
                return JsonResponse(
                    data={"disease_prediction": "Serangan Asam Urat Dan Batu Ginjal"}
                )

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
    if request.method == "POST":
        nutrition_type = request.POST.get("nutrition_type")
        nutrition_value = request.POST.get("nutrition_value")
        check_time = request.POST.get("check_time")
        user = request.user
        nutrition_progress = NutritionProgress(
            nutrition_type=nutrition_type,
            nutrition_value=nutrition_value,
            check_time=check_time,
            user=user,
        )

        nutrition_progress.save()


@login_required(login_url="my_app:normal_login")
def catat_tinggi_berat(request):
    if request.method == "POST":
        custom_user = CustomUser.objects.get(user=request.user)
        height = request.POST.get("height")
        weight = request.POST.get("weight")
        muscle_mass = request.POST.get("muscle_mass")
        fat_mass = request.POST.get("fat_mass")
        check_time = request.POST.get("check_time")

        body_info = UserBodyInfo(
            weight=weight,
            muscle_mass=muscle_mass,
            height=height,
            fat_mass=fat_mass,
            check_time=check_time,
            custom_user=custom_user,
        )
        body_info.save()
        return redirect("my_app:dashboard")

    feature_tour = get_feature_tour(request)
    feature_tour_item = feature_tour["record_bmi_page"]
    return render(
        request,
        "add_bmi.html",
        {"curr_date": date.today(), "feature_tour": feature_tour_item},
    )


# food recommendations


def show_best_results(df, scores_array, top_n=1):
    sorted_indices = scores_array.argsort()[::-1]
    for position, idx in enumerate(sorted_indices[:top_n]):
        row = df.iloc[idx]
        food_name = row["Name"]
        score = scores_array[idx]
        return food_name


def query_food_name(query):
    query_vectorized = settings.SEARCH_COUNT_VEC.transform([query])
    scores = query_vectorized.dot(settings.SEARCH_MATRIX_VEC.transpose())
    scores_array = scores.toarray()[0]
    return show_best_results(settings.FOOD_DF, scores_array)


def get_recommendations(food_name, cosine_sim=settings.FOOD_COSINE_SIM):
    # matched_food_name = query_food_name(food_name)
    idx = settings.INDICES[food_name]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    if len(sim_scores) > 49:
        sim_scores = sim_scores[1:50]
    else:
        sim_scores = sim_scores[1 : len(sim_scores)]

    food_indices = [i[0] for i in sim_scores]

    return settings.FOOD_DF.iloc[food_indices]


def calorie_intake(w, h, a):
    cal = 10 * w + 6.25 * h - 5 * a + 5
    return cal


def get_all_food_recommendations(my_favorite_food):
    foods_recommendation = []
    for myfood in my_favorite_food:
        food_recommendations_from_cbf = get_recommendations(myfood)
        food_recommendations_from_cbf = food_recommendations_from_cbf.to_dict("records")
        for new_food in food_recommendations_from_cbf:
            foods_recommendation.append(new_food)
    return foods_recommendation


def filter_based_on_disease(
    breakfast, lunch, dinner, calorie_intake, disease, category, target_plan
):
    """
    https://journal.ipb.ac.id/index.php/jtep/article/view/52452/27802
    """

    saturated_fat = (
        breakfast["SaturatedFatContent"]
        + lunch["SaturatedFatContent"]
        + dinner["SaturatedFatContent"]
    )
    fat = breakfast["FatContent"] + lunch["FatContent"] + dinner["FatContent"]
    mufa_pufa = fat - saturated_fat

    max_calorie = (
        breakfast["Calories"] + lunch["Calories"] + dinner["Calories"]
        <= calorie_intake * 1.4
    )

    makanan_bervariasi = breakfast != lunch and lunch != dinner and breakfast != dinner
    match_category = (
        breakfast["RecipeCategory"] == category["breakfast"]
        and lunch["RecipeCategory"] == category["lunch"]
        and dinner["RecipeCategory"] == category["dinner"]
    )

    if disease == "diabetes":
        max_fat = (
            breakfast["FatContent"] + lunch["FatContent"] + dinner["FatContent"]
            < 0.25 * calorie_intake
        )
        max_protein = (
            breakfast["ProteinContent"]
            + lunch["ProteinContent"]
            + dinner["ProteinContent"]
            < 0.15 * calorie_intake
        )
        max_carbo = (
            breakfast["CarbohydrateContent"]
            + lunch["CarbohydrateContent"]
            + dinner["CarbohydrateContent"]
            < 0.60 * calorie_intake
        )
        max_sodium = (
            breakfast["SodiumContent"]
            + lunch["SodiumContent"]
            + dinner["SodiumContent"]
            < 2000
        )
        max_cholesterol = (
            breakfast["CholesterolContent"]
            + lunch["CholesterolContent"]
            + dinner["CholesterolContent"]
            < 200
        )
        max_saturated_fat = saturated_fat < 0.07 * calorie_intake
        if len(category) != 0:
            return (
                max_fat
                and max_protein
                and max_carbo
                and max_sodium
                and max_cholesterol
                and max_saturated_fat
                and makanan_bervariasi
                and match_category
                and max_calorie
            )
        else:
            return (
                max_fat
                and max_protein
                and max_carbo
                and max_sodium
                and max_cholesterol
                and max_saturated_fat
                and makanan_bervariasi
                and max_calorie
            )

    elif disease == "cardiovascular":
        max_fat = (
            breakfast["FatContent"] + lunch["FatContent"] + dinner["FatContent"]
            < 0.25 * calorie_intake
        )
        max_protein = (
            breakfast["ProteinContent"]
            + lunch["ProteinContent"]
            + dinner["ProteinContent"]
            < 0.15 * calorie_intake
        )
        max_carbo = (
            breakfast["CarbohydrateContent"]
            + lunch["CarbohydrateContent"]
            + dinner["CarbohydrateContent"]
            < 0.60 * calorie_intake
        )
        max_sodium = (
            breakfast["SodiumContent"]
            + lunch["SodiumContent"]
            + dinner["SodiumContent"]
            < 2000
        )
        max_cholesterol = (
            breakfast["CholesterolContent"]
            + lunch["CholesterolContent"]
            + dinner["CholesterolContent"]
            < 200
        )
        max_saturated_fat = saturated_fat < 0.07 * calorie_intake
        if len(category) != 0:
            return (
                max_fat
                and max_protein
                and max_carbo
                and max_sodium
                and max_cholesterol
                and max_saturated_fat
                and makanan_bervariasi
                and match_category
                and max_calorie,
            )
        else:
            return (
                max_fat
                and max_protein
                and max_carbo
                and max_sodium
                and max_cholesterol
                and max_saturated_fat
                and makanan_bervariasi
                and max_calorie
            )

    elif disease == "hypertension":
        max_fat = (
            breakfast["FatContent"] + lunch["FatContent"] + dinner["FatContent"]
            < 0.25 * calorie_intake
        )
        max_protein = (
            breakfast["ProteinContent"]
            + lunch["ProteinContent"]
            + dinner["ProteinContent"]
            < 0.15 * calorie_intake
        )
        max_carbo = (
            breakfast["CarbohydrateContent"]
            + lunch["CarbohydrateContent"]
            + dinner["CarbohydrateContent"]
            < 0.60 * calorie_intake
        )
        max_sodium = (
            breakfast["SodiumContent"]
            + lunch["SodiumContent"]
            + dinner["SodiumContent"]
            < 1000
        )
        max_saturated_fat = saturated_fat < 0.07 * calorie_intake
        if len(category) != 0:
            return (
                max_fat
                and max_protein
                and max_carbo
                and max_sodium
                and max_saturated_fat
                and makanan_bervariasi
                and match_category
                and max_calorie
            )
        else:
            return (
                max_fat
                and max_protein
                and max_carbo
                and max_sodium
                and max_saturated_fat
                and makanan_bervariasi
                and max_calorie
            )

    elif disease == "normal":
        max_fat = (
            breakfast["FatContent"] + lunch["FatContent"] + dinner["FatContent"]
            < 0.25 * calorie_intake
        )
        max_protein = (
            breakfast["ProteinContent"]
            + lunch["ProteinContent"]
            + dinner["ProteinContent"]
            < 0.15 * calorie_intake
        )
        max_carbo = (
            breakfast["CarbohydrateContent"]
            + lunch["CarbohydrateContent"]
            + dinner["CarbohydrateContent"]
            < 0.60 * calorie_intake
        )
        max_sodium = (
            breakfast["SodiumContent"]
            + lunch["SodiumContent"]
            + dinner["SodiumContent"]
            < 2000
        )
        if category != "":
            return (
                max_fat
                and max_protein
                and max_carbo
                and max_sodium
                and makanan_bervariasi
                and match_category
                and max_calorie
            )
        else:
            return (
                max_fat
                and max_protein
                and max_carbo
                and max_sodium
                and makanan_bervariasi
                and max_calorie
            )
    elif disease == "target_plan":
        max_fat = (
            breakfast["FatContent"] + lunch["FatContent"] + dinner["FatContent"]
            <= target_plan["fat"]
        )
        max_protein = (
            breakfast["ProteinContent"]
            + lunch["ProteinContent"]
            + dinner["ProteinContent"]
            <= target_plan["protein"]
        )
        max_carbo = (
            breakfast["CarbohydrateContent"]
            + lunch["CarbohydrateContent"]
            + dinner["CarbohydrateContent"]
            <= target_plan["carbs"]
        )
        max_sodium = (
            breakfast["SodiumContent"]
            + lunch["SodiumContent"]
            + dinner["SodiumContent"]
            < 2000
        )
        if category != "":
            return (
                max_fat
                and max_protein
                and max_carbo
                and max_sodium
                and makanan_bervariasi
                and match_category
                and max_calorie
            )
        else:
            return (
                max_fat
                and max_protein
                and max_carbo
                and max_sodium
                and makanan_bervariasi
                and max_calorie
            )


def meal_plan(request):
    """

    example request:
    {
        "my_favorite_food": ["salad", "steak", "fried chicken", "pizza"],
        "meal_plan_category": {
                "breakfast": "Vegetable",
                "lunch": "Chicken",
                "dinner": "Steak"
            }
    }

    list food category ada di notebook food_recommendation_system.ipynb
    """
    if request.method == "POST":
        target_plan = TargetPlan.objects.filter(user__user=request.user)
        nutri_target = {}
        if target_plan.exists():
            target_plan = target_plan.first()
            nutri_target = {
                "calories": target_plan.target_calorie,
                "protein": target_plan.target_protein,
                "carbs": target_plan.target_carbs,
                "fat": target_plan.target_fat,
            }
        penyakit_user = RiwayatPenyakit.objects.filter(user__user=request.user)
        list_penyakit = []
        if penyakit_user.exists():
            for penyakit in penyakit_user:
                list_penyakit.append(penyakit.nama_penyakit.lower())
        list_penyakit = set(list_penyakit)

        my_calorie_daily_intake = calorie_intake(80, 170, 22)
        my_favorite_food = [
            request.POST.get("my_favorite_food_1"),
            request.POST.get("my_favorite_food_2"),
            request.POST.get("my_favorite_food_3"),
            request.POST.get("my_favorite_food_4"),
            request.POST.get("my_favorite_food_5"),
        ]
        breakfast_category = request.POST.get("meal_plan_category_breakfast")
        lunch_category = request.POST.get("meal_plan_category_lunch")
        dinner_category = request.POST.get("meal_plan_category_dinner")

        foods_recommendation = get_all_food_recommendations(
            my_favorite_food=my_favorite_food
        )
        random.shuffle(foods_recommendation)
        problem = Problem()
        problem.addVariable("breakfast", foods_recommendation)

        problem.addVariable("lunch", foods_recommendation)

        problem.addVariable("dinner", foods_recommendation)
        max_nutrition = {}

        if "diabetes" in list_penyakit:
            problem.addConstraint(
                lambda breakfast, lunch, dinner: filter_based_on_disease(
                    breakfast,
                    lunch,
                    dinner,
                    my_calorie_daily_intake,
                    "diabetes",
                    {
                        "breakfast": breakfast_category,
                        "lunch": lunch_category,
                        "dinner": dinner_category,
                    },
                    nutri_target,
                ),
                ("breakfast", "lunch", "dinner"),
            )

            max_nutrition = {
                "calorie": my_calorie_daily_intake,
                "fat": 0.25 * my_calorie_daily_intake,
                "protein": 0.15 * my_calorie_daily_intake,
                "carbs": 0.60 * my_calorie_daily_intake,
                "sodium": 2000,
                "cholesterol": 200,
                "penyakit": "diabetes",
            }
        elif "jantung" in list_penyakit:
            problem.addConstraint(
                lambda breakfast, lunch, dinner: filter_based_on_disease(
                    breakfast,
                    lunch,
                    dinner,
                    my_calorie_daily_intake,
                    "cardiovascular",
                    {
                        "breakfast": breakfast_category,
                        "lunch": lunch_category,
                        "dinner": dinner_category,
                    },
                    nutri_target,
                ),
                ("breakfast", "lunch", "dinner"),
            )

            max_nutrition = {
                "calorie": my_calorie_daily_intake,
                "fat": 0.25 * my_calorie_daily_intake,
                "protein": 0.15 * my_calorie_daily_intake,
                "carbs": 0.60 * my_calorie_daily_intake,
                "sodium": 2000,
                "cholesterol": 200,
                "penyakit": "jantung",
            }
        elif "hipertensi" in list_penyakit:
            problem.addConstraint(
                lambda breakfast, lunch, dinner: filter_based_on_disease(
                    breakfast,
                    lunch,
                    dinner,
                    my_calorie_daily_intake,
                    "hypertension",
                    {
                        "breakfast": breakfast_category,
                        "lunch": lunch_category,
                        "dinner": dinner_category,
                    },
                    nutri_target,
                ),
                ("breakfast", "lunch", "dinner"),
            )

            max_nutrition = {
                "calorie": my_calorie_daily_intake,
                "fat": 0.25 * my_calorie_daily_intake,
                "protein": 0.15 * my_calorie_daily_intake,
                "carbs": 0.60 * my_calorie_daily_intake,
                "sodium": 1000,
                "cholesterol": 200,
                "penyakit": "hipertensi",
            }
        elif "fat" in nutri_target:
            problem.addConstraint(
                lambda breakfast, lunch, dinner: filter_based_on_disease(
                    breakfast,
                    lunch,
                    dinner,
                    nutri_target["calories"],
                    "target_plan",
                    {
                        "breakfast": breakfast_category,
                        "lunch": lunch_category,
                        "dinner": dinner_category,
                    },
                    nutri_target,
                ),
                ("breakfast", "lunch", "dinner"),
            )

            max_nutrition = {
                "calorie": nutri_target["calories"],
                "fat": nutri_target["fat"],
                "protein": nutri_target["protein"],
                "carbs": nutri_target["carbs"],
                "sodium": 2000,
                "cholesterol": 200,
                "penyakit": "Target Plan",
            }
        else:
            problem.addConstraint(
                lambda breakfast, lunch, dinner: filter_based_on_disease(
                    breakfast,
                    lunch,
                    dinner,
                    my_calorie_daily_intake,
                    "normal",
                    {
                        "breakfast": breakfast_category,
                        "lunch": lunch_category,
                        "dinner": dinner_category,
                    },
                    nutri_target,
                ),
                ("breakfast", "lunch", "dinner"),
            )

            max_nutrition = {
                "calorie": my_calorie_daily_intake,
                "fat": 0.25 * my_calorie_daily_intake,
                "protein": 0.15 * my_calorie_daily_intake,
                "carbs": 0.60 * my_calorie_daily_intake,
                "sodium": 2000,
                "cholesterol": 200,
                "penyakit": "normal",
            }

        meal_plan_recommendation = problem.getSolution()
        res = {
            "breakfast": {
                "Name": meal_plan_recommendation["breakfast"]["Name"],
                "RecipeCategory": meal_plan_recommendation["breakfast"][
                    "RecipeCategory"
                ],
                "Calories": meal_plan_recommendation["breakfast"]["Calories"],
                "FatContent": meal_plan_recommendation["breakfast"]["FatContent"],
                "SaturatedIngredientsFatContent": meal_plan_recommendation["breakfast"][
                    "SaturatedFatContent"
                ],
                "CholesterolContent": meal_plan_recommendation["breakfast"][
                    "CholesterolContent"
                ],
                "SodiumContent": meal_plan_recommendation["breakfast"]["SodiumContent"],
                "CarbohydrateContent": meal_plan_recommendation["breakfast"][
                    "CarbohydrateContent"
                ],
                "FiberContent": meal_plan_recommendation["breakfast"]["FiberContent"],
                "ProteinContent": meal_plan_recommendation["breakfast"][
                    "ProteinContent"
                ],
                "Ingredients": meal_plan_recommendation["breakfast"][
                    "RecipeIngredientParts"
                ],
            },
            "lunch": {
                "Name": meal_plan_recommendation["lunch"]["Name"],
                "RecipeCategory": meal_plan_recommendation["lunch"]["RecipeCategory"],
                "Calories": meal_plan_recommendation["lunch"]["Calories"],
                "FatContent": meal_plan_recommendation["lunch"]["FatContent"],
                "SaturatedFatContent": meal_plan_recommendation["lunch"][
                    "SaturatedFatContent"
                ],
                "CholesterolContent": meal_plan_recommendation["lunch"][
                    "CholesterolContent"
                ],
                "SodiumContent": meal_plan_recommendation["lunch"]["SodiumContent"],
                "CarbohydrateContent": meal_plan_recommendation["lunch"][
                    "CarbohydrateContent"
                ],
                "FiberContent": meal_plan_recommendation["lunch"]["FiberContent"],
                "ProteinContent": meal_plan_recommendation["lunch"]["ProteinContent"],
                "Ingredients": meal_plan_recommendation["lunch"][
                    "RecipeIngredientParts"
                ],
            },
            "dinner": {
                "Name": meal_plan_recommendation["dinner"]["Name"],
                "RecipeCategory": meal_plan_recommendation["dinner"]["RecipeCategory"],
                "Calories": meal_plan_recommendation["dinner"]["Calories"],
                "FatContent": meal_plan_recommendation["dinner"]["FatContent"],
                "SaturatedFatContent": meal_plan_recommendation["dinner"][
                    "SaturatedFatContent"
                ],
                "CholesterolContent": meal_plan_recommendation["dinner"][
                    "CholesterolContent"
                ],
                "SodiumContent": meal_plan_recommendation["dinner"]["SodiumContent"],
                "CarbohydrateContent": meal_plan_recommendation["dinner"][
                    "CarbohydrateContent"
                ],
                "FiberContent": meal_plan_recommendation["dinner"]["FiberContent"],
                "ProteinContent": meal_plan_recommendation["dinner"]["ProteinContent"],
                "Ingredients": meal_plan_recommendation["dinner"][
                    "RecipeIngredientParts"
                ],
            },
            "curr_date": date.today(),
            "max_nutrition": max_nutrition,
        }

        # return JsonResponse(data=res)

        return render(request, "meal-plan-recommendation.html", res)
    else:
        feature_tour = get_feature_tour(request)
        feature_tour_item = feature_tour["meal_plan_page"]
        return render(
            request,
            "meal-plan.html",
            {
                "food_categories": settings.FOOD_CATEGORIES,
                "food_names": settings.FOOD_NAME,
                "curr_date": date.today(),
                "feature_tour": feature_tour_item,
            },
        )


## AI Chatbot
@login_required(login_url="my_app:normal_login")
def get_chatbot_response(request):
    if request.method == "POST":
        body_unicode = request.body.decode("utf-8")
        body = json.loads(body_unicode)
        question = body["question"]
        chat_history = body["chatHistory"]
        chat_uuid = body["chatUUID"]
        custom_user = CustomUser.objects.get(user=request.user)

        riwayat_penyakit_db = RiwayatPenyakit.objects.filter(user=custom_user)

        riwayat_penyakit = ""
        for penyakit in riwayat_penyakit_db:
            riwayat_penyakit += penyakit.nama_penyakit + ", "
        answer, context = answer_pipeline(question, chat_history, riwayat_penyakit)
        # answer = set(answer.split("\n"))  # hapus duplicate answer
        # answer = "\n".join(answer) # kalau pakai ini jawaban chatbotnya gak urut

        if chat_uuid != 0:
            curr_chat_session = ChatSession.objects.get(id=chat_uuid)
            new_message = Message(
                prompt_content=question,
                chatbot_content=answer,
                context=context,
                chat_session=curr_chat_session,
            )
            new_message.save()

        new_session = {
            "id": 0,
            "created_at": 0,
        }
        session_title = ""
        if chat_uuid == 0:
            session_title = question.split(" ")[:5]
            session_title = " ".join(session_title).lstrip() + " ...."

            new_session = ChatSession(
                message_from=custom_user, session_title=session_title
            )
            new_session.save()

            new_message = Message(
                prompt_content=question,
                chatbot_content=answer,
                context=context,
                chat_session=new_session,
            )
            new_message.save()

            return JsonResponse(
                {
                    "chatbot_message": answer,
                    "context": context,
                    "new_session_id": new_session.id,
                    "new_session_created_at": new_session.created_at,
                    "new_session_title": session_title,
                }
            )

        return JsonResponse(
            {
                "chatbot_message": answer,
                "context": context,
                "new_session_id": 0,
                "new_session_created_at": 0,
                "new_session_title": session_title,
            }
        )


@login_required(login_url="my_app:normal_login")
def chatbot_page(request, id=None):
    custom_user = CustomUser.objects.get(user=request.user)

    chat_sessions = []
    chat_sessions_db = ChatSession.objects.filter(
        message_from=custom_user
    ).prefetch_related("message_set")
    for db_chat in chat_sessions_db:
        chat_sessions.append(db_chat)

    chat_sessions_object = []
    for chat in chat_sessions:
        chat_sessions_object.append(
            {
                "chat_session_id": chat.id,
                "created_at": chat.created_at,
                "session_title": chat.session_title,
            }
        )
    curr_chat_session = None
    if id != None:
        chat_session = ChatSession.objects.get(id=id)
        messages = Message.objects.filter(chat_session=chat_session)
        messages_content = []
        for message in messages:
            messages_content.append(
                {
                    "prompt_content": message.prompt_content,
                    "chatbot_content": message.chatbot_content,
                    "context": message.context,
                }
            )

        curr_chat_session = {
            "chat_session_id": chat_session.id,
            "created_at": chat_session.created_at,
            "messages": messages_content,
        }

    return render(
        request,
        "chatbot-med.html",
        {"chat_sessions": chat_sessions_object, "curr_chat_session": curr_chat_session},
    )


def get_health_condition_based_on_nutrition_val(nutrition_type, nutrition_value, user):

    user_age = user.age
    if nutrition_type == "BLOOD_SUGAR":
        # https://www.halodoc.com/kesehatan/diabetes?srsltid=AfmBOorAHmPVZ_SFLW0q8RSkk8xjPeMjfor1bEsOx3HHs1kgrNbvApR9

        if nutrition_value >= 126:
            return {"disease_prediction": "diabetes"}
        elif nutrition_value >= 100 and nutrition_value <= 125:
            return {"disease_prediction": "prediabetes"}
        else:
            return {"disease_prediction": "Normal"}

    if nutrition_type == "CHOLESTEROL":
        if user_age >= 20:
            if nutrition_value > 200:
                # https://www.siloamhospitals.com/en/informasi-siloam/artikel/kadar-kolesterol-normal
                return {"disease_prediction": "Penyakit Jantung dan stroke"}

            else:
                return {"disease_prediction": "Normal"}
        else:
            if nutrition_value > 170:
                # https://www.siloamhospitals.com/en/informasi-siloam/artikel/kadar-kolesterol-normal
                return {"disease_prediction": "Penyakit Jantung dan stroke"}

            else:
                return {"disease_prediction": "Normal"}
    if nutrition_type == "URIC_ACID":
        # asam urat
        # https://www.siloamhospitals.com/en/informasi-siloam/artikel/prosedur-cek-asam-urat
        if nutrition_value >= 7.0:
            return {"disease_prediction": "Serangan Asam Urat Dan Batu Ginjal"}

        else:
            return {"disease_prediction": "Normal"}
    if nutrition_type == "BLOOD_PRESSURE":
        # https://www.siloamhospitals.com/informasi-siloam/artikel/mengenal-tekanan-darah-normal
        if user_age >= 18:
            if nutrition_value > 120:
                return {"disease_prediction": "Hipertensi"}
            else:
                return {"disease_prediction": "Normal"}
        elif user_age < 18 and user_age >= 13:
            if nutrition_value > 128:
                return {"disease_prediction": "Hipertensi"}
            else:
                return {"disease_prediction": "Normal"}
        elif user_age >= 6 and user_age <= 12:
            if nutrition_value > 131:
                return {"disease_prediction": "Hipertensi"}
            else:
                return {"disease_prediction": "Normal"}
        elif user_age >= 3 and user_age < 6:
            if nutrition_value > 120:
                return {"disease_prediction": "Hipertensi"}
            else:
                return {"disease_prediction": "Normal"}
        else:
            if nutrition_value > 100:
                return {"disease_prediction": "Hipertensi"}
            else:
                return {"disease_prediction": "Normal"}


def nutrition_condition(
    nutrition_type, nutrition_value, user_body_info, penyakit_user, user
):
    calorie_recomm = calorie_intake(
        user_body_info.weight, user_body_info.height, user.age
    )
    if nutrition_type == "CALORIE":
        if nutrition_value < calorie_recomm:
            return "Normal"
        else:
            return "Too High"

    if (
        "diabetes" not in penyakit_user
        and "jantung" not in penyakit_user
        and "hipertensi" not in penyakit_user
    ):
        #  healthy user
        if nutrition_type == "PROTEIN":
            if nutrition_value <= 0.15 * calorie_recomm:
                return "Normal"
            else:
                return "Too High"
        elif nutrition_type == "FAT":
            if nutrition_value <= 0.25 * calorie_recomm:
                return "Normal"
            else:
                return "Too High"
        elif nutrition_type == "CARBOHYDRATE":
            if nutrition_value <= 0.60 * calorie_recomm:
                return "Normal"
            else:
                return "Too High"

    if "jantung" in penyakit_user or "hipertensi" in penyakit_user:
        if nutrition_type == "PROTEIN":
            if nutrition_value <= 0.15 * calorie_recomm:
                return "Normal"
            else:
                return "Too High"
        elif nutrition_type == "FAT":
            if nutrition_value <= 0.25 * calorie_recomm:
                return "Normal"
            else:
                return "Too High"
        elif nutrition_type == "CARBOHYDRATE":
            if nutrition_value <= 0.60 * calorie_recomm:
                return "Normal"
            else:
                return "Too High"

    if "diabetes" in penyakit_user:
        if nutrition_type == "PROTEIN":
            if nutrition_value <= 25 * user_body_info.weight * user.age * 1.65:
                return "Normal"
            else:
                return "Too High"
        elif nutrition_type == "FAT":
            if nutrition_value <= 0.25 * calorie_recomm:
                return "Normal"
            else:
                return "Too High"
        elif nutrition_type == "CARBOHYDRATE":
            if nutrition_value <= 0.60 * calorie_recomm:
                return "Normal"
            else:
                return "Too High"


def truncate(f, n):
    """Truncates/pads a float f to n decimal places without rounding"""
    s = "{}".format(f)
    if "e" in s or "E" in s:
        return "{0:.{1}f}".format(f, n)
    i, p, d = s.partition(".")
    return ".".join([i, (d + "0" * n)[:n]])


@login_required(login_url="my_app:normal_login")
def dashboard(request):
    current_date = date.today()
    person_data = NutritionProgress.objects.filter(user__user=request.user)
    try:
        kadar_data = BloodCodition.objects.filter(user__user=request.user).latest(
            "check_time"
        )
    except:
        kadar_data = None
    curr_kadar = {
        "BLOOD_SUGAR": 0,
        "BLOOD_SUGAR_DISEASE": "Normal",
        "CHOLESTEROL": 0,
        "CHOLESEROL_DISEASE": "Normal",
        "URIC_ACID": 0,
        "URIC_ACID_DISEASE": "Normal",
        "BLOOD_PRESSURE": 0,
        "BLOOD_PRESSURE_DISEASE": "Normal",
    }
    custom_user = CustomUser.objects.get(user=request.user)

    if kadar_data:
        curr_kadar = {
            "BLOOD_SUGAR": kadar_data.blood_sugar,
            "BLOOD_SUGAR_DISEASE": get_health_condition_based_on_nutrition_val(
                "BLOOD_SUGAR", kadar_data.blood_sugar, custom_user
            ),
            "CHOLESTEROL": kadar_data.cholesterol,
            "CHOLESEROL_DISEASE": get_health_condition_based_on_nutrition_val(
                "CHOLESTEROL", kadar_data.cholesterol, custom_user
            ),
            "URIC_ACID": kadar_data.uric_acid,
            "URIC_ACID_DISEASE": get_health_condition_based_on_nutrition_val(
                "URIC_ACID", kadar_data.uric_acid, custom_user
            ),
            "BLOOD_PRESSURE": kadar_data.blood_pressure,
            "BLOOD_PRESSURE_DISEASE": get_health_condition_based_on_nutrition_val(
                "BLOOD_PRESSURE", kadar_data.blood_pressure, custom_user
            ),
        }

    today_foods = person_data.filter(check_time__date=timezone.now().date())
    weekly_foods = person_data.filter(
        check_time__gte=timezone.now() - datetime.timedelta(days=7)
    )

    today_calory = today_foods.aggregate(Sum("calorie"))["calorie__sum"] or 0
    today_carbs = today_foods.aggregate(Sum("carbs"))["carbs__sum"] or 0
    today_protein = today_foods.aggregate(Sum("protein"))["protein__sum"] or 0
    today_fat = today_foods.aggregate(Sum("fat"))["fat__sum"] or 0
    weekly_calory = weekly_foods.aggregate(Sum("calorie"))["calorie__sum"] or 0
    weekly_carbs = weekly_foods.aggregate(Sum("carbs"))["carbs__sum"] or 0
    weekly_protein = weekly_foods.aggregate(Sum("protein"))["protein__sum"] or 0
    weekly_fat = weekly_foods.aggregate(Sum("fat"))["fat__sum"] or 0

    user_body_info = UserBodyInfo.objects.filter(
        custom_user__user=request.user
    ).order_by("-check_time")[0]
    penyakit_user_db = RiwayatPenyakit.objects.filter(user__user=request.user)
    riwayat_penyakit_user = []
    for penyakit in penyakit_user_db:
        riwayat_penyakit_user.append(penyakit.nama_penyakit)
    custom_user = CustomUser.objects.get(user=request.user)

    bmi = truncate(user_body_info.weight / ((user_body_info.height / 100) ** 2), 2)

    # bmi last week dan last year

    today = timezone.now().date()

    start_last_month = today.replace(month=today.month - 1)
    end_last_month = today

    one_week_ago = today - datetime.timedelta(days=7)

    last_month_entries = UserBodyInfo.objects.filter(
        check_time__range=(start_last_month, end_last_month)
    )

    body_last_month = last_month_entries.order_by("-check_time")[
        len(last_month_entries) - 1
    ]
    bmi_last_month = truncate(
        body_last_month.weight / ((body_last_month.height / 100) ** 2), 2
    )
    body_last_month = {
        "weight": body_last_month.weight,
        "height": body_last_month.height,
    }
    one_week_ago_entries = UserBodyInfo.objects.filter(
        check_time__range=(one_week_ago, today)
    )
    body_last_week = one_week_ago_entries.order_by("-check_time")[
        len(one_week_ago_entries) - 1
    ]
    bmi_last_week = truncate(
        body_last_week.weight / ((body_last_week.height / 100) ** 2), 2
    )
    body_last_week = {
        "weight": body_last_week.weight,
        "height": body_last_week.height,
    }

    user_body_info_dict = {
        "weight": user_body_info.weight,
        "height": user_body_info.height,
    }

    detailed_daily_nutrition = (
        NutritionProgress.objects.filter(user__user=request.user)
        .annotate(date=TruncDate("check_time"))
        .values("date")
        .annotate(
            total_calories=Sum("calorie"),
            total_carbs=Sum("carbs"),
            total_protein=Sum("protein"),
            total_fat=Sum("fat"),
        )
        .order_by("date")
    )

    history_kadar = (
        BloodCodition.objects.filter(user__user=request.user)
        .annotate(date=TruncDate("check_time"))
        .values("date")
        .annotate(
            avg_blood_sugar=Avg("blood_sugar"),
            avg_uric_acid=Avg("uric_acid"),
            avg_cholesterol=Avg("cholesterol"),
            avg_blood_pressure=Avg("blood_pressure"),
        )
        .order_by("date")
    )

    daily_nutrition_input_db = NutritionProgress.objects.filter(user__user=request.user)
    daily_nutrition_input = []
    for daily in daily_nutrition_input_db:
        daily_nutrition_input.append(
            [
                daily.check_time,
                daily.nutrition_name,
                daily.calorie,
                daily.carbs,
                daily.fat,
                daily.protein,
            ]
        )

    daily_nutrition = []
    for daily in detailed_daily_nutrition:
        daily_nutrition.append(
            {
                "date": daily["date"],
                "total_calories": daily["total_calories"],
                "total_carbs": daily["total_carbs"],
                "total_protein": daily["total_protein"],
                "total_fat": daily["total_fat"],
            }
        )

    #  blood condition history
    daily_kadar = []
    for daily in history_kadar:
        daily_kadar.append(
            {
                "date": daily["date"],
                "blood_sugar": daily["avg_blood_sugar"],
                "uric_acid": daily["avg_uric_acid"],
                "cholesterol": daily["avg_cholesterol"],
                "blood_pressure": daily["avg_blood_pressure"],
            }
        )

    target_plan = TargetPlan.objects.filter(user__user=request.user)
    nutri_target = {
        "calories": 0.0,
        "protein": 0.0,
        "carbs": 0.0,
        "fat": 0.0,
    }
    if target_plan.exists():
        target_plan = target_plan.first()
        nutri_target = {
            "calories": truncate(target_plan.target_calorie, 2),
            "protein": truncate(target_plan.target_protein, 2),
            "carbs": truncate(target_plan.target_carbs, 2),
            "fat": truncate(target_plan.target_fat, 2),
        }

    feature_tour = get_feature_tour(request)
    feature_tour_dashboard = feature_tour["dashboard_page"]

    context = {
        "today_foods": today_foods,
        "weekly_foods": weekly_foods,
        "today_calory": truncate(today_calory, 2),
        "today_protein": truncate(today_protein, 2),
        "today_carbs": truncate(today_carbs, 2),
        "today_fat": truncate(today_fat, 2),
        "weekly_calory": truncate(weekly_calory, 2),
        "weekly_protein": truncate(weekly_protein, 2),
        "weekly_carbs": truncate(weekly_carbs, 2),
        "weekly_fat": truncate(weekly_fat, 2),
        "calory_condition": nutrition_condition(
            "CALORIE", today_calory, user_body_info, riwayat_penyakit_user, custom_user
        ),
        "protein_condition": nutrition_condition(
            "PROTEIN", today_calory, user_body_info, riwayat_penyakit_user, custom_user
        ),
        "fat_condition": nutrition_condition(
            "FAT", today_calory, user_body_info, riwayat_penyakit_user, custom_user
        ),
        "carbo_condition": nutrition_condition(
            "CARBOHYDRATE",
            today_calory,
            user_body_info,
            riwayat_penyakit_user,
            custom_user,
        ),
        "kadar": curr_kadar,
        "body_info": user_body_info,
        "bmi": bmi,
        "daily_nutrition": daily_nutrition,
        "daily_nutrition_input": daily_nutrition_input,
        "curr_date": current_date,
        "nutri_target": nutri_target,
        "bmi_last_month": bmi_last_month,
        "bmi_last_week": bmi_last_week,
        "body_last_month": body_last_month,
        "body_last_week": body_last_week,
        "user_body_info_dict": user_body_info_dict,
        "curr_bmi": bmi,  # buat selector bmi jangan dihapus,
        "daily_kadar": daily_kadar,
        "feature_tour": feature_tour_dashboard,
    }

    return render(request, "dashboard_cantik.html", context=context)


@login_required(login_url="my_app:normal_login")
def dashboard_cantik(request):
    person_data = NutritionProgress.objects.filter(user__user=request.user)
    today_foods = person_data.filter(check_time__date=timezone.now().date())
    weekly_foods = person_data.filter(
        check_time__gte=timezone.now() - datetime.timedelta(days=7)
    )
    today_calory = today_foods.aggregate(Sum("calorie"))["calorie__sum"]
    today_carbs = today_foods.aggregate(Sum("carbs"))["carbs__sum"]
    today_protein = today_foods.aggregate(Sum("protein"))["protein__sum"]
    today_fat = today_foods.aggregate(Sum("fat"))["fat__sum"]
    weekly_calory = weekly_foods.aggregate(Sum("calorie"))["calorie__sum"]
    weekly_carbs = weekly_foods.aggregate(Sum("carbs"))["carbs__sum"]
    weekly_protein = weekly_foods.aggregate(Sum("protein"))["protein__sum"]
    weekly_fat = weekly_foods.aggregate(Sum("fat"))["fat__sum"]
    context = {
        "today_foods": today_foods,
        "weekly_foods": weekly_foods,
        "today_calory": today_calory,
        "today_protein": today_protein,
        "today_carbs": today_carbs,
        "today_fat": today_fat,
        "weekly_calory": weekly_calory,
        "weekly_protein": weekly_protein,
        "weekly_carbs": weekly_carbs,
        "weekly_fat": weekly_fat,
    }
    return render(request, "dashboard_cantik.html", context=context)


def delete_nutrition(request, pk):
    if request.method == "POST":
        nutrition = NutritionProgress.objects.get(id=pk)
        if request.user != nutrition.user.user:
            return redirect("my_app:dashboard")
        nutrition.delete()
        return redirect("my_app:dashboard")


def logout_view(request):
    logout(request)
    return redirect("my_app:home")


@login_required(login_url="my_app:normal_login")
def riwayat_penyakit(request):
    current_date = date.today()
    custom_user = CustomUser.objects.get(user=request.user)
    riwayat_user = RiwayatPenyakit.objects.filter(user=custom_user)
    riwayat_penyakit = []
    for penyakit in riwayat_user:
        riwayat_penyakit.append(
            {
                "nama_penyakit": penyakit.nama_penyakit,
                "deskripsi_penyakit": penyakit.deskripsi_penyakit,
                "check_time": penyakit.check_time,
            }
        )

    feature_tour = get_feature_tour(request)
    feature_tour_item = feature_tour["medical_history_page"]
    context = {
        "riwayat_user": riwayat_penyakit,
        "curr_date": current_date,
        "feature_tour": feature_tour_item,
    }
    return render(request, "riwayat.html", context=context)


@login_required(login_url="my_app:normal_login")
def add_riwayat(request):
    if request.method == "POST":
        nama_penyakit = request.POST.get("nama_penyakit")
        deskripsi_penyakit = request.POST.get("deskripsi_penyakit")
        custom_user = CustomUser.objects.get(user=request.user)
        riwayat = RiwayatPenyakit(
            nama_penyakit=nama_penyakit,
            deskripsi_penyakit=deskripsi_penyakit,
            user=custom_user,
        )
        riwayat.save()
        return redirect("my_app:riwayat")

    feature_tour = get_feature_tour(request)
    feature_tour_item = feature_tour["add_medical_history_page"]
    return render(
        request,
        "add_riwayat_new.html",
        {"feature_tour": feature_tour_item, "curr_date": date.today()},
    )


def delete_riwayat(request, pk):
    if request.method == "POST":
        custom_user = CustomUser.objects.get(user=request.user)
        riwayat_user = RiwayatPenyakit.objects.get(id=pk)
        if riwayat_user.user != custom_user:
            return redirect("my_app:riwayat")
        riwayat_user.delete()
        return redirect("my_app:riwayat")


@login_required(login_url="my_app:normal_login")
def add_nutrition(request):
    current_date = date.today()
    if request.method == "POST":
        nutrition_name = request.POST.get("nutrition_name")
        calorie = float(request.POST.get("calorie"))
        carbs = float(request.POST.get("carbs"))
        protein = float(request.POST.get("protein"))
        fat = float(request.POST.get("fat"))
        custom_user = CustomUser.objects.get(user=request.user)
        nutrisi = NutritionProgress(
            nutrition_name=nutrition_name,
            calorie=calorie,
            carbs=carbs,
            protein=protein,
            fat=fat,
            user=custom_user,
        )
        nutrisi.save()
        if request.POST.get("redirect"):
            return redirect("my_app:dashboard")
        return redirect("my_app:dashboard")

    feature_tour = get_feature_tour(request)
    feature_tour_item = feature_tour["add_nutrition_page"]
    return render(
        request,
        "add_nutrition_new.html",
        {"curr_date": current_date, "feature_tour": feature_tour_item},
    )


@login_required(login_url="my_app:normal_login")
def add_target(request):
    current_date = date.today()

    if request.method == "POST":

        calorie = request.POST.get("calorie_input")
        carbs = request.POST.get("carbo_input")
        protein = request.POST.get("protein_input")
        fat = request.POST.get("fat_input")
        target = TargetPlan.objects.create(
            target_calorie=calorie,
            target_carbs=carbs,
            target_protein=protein,
            target_fat=fat,
            user=CustomUser.objects.get(user=request.user),
        )
        target.save()
        return redirect("my_app:dashboard")

    feature_tour = get_feature_tour(request)
    feature_tour_item = feature_tour["add_target_page"]
    return render(
        request,
        "target-plan.html",
        {"curr_date": current_date, "feature_tour": feature_tour_item},
    )


@login_required(login_url="my_app:normal_login")
def add_blood_condition(request):
    if request.method == "POST":
        blood_sugar = request.POST.get("blood_sugar")
        uric_acid = request.POST.get("uric_acid")
        cholesterol = request.POST.get("cholesterol")
        blood_pressure = request.POST.get("blood_pressure")
        blood_condition = BloodCodition.objects.create(
            user=CustomUser.objects.get(user=request.user),
            blood_sugar=blood_sugar,
            uric_acid=uric_acid,
            cholesterol=cholesterol,
            blood_pressure=blood_pressure,
        )
        blood_condition.save()
        return redirect("my_app:dashboard")
    try:
        blood_data = BloodCodition.objects.filter(user__user=request.user).latest(
            "check_time"
        )
    except:
        blood_data = {
            "blood_sugar": 0,
            "uric_acid": 0,
            "cholesterol": 0,
            "blood_pressure": 0,
        }
    feature_tour = get_feature_tour(request)
    feature_tour_item = feature_tour["add_blood_condition_page"]
    context = {"blood_data": blood_data, "feature_tour": feature_tour_item}
    

    return render(request, "add_blood.html", context=context)


@login_required(login_url="my_app:normal_login")
def feature_tour(request):
    if request.method == "POST":
        user = CustomUser.objects.get(user=request.user)
        user_feature_tour = FeatureTour.objects.filter(user=user)
        if user_feature_tour.exists() == False:
            new_feature_tour = FeatureTour(user=user)
            new_feature_tour.save()
            return JsonResponse({"message": "Success save new feature tour user"})
        else:
            user_feature_tour = user_feature_tour.get()

            body_unicode = request.body.decode("utf-8")
            body_data = json.loads(body_unicode)
            item = body_data.get("item")

            match item:
                case "dashboard_page":
                    user_feature_tour.dashboard_page = True
                case "food_nutrition_page":
                    user_feature_tour.food_nutrition_page = True
                case "medical_history_page":
                    user_feature_tour.medical_history_page = True
                case "add_nutrition_page":
                    user_feature_tour.add_nutrition_page = True
                case "add_medical_history_page":
                    user_feature_tour.add_medical_history_page = True
                case "meal_plan_page":
                    user_feature_tour.meal_plan_page = True
                case "record_bmi_page":
                    user_feature_tour.record_bmi_page = True
                case "add_target_page":
                    user_feature_tour.add_target_page = True
                case "add_blood_condition_page":
                    user_feature_tour.add_blood_condition_page = True
            user_feature_tour.save()

            return JsonResponse({"message": "Success update feature tour user"})


def get_feature_tour(request):
    feature_tour_user = FeatureTour.objects.filter(user__user=request.user)
    if feature_tour_user.exists() == False:
        user= CustomUser.objects.get(user=request.user)
        feature_tour_user = FeatureTour(user=user)
        feature_tour_user.save()
    else:
        feature_tour_user = feature_tour_user.get()
        
        
    feature_tour_data = {
        "dashboard_page": feature_tour_user.dashboard_page,
        "food_nutrition_page": feature_tour_user.food_nutrition_page,
        "medical_history_page": feature_tour_user.medical_history_page,
        "add_nutrition_page": feature_tour_user.add_nutrition_page,
        "add_medical_history_page": feature_tour_user.add_medical_history_page,
        "meal_plan_page": feature_tour_user.meal_plan_page,
        "record_bmi_page": feature_tour_user.record_bmi_page,
        "add_target_page": feature_tour_user.add_target_page,
        "add_blood_condition_page": feature_tour_user.add_blood_condition_page,
    }
    return feature_tour_data


"""
def get_chatbot_response():
    return


def chatbot_page():
    return
"""
