"""
Django settings for welllahh project.

Generated by 'django-admin startproject' using Django 5.1.

For more information on this file, see
https://docs.djangoproject.com/en/5.1/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/5.1/ref/settings/
"""

import os
import pickle
import time
import urllib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

# https://drive.google.com/file/d/1j6LH7JusFs2TMljP05mBZvECeHIxbrYk/view?usp=sharing
# https://drive.google.com/file/d/11KVVZHT3cSDXeu7yPR7dGJ2NBm0IqsFX/view?usp=sharing
# https://drive.google.com/file/d/1US8wd4AsafsVGDl-HWcD0AzqTm-l-FTK/view?usp=sharing
# https://drive.google.com/file/d/1Q3vjH09Sk0KTNQyT_BRQw3mofP-yw30e/view?usp=sharing

if os.path.exists("./recipes_sedikit.csv") == False:
    print("downloading pickle & csv file for food recommendation system...")
    url = "https://drive.google.com/uc?export=download&id=1j6LH7JusFs2TMljP05mBZvECeHIxbrYk"
    path = "count_matrix_name.pkl"
    urllib.request.urlretrieve(url, path)

    url = "https://drive.google.com/uc?export=download&id=11KVVZHT3cSDXeu7yPR7dGJ2NBm0IqsFX"
    path = "search_count_vec.pkl"
    urllib.request.urlretrieve(url, path)

    url = "https://drive.google.com/uc?export=download&id=1Q3vjH09Sk0KTNQyT_BRQw3mofP-yw30e"
    path = "recipes_sedikit.csv"
    urllib.request.urlretrieve(url, path)
    print("download complete!")

print("loading food image classification model....")
INDOFOOD_IMAGE_MODEL = tf.keras.models.load_model("best_model_86.keras")
print("selesai food image classification model....")


with open("recommendation_cosine_sim.pkl", "rb") as file:
    FOOD_COSINE_SIM = pickle.load(file)
with open("search_count_vec.pkl", "rb") as file:
    SEARCH_COUNT_VEC = pickle.load(file)

with open("count_matrix_name.pkl", "rb") as file:
    SEARCH_MATRIX_VEC = pickle.load(file)

path2 = "./recipes_sedikit.csv"

FOOD_CATEGORIES = [
    "Frozen Desserts",
    "Soy/Tofu",
    "Pie",
    "Chicken",
    "Vegetable",
    "Chicken Breast",
    "Dessert",
    "Beverages",
    "Southwestern U.S.",
    "Stew",
    "Lactose Free",
    "Weeknight",
    "Yeast Breads",
    "Cheesecake",
    "Sauces",
    "High In...",
    "Brazilian",
    "Brown Rice",
    "Oranges",
    "Free Of...",
    "Low Protein",
    "Potato",
    "Halibut",
    "Lamb/Sheep",
    "Breads",
    "Spaghetti",
    "Lunch/Snacks",
    "Beans",
    "Very Low Carbs",
    "Pineapple",
    "Whole Chicken",
    "Low Cholesterol",
    "< 30 Mins",
    "Chicken Livers",
    "Coconut",
    "< 60 Mins",
    "Poultry",
    "Quick Breads",
    "Steak",
    "Healthy",
    "Pork",
    "Scones",
    "Lobster",
    "Rice",
    "Punch Beverage",
    "Drop Cookies",
    "Spreads",
    "Bar Cookie",
    "Crab",
    "Pears",
    "Cheese",
    "Savory Pies",
    "Breakfast",
    "Chowders",
    "Candy",
    "Chutneys",
    "White Rice",
    "Tex Mex",
    "German",
    "Meat",
    "Fruit",
    "European",
    "Smoothies",
    "Hungarian",
    "Onions",
    "New Zealand",
    "Indonesian",
    "Lentil",
    "Summer",
    "Long Grain Rice",
    "Southwest Asia (middle East)",
    "Spanish",
    "Jellies",
    "Gelatin",
    "Chicken Thigh & Leg",
    "Cauliflower",
    "Tuna",
    "Citrus",
    "Apple",
    "Berries",
    "Greek",
    "Peppers",
    "Clear Soup",
    "Mexican",
    "Raspberries",
    "Crawfish",
    "Beef Organ Meats",
    "Strawberry",
    "Shakes",
    "Short Grain Rice",
    "Salad Dressings",
    "Manicotti",
    "Spicy",
    "< 15 Mins",
    "Cajun",
    "Sourdough Breads",
    "Oven",
    "Microwave",
    "Asian",
    "Corn",
    "Melons",
    "Swiss",
    "Papaya",
    "Broil/Grill",
    "No Cook",
    "< 4 Hours",
    "Roast",
    "Curries",
    "Orange Roughy",
    "Thai",
    "Canadian",
    "Bass",
    "Veal",
    "Medium Grain Rice",
    "Japanese",
    "Penne",
    "Mussels",
    "Elk",
    "Colombian",
    "High Protein",
    "Black Beans",
    "Rabbit",
    "Caribbean",
    "Turkish",
    "Kid Friendly",
    "Christmas",
    "For Large Groups",
    "One Dish Meal",
    "Whole Turkey",
    "Chinese",
    "Roast Beef",
    "Grains",
    "Russian",
    "Yam/Sweet Potato",
    "Native American",
    "Trout",
    "Gumbo",
    "Vegan",
    "African",
    "Meatballs",
    "Whole Duck",
    "Scandinavian",
    "Greens",
    "Catfish",
    "Dehydrator",
    "Duck Breasts",
    "Ham",
    "Stocks",
    "Savory",
    "Vietnamese",
    "Stir Fry",
    "Polish",
    "Deer",
    "Wild Game",
    "Pheasant",
    "No Shell Fish",
    "Spring",
    "Collard Greens",
    "Quail",
    "Canning",
    "Moroccan",
    "Pressure Cooker",
    "Squid",
    "Winter",
    "Pasta Shells",
    "Danish",
    "Lebanese",
    "Creole",
    "Tarts",
    "Spinach",
    "Homeopathy/Remedies",
    "Austrian",
    "Thanksgiving",
    "Moose",
    "Swedish",
    "High Fiber",
    "Norwegian",
    "Kosher",
    "Australian",
    "Meatloaf",
]


FOOD_DF = pd.read_csv(path2)

INDICES = pd.Series(FOOD_DF.index, index=FOOD_DF["Name"])

INDOFOOD_NUTRITIONS_DF = pd.read_csv("indofood_with_its_nutritions.csv")


df2 = pd.read_csv("all_food_names.csv")

INDOFOOD_IMAGE_LABELS = df2.values.tolist()

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent


# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/5.1/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = "django-insecure-#zrpkqrmd6du&!8^v5c6nu#^iuvhz^_3z*-)bz4t(q^5*@l7wf"

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = []


# Application definition

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "my_app",
    'mathfilters'
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "welllahh.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [os.path.join(BASE_DIR, "templates")],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "welllahh.wsgi.application"


# Database
# https://docs.djangoproject.com/en/5.1/ref/settings/#databases

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}


# Password validation
# https://docs.djangoproject.com/en/5.1/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.CommonPasswordValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.NumericPasswordValidator",
    },
]


# Internationalization
# https://docs.djangoproject.com/en/5.1/topics/i18n/

LANGUAGE_CODE = "en-us"

TIME_ZONE = "UTC"

USE_I18N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/5.1/howto/static-files/

STATIC_URL = "static/"

# Default primary key field type
# https://docs.djangoproject.com/en/5.1/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"


MEDIA_ROOT = os.path.join(BASE_DIR, "media")
MEDIA_URL = "/media/"
